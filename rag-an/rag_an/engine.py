"""Recherche (sémantique, lexicale, hybride), reranking et mise en forme des extraits pour l'agent.

La fusion par rang (RRF) sert seulement à classer : un score RRF n'est pas une pertinence. Les scores affichés
sont ceux d'origine : cosinus Solon (0-1), BM25 (non borné) et probabilité du reranker (0-1).
"""
from __future__ import annotations

import re
from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from .communs import hidden_metadata, norm_name, token_counter
from .config import Settings, get_settings
from .lexical import BM25Index, load_bm25
from .models import detect_device, load_local_first
from .stats import lire_periode
from .stores import get_vector_store

SYSTEM_PROMPT = (
    "Tu es un assistant d'analyse des débats de l'Assemblée nationale française. "
    "Tu réponds toujours en français, de façon concise et factuelle."
)

# Pas de phrase de refus imposée mot pour mot : un modèle 7B tend à la réciter même quand il a la réponse.
RULES = (
    "RÈGLES IMPÉRATIVES :\n"
    "1. Réponds UNIQUEMENT à partir de ces extraits, sans aucune connaissance externe.\n"
    "2. Commence directement par la réponse. Un chiffre ou un fait avancé par un orateur répond à la "
    "question : donne-le en précisant qui l'avance. Uniquement si aucun extrait ne traite du sujet, "
    "dis-le en une phrase.\n"
    "3. N'attribue un rôle (présidente, ministre, rapporteur…) que s'il est écrit dans l'extrait.\n"
    "4. Cite tes sources entre crochets avec leur numéro, ex. [1] ou [2][4], après chaque fait : le numéro de "
    "l'extrait où ce fait est écrit, dont l'orateur est celui qui l'avance.\n"
    "5. L'auteur d'un propos est l'orateur indiqué devant ce propos ; ne confonds pas les orateurs entre eux. "
    "Pour la position d'un groupe ou d'un parti, ne retiens que les propos de ses membres, et non ceux des "
    "orateurs qui lui répondent ou le critiquent.\n"
    "6. N'ajoute aucune étiquette ni commentaire sur ta propre réponse."
)

SCORE_KEYS = ("score_semantique", "rang_semantique", "score_lexical", "rang_lexical", "score_reranker")
RRF_K = 60  # constante usuelle de la Reciprocal Rank Fusion


def scored_node(node, scores: dict) -> NodeWithScore:
    """Copie du chunk avec ses scores en métadonnées, affichés à l'utilisateur mais cachés au LLM."""
    metadata = {**node.metadata, **scores}
    hidden = hidden_metadata(metadata)
    for keys in hidden.values():
        keys += [k for k in SCORE_KEYS if k not in keys]
    return NodeWithScore(node=TextNode(id_=node.node_id, text=node.text, metadata=metadata, **hidden))


class SearchRetriever(BaseRetriever):
    """Recherche sémantique (pgvector), lexicale (BM25) ou les deux fusionnées par rang (RRF).

    Filtres facultatifs : `periode` = (début, fin) en dates, `orateur` = sous-chaîne du nom.
    """

    def __init__(self, index: VectorStoreIndex, bm25: BM25Index | None, k: int, mode: str = "hybride",
                 periode=None, orateur: str | None = None):
        super().__init__()
        self.index, self.bm25, self.k, self.mode = index, bm25, k, mode
        self.periode, self.orateur = periode, orateur

    def _semantique(self, qb: QueryBundle) -> list[NodeWithScore]:
        filters = []
        if self.periode:
            a, b = self.periode
            filters += [MetadataFilter(key="date_iso", value=a.isoformat(), operator=FilterOperator.GTE),
                        MetadataFilter(key="date_iso", value=b.isoformat(), operator=FilterOperator.LTE)]
        if self.orateur:  # norm_name ne garde que [a-z0-9 ] : pas d'apostrophe qui casserait le SQL
            filters.append(MetadataFilter(key="orateurs_norm", value=norm_name(self.orateur),
                                          operator=FilterOperator.TEXT_MATCH))
        retriever = self.index.as_retriever(similarity_top_k=self.k,
                                            filters=MetadataFilters(filters=filters) if filters else None)
        return [scored_node(n.node, {"score_semantique": round(float(n.score), 3), "rang_semantique": rank})
                for rank, n in enumerate(retriever.retrieve(qb), 1)]

    def _lexicale(self, qb: QueryBundle) -> list[NodeWithScore]:
        found = self.bm25.search(qb.query_str, self.k, self.periode, self.orateur)
        return [scored_node(TextNode(id_=r["id"], text=r["text"], metadata=r["metadata"]),
                            {"score_lexical": round(score, 2), "rang_lexical": rank})
                for rank, (score, r) in enumerate(found, 1)]

    def _retrieve(self, qb: QueryBundle) -> list[NodeWithScore]:
        if self.mode == "semantique":
            return self._semantique(qb)
        if self.mode == "lexicale":
            return self._lexicale(qb)
        # hybride : somme des 1 / (RRF_K + rang) de chaque recherche qui a trouvé le chunk
        fused: dict[str, NodeWithScore] = {}
        for results in (self._semantique(qb), self._lexicale(qb)):
            for rank, n in enumerate(results, 1):
                cur = fused.setdefault(n.node.node_id, NodeWithScore(node=n.node, score=0.0))
                cur.score += 1 / (RRF_K + rank)
                cur.node.metadata.update({k: v for k, v in n.node.metadata.items() if k in SCORE_KEYS})
        return sorted(fused.values(), key=lambda n: n.score, reverse=True)



@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    """Charge le reranker une seule fois par processus (un Reranker est recréé à chaque recherche)."""
    from sentence_transformers import CrossEncoder

    return load_local_first(CrossEncoder, model_name, device=detect_device())


class Reranker(BaseNodePostprocessor):
    """Cross-encoder : reclasse les `depth` premiers candidats et note `score_reranker` (0-1)."""

    model_name: str
    depth: int = 60

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        nodes = nodes[:self.depth]
        if not nodes:
            return nodes
        pairs = [(query_bundle.query_str, n.node.get_content()) for n in nodes]
        scores = _cross_encoder(self.model_name).predict(pairs)
        for n, p in zip(nodes, scores, strict=True):  # bge-reranker renvoie déjà une probabilité (sigmoïde appliquée)
            n.node.metadata["score_reranker"] = round(float(p), 3)
            n.score = float(p)
        return sorted(nodes, key=lambda n: n.score, reverse=True)


class TokenBudget(BaseNodePostprocessor):
    """Garde les extraits les mieux classés jusqu'à `budget` tokens (et au plus `max_nodes`)."""

    budget: int
    max_nodes: int
    model_name: str

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        count = token_counter(self.model_name)
        out, used = [], 0
        for n in nodes:
            t = count(n.node.get_content(metadata_mode=MetadataMode.LLM))
            if out and (used + t > self.budget or len(out) >= self.max_nodes):
                break
            out.append(n)
            used += t
        return out


def _piece(doc_id: str) -> tuple[str, int]:
    """« CRS…#50-3 » -> (« CRS…#50 », 3) : les morceaux d'un même discours partagent la base."""
    m = re.fullmatch(r"(.+#\d+)(?:-(\d+))?", doc_id or "")
    return (m[1], int(m[2] or 0)) if m else (doc_id, 0)


def _join_pieces(texts: list[str | None]) -> str:
    """Recolle les morceaux d'un discours sans « (suite) : » ni recouvrement ; None devient « […] »."""
    out = ""
    for t in texts:
        if t is None:
            out += "\n[…]"
            continue
        if not out:
            out = t
            continue
        body = t.split(" (suite) : ", 1)[-1]
        # recouvrement : le début du morceau répète la fin du précédent
        n = next((k for k in range(min(len(out), len(body)), 15, -1) if out.endswith(body[:k])), 0)
        out += "\n" + body[n:].lstrip()
    return out


class Passages(BaseNodePostprocessor):
    """Rend au LLM des passages lisibles plutôt que des fragments.

    Les petits chunks retrouvent mieux mais se lisent mal : les morceaux retenus d'un même discours sont recollés,
    puis le budget restant ajoute aux passages les mieux classés le morceau suivant, à défaut le précédent.
    """

    budget: int
    model_name: str
    bm25_dir: str

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        bm25 = load_bm25(self.bm25_dir)
        if bm25 is None or not nodes:
            return nodes
        count = token_counter(self.model_name)
        groups: dict[str, dict] = {}  # base du discours -> {"best": nœud le mieux classé, "idx": morceaux}
        order: list[str] = []
        for n in nodes:
            doc_id = n.node.metadata.get("doc_id") or n.node.node_id
            base, k = _piece(doc_id) if n.node.metadata.get("type") == "debat" else (doc_id, None)
            if base not in groups:
                groups[base] = {"best": n, "idx": set()}
                order.append(base)
            if k is not None:
                groups[base]["idx"].add(k)

        def text_of(base: str, idx: set[int]) -> str:
            if not idx:
                return groups[base]["best"].node.text
            lo, hi = min(idx), max(idx)
            texts = []
            for k in range(lo, hi + 1):
                r = bm25.by_id.get(base if k == 0 else f"{base}-{k}")
                if k in idx or (r and k - 1 in idx and k + 1 in idx):  # un trou d'un morceau est comblé
                    texts.append(r["text"] if r else None)
                elif texts and texts[-1] is not None:
                    texts.append(None)
            return _join_pieces(texts)

        # en-tête de métadonnées vu par le LLM : compté une fois par passage
        head = {b: count(g["best"].node.get_content(metadata_mode=MetadataMode.LLM)) - count(g["best"].node.text)
                for b, g in groups.items()}
        used = sum(head[b] + count(text_of(b, g["idx"])) for b, g in groups.items())
        for base in order:  # budget restant : un morceau voisin pour les passages les mieux classés
            idx = groups[base]["idx"]
            if not idx:
                continue
            for k in (max(idx) + 1, min(idx) - 1):
                if k < 0 or (base if k == 0 else f"{base}-{k}") not in bm25.by_id:
                    continue
                extra = count(text_of(base, idx | {k})) - count(text_of(base, idx))
                if used + extra <= self.budget:
                    idx.add(k)
                    used += extra
                    break

        out = []
        for base in order:
            best = groups[base]["best"]
            node = best.node
            text = text_of(base, groups[base]["idx"])
            if text != node.text:
                node = TextNode(id_=node.node_id, text=text, metadata=dict(node.metadata),
                                excluded_embed_metadata_keys=list(node.excluded_embed_metadata_keys),
                                excluded_llm_metadata_keys=list(node.excluded_llm_metadata_keys))
            out.append(NodeWithScore(node=node, score=best.score))
        return out


class SourceNumbering(BaseNodePostprocessor):
    """Numérote les extraits (métadonnée « source_n ») pour que le LLM puisse les citer [n]."""

    start: int = 1

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        for i, n in enumerate(nodes, self.start):
            n.node.metadata["source_n"] = i
        return nodes


def build_index(s: Settings | None = None) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(get_vector_store(s or get_settings()))


def build_retriever(index: VectorStoreIndex, s: Settings | None = None, mode: str = "hybride",
                    date_ou_periode: str | None = None, orateur: str | None = None) -> SearchRetriever:
    s = s or get_settings()
    bm25 = load_bm25(s.bm25_dir)
    if bm25 is None:
        mode = "semantique"  # index BM25 pas encore construit
    # 3 × top_k candidats : marge pour le reranker et le budget de tokens
    return SearchRetriever(index, bm25, k=3 * s.top_k, mode=mode, periode=lire_periode(date_ou_periode, s),
                           orateur=orateur)


def build_postprocessors(s: Settings | None = None, numbering_start: int = 1) -> list[BaseNodePostprocessor]:
    s = s or get_settings()
    pps: list[BaseNodePostprocessor] = []
    if s.rerank_model:
        pps.append(Reranker(model_name=s.rerank_model, depth=3 * s.top_k))
    pps.append(TokenBudget(budget=s.context_token_budget, max_nodes=s.top_k, model_name=s.embed_model))
    pps.append(Passages(budget=s.context_token_budget, model_name=s.embed_model, bm25_dir=s.bm25_dir))
    pps.append(SourceNumbering(start=numbering_start))
    return pps


def score_label(m: dict) -> str:
    parts = []
    if "score_reranker" in m:
        parts.append(f"reranker {m['score_reranker']:.2f}")
    if "score_semantique" in m:
        parts.append(f"sémantique {m['score_semantique']:.2f} (n°{m['rang_semantique']})")
    if "score_lexical" in m:
        parts.append(f"BM25 {m['score_lexical']:.1f} (n°{m['rang_lexical']})")
    return " · ".join(parts)


def source_label(m: dict) -> str:
    if m.get("type") == "declaration_politique":
        return f"Déclaration politique — {m.get('groupe')}"
    orateurs = (m.get("orateurs") or "").split(" ; ")
    who = " ; ".join(orateurs[:3]) + (f" et {len(orateurs) - 3} autre(s)" if len(orateurs) > 3 else "")
    return f"{m.get('date_seance', '')} — {m.get('sujet', '')} — {who}"


def format_sources(source_nodes: list[NodeWithScore]) -> list[dict]:
    out = []
    for n in source_nodes:
        m = n.node.metadata
        if "source_n" not in m:
            continue
        out.append({
            "n": m["source_n"],
            "label": source_label(m),
            "scores": score_label(m),
            "doc_id": n.node.ref_doc_id or m.get("doc_id"),
        })
    return out
