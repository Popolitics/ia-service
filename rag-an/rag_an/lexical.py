"""Recherche lexicale BM25 (bm25s, racinisation française Snowball), en complément de la recherche sémantique :
termes exacts (noms propres, sigles, numéros d'amendement) que les embeddings rapprochent mal. L'index est
construit à l'ingestion à partir des chunks de PostgreSQL (mêmes identifiants) et sauvegardé dans `index_bm25/`.
"""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_FRENCH

from .config import Settings, get_settings
from .communs import norm_name


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


STEMMER = Stemmer.Stemmer("french")
STOPWORDS = sorted({strip_accents(w) for w in STOPWORDS_FRENCH})


def tokenize(texts: list[str]):
    return bm25s.tokenize([strip_accents(t.replace("’", "'")) for t in texts], stopwords=STOPWORDS,
                          stemmer=STEMMER.stemWords, show_progress=False)


def index_text(r: dict) -> str:
    """Texte indexé : le chunk + son sujet et ses orateurs (« Retailleau Marseille » doit le retrouver)."""
    extra = " ".join(str(r["metadata"].get(k) or "") for k in ("sujet", "orateurs", "groupe", "document"))
    return f"{extra}\n{r['text']}"


class BM25Index:
    """Index BM25 + la liste des chunks indexés ({"id", "text", "metadata"}, dans le même ordre)."""

    def __init__(self, bm25: bm25s.BM25, records: list[dict]):
        self.bm25, self.records = bm25, records
        self.by_id = {r["id"]: r for r in records}  # chunks voisins d'un extrait (engine.Passages)

    @classmethod
    def build(cls, records: list[dict]) -> BM25Index:
        bm25 = bm25s.BM25()
        bm25.index(tokenize([index_text(r) for r in records]), show_progress=False)
        return cls(bm25, records)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.bm25.save(directory, show_progress=False)
        # records.jsonl en dernier, écrit à côté puis renommé : un chat lancé ne relit l'index qu'une fois complet
        tmp = directory / "records.jsonl.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(directory / "records.jsonl")

    @classmethod
    def load(cls, directory: str | Path) -> BM25Index:
        directory = Path(directory)
        with (directory / "records.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        return cls(bm25s.BM25.load(directory, show_progress=False), records)

    def search(self, query: str, k: int, periode=None, orateur: str | None = None) -> list[tuple[float, dict]]:
        """Les k meilleurs chunks [(score, chunk)], éventuellement limités à une période et à un orateur."""
        mask = None
        if periode or orateur:  # filtres appliqués AVANT de prendre les k meilleurs
            debut, fin = (d.isoformat() for d in periode) if periode else ("", "9999")
            nom = norm_name(orateur)
            mask = np.array([debut <= (r["metadata"].get("date_iso") or "") <= fin
                             and nom in r["metadata"].get("orateurs_norm", "")
                             for r in self.records], dtype=np.float32)
            if not mask.any():
                return []
        q = tokenize([query])
        if not q.ids or not q.ids[0]:  # que des mots vides
            return []
        docs, scores = self.bm25.retrieve(q, k=min(k, len(self.records)), show_progress=False, weight_mask=mask)
        return [(float(s), self.records[int(d)]) for d, s in zip(docs[0], scores[0], strict=True) if s > 0]


def load_bm25(directory: str) -> BM25Index | None:
    """L'index sauvegardé, ou None s'il n'a pas encore été construit. Gardé en mémoire, et relu quand une
    ingestion l'a reconstruit (date de modification du fichier) : un chat déjà lancé n'a pas à être redémarré."""
    path = Path(directory) / "records.jsonl"
    return _load_bm25(directory, path.stat().st_mtime) if path.exists() else None


@lru_cache(maxsize=1)
def _load_bm25(directory: str, mtime: float) -> BM25Index:  # noqa: ARG001
    """`mtime` ne sert qu'à la clé du cache : un index reconstruit (nouvelle date) est relu."""
    return BM25Index.load(directory)


def build_bm25_index(s: Settings | None = None) -> int:
    """Reconstruit l'index BM25 à partir de TOUS les chunks présents dans PostgreSQL (table LlamaIndex)."""
    from .stores import iter_chunks

    s = s or get_settings()
    records = list(iter_chunks(s))
    BM25Index.build(records).save(s.bm25_dir)
    return len(records)


if __name__ == "__main__":  # uv run python -m rag_an.lexical : reconstruit l'index
    print(f"🔤 Index BM25 reconstruit : {build_bm25_index()} chunks -> {get_settings().bm25_dir}/")
