"""Ingestion incrémentale : documents -> JSON silver + MongoDB (sources) + PostgreSQL (chunks, vecteurs,
tables de statistiques) + index BM25.

Usage :
    uv run rag-ingest docs/            # documents nouveaux ou modifiés seulement ; donner TOUTES les sources
    uv run rag-ingest silver/ docs/    # plusieurs dossiers
    uv run rag-ingest docs/ --reset    # repart de zéro (requis après changement d'embedding ou de taille de chunk)
"""
from __future__ import annotations

import argparse
import logging
import time

from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo

from .communs import load_tokenizer, token_counter
from .config import get_settings
from .lexical import build_bm25_index
from .models import build_embed_model
from .parsing import load_directories
from .stats import load_silver
from .stores import get_docstore, get_vector_store, mongo_db, reset_vector_store, save_source_documents


class ConditionalSentenceSplitter(SentenceSplitter):
    """Ne redécoupe pas les comptes rendus, déjà découpés par chunking.py."""

    def _parse_nodes(self, nodes, show_progress=False, **kwargs) -> list[BaseNode]:
        parsed = []
        for node in nodes:
            if node.metadata.get("chunked", False):
                # le chunk est son propre document source : sans ref_doc_id, pgvector garderait l'ancienne version
                # d'un chunk modifié lors de la réingestion
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=node.node_id)
                parsed.append(node)
            else:
                parsed.extend(super()._parse_nodes([node], show_progress=show_progress, **kwargs))
        return parsed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directories", nargs="*",
                    help="dossiers des sources : XML ou JSON silver des séances, decl_*.txt (défaut : docs)")
    ap.add_argument("--reset", action="store_true",
                    help="supprime les chunks, le registre et les documents avant ingestion")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    s = get_settings()
    args.directories = args.directories or ["docs"]
    t0 = time.perf_counter()
    count = token_counter(s.embed_model)
    if s.chunk_mode not in ("taille", "paragraphe", "theme"):
        raise SystemExit(f"RAG_CHUNK_MODE inconnu : {s.chunk_mode!r} (taille, paragraphe ou theme)")
    embed_model = build_embed_model(s)
    # le découpage par thème encode chaque phrase des longs discours
    embed = embed_model.get_text_embedding_batch if s.chunk_mode == "theme" else None
    docs, seances = load_directories(args.directories, s.chunk_max_tokens, s.chunk_overlap_tokens, count,
                                     silver_out=s.silver_dir, mode=s.chunk_mode, embed=embed)
    print(f"📄 {len(seances)} séances (JSON silver dans {s.silver_dir}/) -> {len(docs)} documents à indexer, "
          f"découpage « {s.chunk_mode} » {s.chunk_max_tokens} tokens ({time.perf_counter() - t0:.1f}s)")

    if args.reset:
        reset_vector_store(s)
        db = mongo_db(s)
        for name in db.list_collection_names():
            if name.startswith("ingestion") or name == "documents":
                db.drop_collection(name)
        print("🧹 Index PostgreSQL et registre MongoDB vidés")

    load_silver(seances, s)
    print("📊 Tables de statistiques silver_* à jour (séances, interventions, votes, points, élections)")

    vector_store = get_vector_store(s)
    save_source_documents(docs, s)

    # déclarations (texte libre) : découpées ici, en tokens du modèle d'embedding
    tok = load_tokenizer(s.embed_model)
    splitter = ConditionalSentenceSplitter(chunk_size=s.chunk_max_tokens - 16,
                                           chunk_overlap=s.chunk_overlap_tokens,
                                           tokenizer=lambda t: tok(t, add_special_tokens=False)["input_ids"])
    pipeline = IngestionPipeline(
        transformations=[splitter, embed_model],
        docstore=get_docstore(s),
        vector_store=vector_store,
        # supprime aussi les documents absents : les dossiers donnés doivent contenir TOUTES les sources
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
    )
    t1 = time.perf_counter()
    nodes = pipeline.run(documents=docs, show_progress=True)
    print(f"⚡ {len(nodes)} chunks (ré)indexés dans PostgreSQL en {time.perf_counter() - t1:.1f}s "
          f"(documents inchangés ignorés)")

    n = build_bm25_index(s)
    print(f"🔤 Index BM25 reconstruit : {n} chunks -> {s.bm25_dir}/")


if __name__ == "__main__":
    main()
