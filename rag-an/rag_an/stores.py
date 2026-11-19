"""Accès aux bases. PostgreSQL : chunks et embeddings (pgvector), tables silver (stats.py).
MongoDB : documents sources complets, registre d'ingestion (hash, pour ne ré-indexer que ce qui a changé)
et historique des conversations.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache

import psycopg2
from llama_index.core import Document
from llama_index.core.llms import ChatMessage
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore
from pymongo import MongoClient

from .config import Settings, get_settings


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
@contextmanager
def connect(s: Settings | None = None):
    s = s or get_settings()
    conn = psycopg2.connect(host=s.pg_host, port=s.pg_port, user=s.pg_user, password=s.pg_password,
                            dbname=s.pg_database)
    try:
        yield conn
    finally:
        conn.close()


def get_vector_store(s: Settings | None = None) -> PGVectorStore:
    """Table des chunks et index HNSW (similarité cosinus).

    Deux réglages évitent que l'index renvoie moins de résultats que demandé :
    - hnsw.ef_search (s.hnsw_ef_search) : l'index ne renvoie jamais plus de résultats que ce nombre ;
    - hnsw.iterative_scan = strict_order (pgvector >= 0.8) : les filtres (date, orateur) s'appliquent après la
      recherche dans l'index ; sans ce réglage, un filtre sélectif ne laisse passer que quelques résultats.
    """
    s = s or get_settings()
    return PGVectorStore.from_params(
        host=s.pg_host, port=str(s.pg_port), user=s.pg_user, password=s.pg_password, database=s.pg_database,
        table_name=s.pg_table,  # LlamaIndex préfixe la table par « data_ »
        embed_dim=s.embed_dim,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": s.hnsw_ef_search,
                     "hnsw_dist_method": "vector_cosine_ops"},
        create_engine_kwargs={"connect_args": {"options": "-c hnsw.iterative_scan=strict_order"}},
    )


def reset_vector_store(s: Settings | None = None) -> None:
    """Supprime la table pgvector pour recréer son schéma avec la dimension courante."""
    s = s or get_settings()
    with connect(s) as conn, conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS public."data_{s.pg_table}"')
        conn.commit()


def iter_chunks(s: Settings | None = None):
    """Tous les chunks indexés dans PostgreSQL : {"id", "text", "metadata"} (pour construire l'index BM25)."""
    s = s or get_settings()
    with connect(s) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT node_id, text, metadata_ FROM public."data_{s.pg_table}" ORDER BY node_id')
        for node_id, text, meta in cur:
            meta = meta if isinstance(meta, dict) else json.loads(meta)
            # on retire les clés internes de LlamaIndex (« _node_content »…, identifiants en double)
            meta = {k: v for k, v in meta.items() if not k.startswith("_") and k not in ("document_id", "ref_doc_id")}
            yield {"id": node_id, "text": text, "metadata": meta}


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _mongo_client(uri: str) -> MongoClient:
    """Un seul client par adresse, réutilisé (il gère son propre pool de connexions)."""
    return MongoClient(uri)


def mongo_db(s: Settings | None = None):
    s = s or get_settings()
    return _mongo_client(s.mongo_uri)[s.mongo_db]


def get_docstore(s: Settings | None = None) -> MongoDocumentStore:
    """Registre d'ingestion LlamaIndex (doc_id -> hash) utilisé par l'IngestionPipeline."""
    s = s or get_settings()
    return MongoDocumentStore.from_uri(uri=s.mongo_uri, db_name=s.mongo_db, namespace="ingestion")


def save_source_documents(docs: list[Document], s: Settings | None = None) -> None:
    """Stocke le texte intégral + métadonnées de chaque document source (consultation, citations)."""
    coll = mongo_db(s)["documents"]
    now = datetime.now(timezone.utc)
    for d in docs:
        coll.update_one({"_id": d.doc_id},
                        {"$set": {"text": d.text, "metadata": d.metadata, "hash": d.hash, "updated_at": now}},
                        upsert=True)


def get_source_document(doc_id: str, s: Settings | None = None) -> dict | None:
    return mongo_db(s)["documents"].find_one({"_id": doc_id})


class ConversationStore:
    """Historique des conversations dans MongoDB (une conversation = un document)."""

    def __init__(self, s: Settings | None = None):
        self.coll = mongo_db(s)["conversations"]

    def load(self, session_id: str) -> list[ChatMessage]:
        conv = self.coll.find_one({"_id": session_id}) or {}
        return [ChatMessage(role=m["role"], content=m["content"]) for m in conv.get("messages", [])]

    def append(self, session_id: str, role: str, content: str, sources: list[dict] | None = None) -> None:
        msg = {"role": role, "content": content, "ts": datetime.now(timezone.utc)}
        if sources:
            msg["sources"] = sources
        self.coll.update_one({"_id": session_id},
                             {"$push": {"messages": msg}, "$setOnInsert": {"created_at": msg["ts"]}},
                             upsert=True)

    def reset(self, session_id: str) -> None:
        self.coll.delete_one({"_id": session_id})
