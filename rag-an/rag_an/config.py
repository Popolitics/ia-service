"""Configuration centralisée, surchargeable par variables d'environnement ou fichier .env (préfixe RAG_)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- LLM (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "mistral"
    llm_context_window: int = 8192
    llm_timeout: float = 300.0  # le premier appel charge le modèle en mémoire, souvent long
    llm_temperature: float = 0.1
    llm_keep_alive: str = "30m"  # garde le modèle chargé entre deux questions ("-1" = indéfiniment)
    llm_max_tokens: int = 1200  # plafond de la réponse : Mistral peut boucler sur la même ligne sans fin
    llm_preload: bool = True  # charge Mistral dans Ollama dès le lancement du chat, en parallèle de l'embedding

    # --- Embeddings (sentence-transformers, en local) ---
    embed_model: str = "OrdalieTech/Solon-embeddings-large-0.1"
    embed_dim: int = 1024
    embed_device: str | None = None  # None = auto (cuda > mps > cpu)
    embed_batch_size: int = 16
    embed_fp16: bool = True  # demi-précision sur GPU : chargement et encodage ~2x plus rapides
    embed_query_instruction: str = "query : "  # préfixe des questions exigé par Solon

    # --- Découpage (voir chunking.py) ---
    # Tokens du modèle d'embedding, en-tête de métadonnées COMPRIS : ne pas dépasser sa fenêtre (Solon : 512),
    # sinon la fin du chunk est ignorée sans avertissement.
    chunk_max_tokens: int = 192
    chunk_overlap_tokens: int = 24  # recouvrement (en phrases entières) quand un long discours est découpé
    # Découpage des discours plus longs que chunk_max_tokens : « taille » (morceaux de taille voisine, avec
    # recouvrement), « paragraphe » (entre deux paragraphes seulement) ou « theme » (là où le sujet change ;
    # encode chaque phrase, plus lent).
    chunk_mode: str = "taille"

    # --- Données silver (JSON produits par silver.py) ---
    silver_dir: str = "silver"
    bm25_dir: str = "index_bm25"  # index lexical BM25 (bm25s), reconstruit à chaque ingestion

    # --- PostgreSQL / pgvector ---
    pg_host: str = "localhost"
    pg_port: int = 55432
    pg_user: str = "rag"
    pg_password: str = "rag"
    pg_database: str = "rag"
    pg_table: str = "an_chunks"  # LlamaIndex préfixe la table par "data_"

    # --- MongoDB ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "rag_an"

    # --- Recherche / génération ---
    top_k: int = 20  # plafond large : c'est context_token_budget qui limite réellement
    # L'index HNSW ne renvoie jamais plus de résultats que ce nombre de candidats : doit être >= 3 × top_k.
    # Réglage de session, sans réindexation (voir stores.py).
    hnsw_ef_search: int = 100
    # Cross-encoder qui reclasse les 3 × top_k candidats de la recherche hybride ; None = désactivé
    rerank_model: str | None = "BAAI/bge-reranker-v2-m3"
    # L'agent est sans état : il reçoit à chaque question les `history_messages` derniers messages (8 = 4 échanges)
    history_messages: int = 8
    # Longueur maximale de chaque message d'historique montré au LLM : un résumé de séance peut dépasser
    # 3 000 caractères, et 8 messages entiers déborderaient la fenêtre de 8 192 tokens
    history_max_chars: int = 1000
    # Budget d'extraits envoyés au LLM, en tokens (pris par pertinence) : au-delà, Mistral 7B confond chiffres
    # et orateurs
    context_token_budget: int = 1500



@lru_cache
def get_settings() -> Settings:
    return Settings()
