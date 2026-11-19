"""Chargement des modèles : LLM (Ollama) et embedding (Solon).

Les modèles Hugging Face sont lus d'abord dans le cache local : sinon sentence-transformers interroge le Hub
pour chaque fichier à chaque lancement (≈ 60 s pour Solon contre 2 s).
"""
from __future__ import annotations

import logging
import threading

from llama_index.core import Settings as LISettings

from .config import Settings, get_settings

log = logging.getLogger(__name__)


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_local_first(loader, *args, **kwargs):
    """Appelle un chargeur Hugging Face sur le cache local, puis sur le réseau si le modèle n'y est pas."""
    try:
        return loader(*args, local_files_only=True, **kwargs)
    except (OSError, ValueError) as e:  # absent du cache (1er lancement)
        log.info("modèle absent du cache local, téléchargement : %s", e)
        return loader(*args, **kwargs)


def build_embed_model(s: Settings | None = None):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    s = s or get_settings()
    device = s.embed_device or detect_device()
    kwargs = {}
    if device == "cuda" and s.embed_fp16:
        import torch

        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return load_local_first(HuggingFaceEmbedding, model_name=s.embed_model, device=device,
                            embed_batch_size=s.embed_batch_size, query_instruction=s.embed_query_instruction,
                            trust_remote_code=True, **kwargs)


def build_llm(s: Settings | None = None):
    from llama_index.llms.ollama import Ollama

    s = s or get_settings()
    return Ollama(model=s.llm_model, base_url=s.ollama_url, request_timeout=s.llm_timeout,
                  context_window=s.llm_context_window, temperature=s.llm_temperature, keep_alive=s.llm_keep_alive,
                  additional_kwargs={"num_predict": s.llm_max_tokens})


def preload_llm(s: Settings | None = None) -> threading.Thread:
    """Fait charger le LLM par Ollama (requête vide) dans un thread, pendant le chargement de l'embedding.

    Sans cela, Ollama ne charge le modèle qu'à la première question. Renvoie le thread.
    """
    s = s or get_settings()

    def run():
        import httpx

        try:
            httpx.post(f"{s.ollama_url}/api/generate",
                       # même num_ctx que les questions : sinon Ollama charge Mistral en 4 096 et le recharge en
                       # 8 192 à la 1re question (+5 s, mesuré le 16/11)
                       json={"model": s.llm_model, "prompt": "", "keep_alive": s.llm_keep_alive,
                             "options": {"num_ctx": s.llm_context_window}},
                       timeout=s.llm_timeout)
        except httpx.HTTPError as e:  # Ollama arrêté : l'erreur sera signalée à la première question
            log.warning("préchargement de %s impossible : %s", s.llm_model, e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def configure(s: Settings | None = None) -> None:
    """Renseigne les Settings globaux de LlamaIndex (modèle d'embedding et LLM)."""
    s = s or get_settings()
    LISettings.embed_model = build_embed_model(s)
    LISettings.llm = build_llm(s)
