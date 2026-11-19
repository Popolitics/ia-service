"""Fonctions communes : normalisation des noms d'orateurs, comptage de tokens, métadonnées cachées.

Ne dépend que de la bibliothèque standard (transformers n'est importé qu'à l'appel de load_tokenizer).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable
from functools import lru_cache


def norm_name(s: str | None) -> str:
    """« Mme Élisa Martin » -> « elisa martin » (comparaison sans civilité, casse ni accents)."""
    s = "".join(c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn").lower()
    s = re.sub(r"^(m\.|mme|mmes|mm\.|mlle|monsieur|madame)\s+", "", s.strip())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def load_tokenizer(model_name: str):
    """Tokenizer Hugging Face, lu dans le cache local, ou téléchargé s'il n'y est pas."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).info("tokenizer absent du cache local, téléchargement : %s", e)
        tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = 10**6  # on compte des tokens, on n'encode pas : pas d'avertissement « 513 > 512 »
    return tok


@lru_cache(maxsize=4)
def token_counter(model_name: str | None) -> Callable[[str], int]:
    """Compteur de tokens du modèle d'embedding (sans modèle : 4 caractères par token, mesuré sur Solon)."""
    if model_name is None:
        return lambda s: len(s) // 4 + 1
    tok = load_tokenizer(model_name)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


# Métadonnées vues par le modèle d'embedding : en-tête court, compté dans le budget du chunk
EMBED_METADATA_KEYS = ("date_seance", "sujet", "amendement", "document", "groupe")
# Métadonnées montrées au LLM ; les autres (identifiants, scores…) ne servent qu'au programme
LLM_METADATA_KEYS = ("source_n", "date_seance", "sujet", "amendement", "article", "orateurs", "presidence",
                     "document", "groupe")


def hidden_metadata(metadata: dict) -> dict:
    """Arguments d'un Document ou d'un TextNode qui cachent au LLM et à l'embedding les métadonnées techniques."""
    return {"excluded_embed_metadata_keys": [k for k in metadata if k not in EMBED_METADATA_KEYS],
            "excluded_llm_metadata_keys": [k for k in metadata if k not in LLM_METADATA_KEYS]}
