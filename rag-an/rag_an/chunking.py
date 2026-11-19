"""Découpage d'une séance silver en chunks : seuls les discours et les longues interventions du président sont indexés.

Un discours n'est jamais coupé au milieu d'une phrase ; les discours courts d'un même contexte sont regroupés.
Le budget est compté en tokens d'embedding, en-tête compris : au-delà de 512 tokens, Solon ignore la fin du texte.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from .communs import norm_name, token_counter

if TYPE_CHECKING:
    import numpy as np

# Interventions du président indexées à partir de cette longueur (hommages, annonces)
PRESIDENCE_MIN_CARACTERES = 300
_SENTENCE_END = re.compile(r"(?<=[.!?…»])\s+(?=[«A-ZÀ-ÖØ-Þ0-9\"(])")
# « M. Bruno Retailleau », « MM. Lainé et Marle », « art. 3 » : le point n'y termine pas la phrase
_ABBREV = re.compile(r"(?:^|\s)(?:M|MM|Mme|Mmes|Me|Dr|Pr|St|Ste|art|cf|p|no|nos|vol|J\.-C|etc)\.$")


def speaker_label(i: dict) -> str:
    nom = i["orateur"] or "?"
    if i.get("qualite") and i["qualite"] not in nom:
        nom = f"{nom} ({i['qualite']})"
    if i.get("groupe"):
        nom = f"{nom} ({i['groupe']})"
    return nom


def header(seance: dict, i: dict) -> dict:
    """Métadonnées vues par l'embedding et le LLM ; courtes car elles comptent dans le budget."""
    h = {"date_seance": seance["date_libelle"], "sujet": i["sujet"]}
    if i.get("amendement"):
        h["amendement"] = i["amendement"]
    return h


def header_text(h: dict) -> str:
    # même format que LlamaIndex, pour que le comptage des tokens soit juste
    return "\n".join(f"{k}: {v}" for k, v in h.items()) + "\n\n"


# Marqueurs de fin de ligne. _LIST : ligne finie par « : » ou « ; » (« Ont obtenu : / Mme X : 198 voix ; »).
# On évite d'y couper : séparée de son intitulé, une ligne de liste ressemble à une prise de parole.
_PARA, _LIST = "\n", "\n+"


def _sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n"):
        parts: list[str] = []
        for p in _SENTENCE_END.split(para):
            if parts and _ABBREV.search(parts[-1]):
                parts[-1] += " " + p
            elif p.strip():
                parts.append(p)
        out.extend(parts)
        out.append(_LIST if re.search(r"[:;]\s*$", para) else _PARA)
    return out[:-1]


def split_long(label: str, text: str, budget: int, overlap: int, count: Callable[[str], int]) -> list[str]:
    """Découpe un discours trop long en morceaux d'au plus `budget` tokens, aux frontières de phrases."""
    units = _sentences(text)
    tokens = [0 if u in (_PARA, _LIST) else count(u) for u in units]
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = count(label) + 2
    # morceaux de taille voisine plutôt que pleins suivis d'un reste minuscule
    total = cur_tokens + sum(tokens)
    n = 1 if total <= budget else 1 + -(-(total - budget) // max(1, budget - overlap - count(label) - 8))
    target = min(budget, total // n + overlap + 16)

    def flush():
        body = " ".join(cur).replace(f" {_LIST} ", "\n").replace(f" {_PARA} ", "\n").strip()
        if body:
            prefix = f"{label} : " if not pieces else f"{label} (suite) : "
            pieces.append(prefix + body)

    list_head = None  # intitulé de la liste en cours, répété si la liste est coupée
    for idx, (u, t) in enumerate(zip(units, tokens, strict=True)):
        in_list = idx > 0 and units[idx - 1] == _LIST
        if u == _PARA:
            list_head = None
        elif u == _LIST and list_head is None and units[idx - 1].rstrip().endswith(":"):
            list_head = units[idx - 1]
        too_big = cur_tokens + t > budget
        if cur and t and (too_big or (cur_tokens + t / 2 > target and not in_list)):
            flush()
            # recouvrement : dernières phrases du morceau précédent, sauf au milieu d'une liste
            keep, kept = [], 0
            if in_list and list_head and count(list_head) <= overlap:
                keep, kept = [list_head, _LIST], count(list_head)
            elif not in_list:
                for j in range(len(cur) - 1, -1, -1):
                    k = 0 if cur[j] in (_PARA, _LIST) else count(cur[j])
                    # pas de recouvrement par une ligne de liste : sans son intitulé, elle égare
                    if kept + k > overlap or cur[j] == _LIST or (j > 0 and cur[j - 1] == _LIST):
                        break
                    keep.insert(0, cur[j])
                    kept += k
            while keep and keep[0] in (_PARA, _LIST):
                keep.pop(0)
            cur, cur_tokens = keep, count(label) + 8 + kept
        if t > budget:  # phrase démesurée : coupure brute en mots
            words = u.split()
            step = max(1, len(words) * budget // (t + 1) - 8)
            for j in range(0, len(words), step):
                cur = [" ".join(words[j:j + step])]
                flush()
            cur, cur_tokens = [], count(label) + 2
            continue
        if not cur and u in (_PARA, _LIST):
            continue
        cur.append(u)
        cur_tokens += t
    flush()
    return pieces


def _paragraphs(text: str) -> list[str]:
    """Paragraphes d'un discours ; les lignes d'une même liste restent ensemble."""
    out: list[str] = []
    for line in text.split("\n"):
        if out and re.search(r"[:;]\s*$", out[-1].rsplit("\n", 1)[-1]):
            out[-1] += "\n" + line
        elif line.strip():
            out.append(line)
    return out


def split_paragraphs(label: str, text: str, budget: int, count: Callable[[str], int]) -> list[str]:
    """Coupe entre deux paragraphes, sans recouvrement ; un paragraphe trop long passe par split_long."""
    paras = _paragraphs(text)
    sizes = [count(p) + 1 for p in paras]
    total = count(label) + 2 + sum(sizes)
    target = total / -(-total // budget)  # nombre minimal de morceaux, de taille voisine
    pieces: list[str] = []
    cur, cur_tokens = [], count(label) + 2

    def flush():
        if cur:
            pieces.append(f"{label}{' (suite)' if pieces else ''} : " + "\n".join(cur))

    for p, t in zip(paras, sizes, strict=True):
        if cur and (cur_tokens + t > budget or cur_tokens + t / 2 > target):
            flush()
            cur, cur_tokens = [], count(label) + 2
        if count(label) + 2 + t > budget:  # paragraphe démesuré : aux phrases
            pieces.extend(split_long(label, p, budget, 0, count))
            continue
        cur.append(p)
        cur_tokens += t
    flush()
    return pieces


def split_theme(label: str, text: str, budget: int, count: Callable[[str], int],
                embed: Callable[[list[str]], "np.ndarray"], min_tokens: int = 48, window: int = 3) -> list[str]:
    """Coupe récursivement à la frontière où le sujet change le plus, sans recouvrement.

    Similarité cosinus entre les moyennes des `window` phrases d'avant et d'après ; les fins de paragraphe sont
    légèrement favorisées, les lignes de liste exclues, et chaque côté garde au moins `min_tokens`.
    """
    import numpy as np

    units = _sentences(text)
    sents, para_end, list_end = [], [], []  # nature de la frontière qui suit la phrase k
    for u in units:
        if u in (_PARA, _LIST):
            if sents:
                para_end[-1], list_end[-1] = u == _PARA, u == _LIST
            continue
        sents.append(u)
        para_end.append(False)
        list_end.append(False)
    sizes = [count(s) + 1 for s in sents]
    V = np.asarray(embed(sents))
    sim = []
    for k in range(len(sents) - 1):
        a = V[max(0, k + 1 - window):k + 1].mean(0)
        b = V[k + 1:k + 1 + window].mean(0)
        sim.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) - (0.05 if para_end[k] else 0))
    room = budget - count(label) - 12

    def cut(a: int, b: int) -> list[tuple[int, int]]:
        if sum(sizes[a:b]) <= room or b - a == 1:
            return [(a, b)]
        cands = [k for k in range(a, b - 1) if not list_end[k]
                 and sum(sizes[a:k + 1]) >= min_tokens and sum(sizes[k + 1:b]) >= min_tokens]
        if not cands:  # aucune frontière admissible : coupe au milieu, en tokens
            half, acc, k = sum(sizes[a:b]) / 2, 0, a
            while k < b - 2 and acc + sizes[k] < half:
                acc += sizes[k]
                k += 1
            cands = [k]
        k = min(cands, key=lambda k: sim[k])
        return cut(a, k + 1) + cut(k + 1, b)

    pieces: list[str] = []
    for a, b in cut(0, len(sents)):
        body = ""
        for k in range(a, b):
            body += sents[k] + ("" if k == b - 1 else "\n" if para_end[k] or list_end[k] else " ")
        prefix = f"{label} : " if not pieces else f"{label} (suite) : "
        if sizes[a] > room:  # phrase démesurée : découpage par taille
            pieces.extend(split_long(label, body, budget, 0, count))
        else:
            pieces.append(prefix + body)
    return pieces


def chunk_seance(silver: dict, max_tokens: int = 512, overlap_tokens: int = 64,
                 count: Callable[[str], int] | None = None, group: bool = True, mode: str = "taille",
                 embed: Callable | None = None) -> list[dict]:
    """Chunks d'une séance : [{"text", "metadata", "paragraphes", "tokens"}].

    max_tokens : budget total, en-tête compris.
    mode       : découpage des discours trop longs (RAG_CHUNK_MODE) : « taille », « paragraphe » ou « theme »
                 (ce dernier demande `embed`).
    """
    count = count or token_counter(None)
    seance = silver["seance"]
    units = [
        i for i in silver["interventions"]
        if i["type"] == "discours"
        or (i["type"] == "presidence" and i["nb_caracteres"] >= PRESIDENCE_MIN_CARACTERES)
    ]
    chunks: list[dict] = []
    buf: list[dict] = []
    buf_tokens = 0
    buf_key = None

    def emit(parts: list[str], members: list[dict], h: dict):
        first = members[0]
        text = "\n\n".join(parts)
        chunks.append({
            "text": text,
            "tokens": count(header_text(h) + text),
            "metadata": {
                **h,
                "seance_id": seance["uid"],
                "date_iso": seance["date"],
                "num_seance": seance["num_seance"],
                "presidence": first.get("president_seance") or "",
                "orateurs": " ; ".join(dict.fromkeys(speaker_label(m) for m in members)),
                # forme normalisée (sans accents ni civilité) pour filtrer par orateur
                "orateurs_norm": " ; ".join(dict.fromkeys(norm_name(m["orateur"]) for m in members)),
                "orateur_ids": " ".join(dict.fromkeys(m["id_acteur"] or "" for m in members)).strip(),
                "article": first.get("article"),
                "ordre": first["ordre_debut"],
                "type": "debat",
            },
            "paragraphes": [p for m in members for p in m["paragraphes"]],
        })

    def flush():
        nonlocal buf, buf_tokens
        if buf:
            h = header(seance, buf[0])
            emit([f"{speaker_label(m)} : {m['texte']}" for m in buf], buf, h)
        buf, buf_tokens = [], 0

    for i in units:
        h = header(seance, i)
        # marge : tokens spéciaux, et la somme des phrases comptées une à une sous-estime le total
        budget = max_tokens - count(header_text(h)) - max(6, max_tokens // 25)
        key = (i["sujet"], i.get("article"), i.get("amendement"), i.get("sous_amendement"))
        label = speaker_label(i)
        t = count(f"{label} : {i['texte']}") + 2
        if buf and (not group or key != buf_key or buf_tokens + t > budget):
            flush()
        if t > budget:
            if mode == "paragraphe":
                pieces = split_paragraphs(label, i["texte"], budget, count)
            elif mode == "theme":
                pieces = split_theme(label, i["texte"], budget, count, embed)
            else:
                pieces = split_long(label, i["texte"], budget, overlap_tokens, count)
            for piece in pieces:
                emit([piece], [i], h)
            continue
        buf.append(i)
        buf_tokens += t
        buf_key = key
    flush()

    # identifiants stables : séance#ordre du 1er discours, suffixé si plusieurs chunks partent du même
    seen: dict[str, int] = {}
    for c in chunks:
        base = f"{seance['uid']}#{c['metadata']['ordre']}"
        n = seen.get(base, 0)
        seen[base] = n + 1
        c["metadata"]["doc_id"] = base if n == 0 else f"{base}-{n}"
    return chunks
