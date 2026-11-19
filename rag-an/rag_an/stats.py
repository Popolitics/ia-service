"""Statistiques sur les séances : tables silver dans PostgreSQL et requêtes paramétrées pour l'agent.

Compter (prises de parole, votes) ou résumer une séance entière ne se fait pas par recherche de passages.
L'agent appelle des fonctions à paramètres (orateur, date, sujet…) plutôt que d'écrire du SQL : un modèle 7B
se trompe souvent en SQL libre, alors qu'il remplit bien des paramètres.

Tables (JSON produits par silver.py) :
    silver_seances        une ligne par séance
    silver_interventions  une ligne par prise de parole (discours, interruption, presidence, didascalie)
    silver_votes          une ligne par vote
    silver_points         le plan de l'ordre du jour
    silver_elections      les élections et nominations
"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import Json, execute_values

from .communs import norm_name
from .config import Settings
from .stores import connect

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS silver_seances (
    uid text PRIMARY KEY, date date, heure_debut text, date_libelle text, num_seance int,
    num_seance_jour int, legislature int, session text, presidences text[], fichier text,
    nb_discours int, nb_interruptions int, nb_votes int, heure_fin text, seance_ref text
);
CREATE TABLE IF NOT EXISTS silver_interventions (
    id text PRIMARY KEY, seance_uid text, date date, ordre_debut int, ordre_fin int, type text,
    orateur text, orateur_norm text, id_acteur text, qualite text, role text, president_seance text,
    sujet text, section text, sous_section text, article text, amendement text, interrompt text,
    nb_interruptions int, nb_mots int, texte text
);
CREATE INDEX IF NOT EXISTS silver_interventions_seance ON silver_interventions (seance_uid);
CREATE INDEX IF NOT EXISTS silver_interventions_orateur ON silver_interventions (orateur_norm);
CREATE TABLE IF NOT EXISTS silver_votes (
    id text PRIMARY KEY, seance_uid text, date date, ordre int, objet text, numeros text[], resultat text,
    scrutin_public boolean, votants int, exprimes int, pour int, contre int, sujet text, texte text
);
CREATE TABLE IF NOT EXISTS silver_points (
    seance_uid text, ordre int, niveau int, titre text, chemin text
);
CREATE INDEX IF NOT EXISTS silver_points_seance ON silver_points (seance_uid);
CREATE TABLE IF NOT EXISTS silver_elections (
    id text PRIMARY KEY, seance_uid text, date date, ordre int, objet text, tour int, votants int,
    exprimes int, majorite_absolue int, candidats jsonb, elus text[], proclamation text, sujet text
);
-- comparaison sans accents ni casse (sujets) : extension unaccent (contrib PostgreSQL)
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE OR REPLACE FUNCTION unaccent_lower(text) RETURNS text LANGUAGE sql IMMUTABLE AS
    $$ SELECT regexp_replace(lower(public.unaccent($1)), '[^a-z0-9]+', ' ', 'g') $$;
"""


def load_silver(seances: list[dict], s: Settings | None = None) -> None:
    """Charge (ou recharge) des séances silver dans les tables, séance par séance."""
    with connect(s) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        for sj in seances:
            se, st = sj["seance"], sj["statistiques"]
            uid = se["uid"]
            for table in ("silver_interventions", "silver_votes", "silver_points", "silver_elections"):
                cur.execute(f"DELETE FROM {table} WHERE seance_uid = %s", (uid,))
            cur.execute("DELETE FROM silver_seances WHERE uid = %s", (uid,))
            cur.execute(
                "INSERT INTO silver_seances (uid, date, heure_debut, date_libelle, num_seance, num_seance_jour, "
                "legislature, session, presidences, fichier, nb_discours, nb_interruptions, nb_votes, heure_fin, "
                "seance_ref) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (uid, se["date"], se["heure_debut"], se["date_libelle"], se["num_seance"], se["num_seance_jour"],
                 se["legislature"], se["session"], sj["presidences"], se["fichier"],
                 st["interventions"].get("discours", 0), st["interventions"].get("interruption", 0), st["votes"],
                 se.get("heure_fin"), se.get("seance_ref")))
            if sj.get("points"):
                execute_values(cur, "INSERT INTO silver_points VALUES %s", [
                    (uid, p["ordre"], p["niveau"], p["titre"], p["chemin"]) for p in sj["points"]])
            if sj.get("elections"):
                execute_values(cur, "INSERT INTO silver_elections VALUES %s", [
                    (e["id"], uid, se["date"], e["ordre"], e["objet"], e["tour"], e["votants"], e["exprimes"],
                     e["majorite_absolue"], Json(e["candidats"]), e["elus"], e["proclamation"], e["sujet"])
                    for e in sj["elections"]])
            execute_values(cur, "INSERT INTO silver_interventions VALUES %s", [
                (i["id"], uid, se["date"], i["ordre_debut"], i["ordre_fin"], i["type"], i["orateur"],
                 norm_name(i["orateur"]), i["id_acteur"], i["qualite"], i["role"], i["president_seance"],
                 i["sujet"], i["section"], i["sous_section"], i["article"], i["amendement"], i.get("interrompt"),
                 len(i.get("interruptions", [])), i["nb_mots"], i["texte"])
                for i in sj["interventions"]
            ])
            if sj["votes"]:
                execute_values(cur, "INSERT INTO silver_votes (id, seance_uid, date, ordre, objet, numeros, resultat, "
                               "scrutin_public, votants, exprimes, pour, contre, sujet, texte) VALUES %s", [
                    (v["id"], uid, se["date"], v["ordre"], v["objet"], v["numeros"], v["resultat"], v["scrutin_public"],
                     *((v["scrutin"] or {}).get(k) for k in ("votants", "exprimes", "pour", "contre")),
                     v["sujet"], v["texte"])
                    for v in sj["votes"]
                ])
        conn.commit()


# ---------------------------------------------------------------------------
# Dates en français
# ---------------------------------------------------------------------------
MOIS = {m: i for i, m in enumerate(["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout",
                                     "septembre", "octobre", "novembre", "decembre"], 1)}


SAISONS = {"printemps": (3, 5), "ete": (6, 8), "automne": (9, 11), "hiver": (12, 2)}  # mois de début et de fin
_M = "|".join(MOIS)
# une mention de date dans le texte normalisé par norm_name (« 18/07/2024 » y devient « 18 07 2024 »)
_DATE = re.compile(
    r"\b(?P<iso_a>\d{4}) (?P<iso_m>\d{1,2}) (?P<iso_j>\d{1,2})\b"                   # 2024-07-18
    r"|\b(?P<num_j>\d{1,2}) (?P<num_m>\d{1,2}) (?P<num_a>\d{4}|\d{2})\b"            # 18/07/2024, 18/07/24
    rf"|\b(?P<j>\d{{1,2}})(?:er)?(?: (?P<m>{_M}))?(?: (?P<a>\d{{4}}))?\b"            # 18 juillet 2024, 21 (au…)
    rf"|\b(?P<saison>{'|'.join(SAISONS)})(?: (?P<sa>\d{{4}}))?\b"                    # été 2025
    rf"|\b(?P<m2>{_M})(?: (?P<a2>\d{{4}}))?\b"                                       # juillet 2024, juillet
    r"|\b(?P<a3>(?:19|20)\d{2})\b")                                                  # 2024
# mots qui annoncent un jour seul, dont le mois et l'année viennent ensuite : « du 21 au 25 octobre 2024 »
_AVANT_JOUR_SEUL = {"du", "le", "les", "entre", "au", "et", "a"}


class DateInvalide(ValueError):
    """Date impossible (« 31 juin 2024 ») : à signaler, plutôt que de chercher sans filtre."""


def _fin_du_mois(a: int, m: int) -> date:
    return (date(a, m, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)


def parse_periode(text: str | None, reference: Callable[[], date] | None = None) -> tuple[date, date] | None:
    """Texte d'une date ou d'une période -> (début, fin) inclus, ou None si le texte ne contient pas de date.

    « 18 juillet 2024 », « 1er août 2024 », « 18/07/2024 », « 18/07/24 », « 2024-07-18 », « juillet 2024 »,
    « 2024 », « été 2025 », « semaine du 21 octobre 2024 » ; plusieurs dates -> de la première à la dernière :
    « du 21 au 25 octobre 2024 », « entre le 1er et le 3 juillet 2025 », « octobre et novembre 2024 ».
    Un jour ou un mois sans année (« le 22 octobre ») prend l'année de sa dernière occurrence au plus tard à la
    date `reference()` (la dernière séance de la base ; aujourd'hui par défaut).
    Lève DateInvalide pour une date impossible.
    """
    if not text:
        return None
    t = norm_name(text).replace(" er ", " ")
    mentions = []  # [jour, mois, année, saison] ; None = non précisé
    for x in _DATE.finditer(t):
        if x["iso_a"]:
            mentions.append([int(x["iso_j"]), int(x["iso_m"]), int(x["iso_a"]), None])
        elif x["num_a"]:
            a = int(x["num_a"])
            mentions.append([int(x["num_j"]), int(x["num_m"]), a + 2000 if a < 100 else a, None])
        elif x["j"]:
            avant = t[:x.start()].split()[-1:]
            if not x["m"] and avant and avant[0] not in _AVANT_JOUR_SEUL:
                continue  # nombre isolé qui n'est pas un jour (« séance n° 2 »)
            mentions.append([int(x["j"]), MOIS.get(x["m"]), int(x["a"]) if x["a"] else None, None])
        elif x["saison"]:
            mentions.append([None, None, int(x["sa"]) if x["sa"] else None, x["saison"]])
        elif x["m2"]:
            mentions.append([None, MOIS[x["m2"]], int(x["a2"]) if x["a2"] else None, None])
        else:
            mentions.append([None, None, int(x["a3"]), None])
    # « du 21 au 25 octobre 2024 » : un jour sans mois ni année prend ceux de la date suivante
    for cur, nxt in zip(reversed(mentions[:-1]), reversed(mentions[1:]), strict=True):
        if cur[0] is not None and cur[1] is None:
            cur[1] = nxt[1]
        if cur[2] is None and cur[1] is not None and nxt[1] is not None and nxt[2] is not None:
            cur[2] = nxt[2] - (cur[1] > nxt[1])  # « du 15 décembre au 10 janvier 2025 » : décembre 2024
    ref = None
    periodes = []
    for j, m, a, saison in mentions:
        if j is not None and m is None:
            continue  # jour seul jamais complété
        if a is None and (m or saison):  # année manquante : la dernière occurrence passée
            ref = ref or (reference() if reference else date.today())
            m_debut = m or SAISONS[saison][0]
            a = ref.year if (m_debut, j or 1) <= (ref.month, ref.day) else ref.year - 1
        try:
            if saison:
                m1, m2 = SAISONS[saison]
                periodes.append((date(a, m1, 1), _fin_du_mois(a + (m2 < m1), m2)))
            elif j is not None:
                periodes.append((date(a, m, j), date(a, m, j)))
            elif m is not None:
                periodes.append((date(a, m, 1), _fin_du_mois(a, m)))
            else:
                periodes.append((date(a, 1, 1), date(a, 12, 31)))
        except ValueError:
            raise DateInvalide(f"Date invalide : « {text} ».") from None
    if not periodes:
        return None
    debut, fin = min(p[0] for p in periodes), max(p[1] for p in periodes)
    if "semaine" in t.split() and debut == fin:
        fin = debut + timedelta(days=6)
    return debut, fin


def derniere_seance(s: Settings | None = None) -> date:
    """Date de la dernière séance chargée (référence des dates sans année) ; aujourd'hui si la base est vide."""
    try:
        return query("SELECT max(date) FROM silver_seances", [], s)[0][0] or date.today()
    except psycopg2.Error:  # tables silver pas encore créées
        return date.today()


# « la séance la plus récente », « la dernière séance », « le plus récent » ; « la première séance »
_DERNIERE = re.compile(r"\b(?:derniere|plus recente?|recente)\b")
_PREMIERE = re.compile(r"\b(?:premiere|plus ancienne?)\b")


def lire_periode(date_ou_periode: str | None, s: Settings | None = None) -> tuple[date, date] | None:
    """parse_periode avec la dernière séance de la base comme référence des dates sans année. Sans date,
    « la dernière séance » / « la plus récente » donne le jour de la dernière séance, « la première » celui de la
    première."""
    periode = parse_periode(date_ou_periode, lambda: derniere_seance(s))
    if periode is None and date_ou_periode:
        t = norm_name(date_ou_periode)
        sql = ("SELECT max(date) FROM silver_seances" if _DERNIERE.search(t)
               else "SELECT min(date) FROM silver_seances" if _PREMIERE.search(t) else None)
        if sql and (d := query(sql, [], s)[0][0]):
            return d, d
    return periode


def _where(orateur=None, periode=None, sujet=None, types=None, alias="i") -> tuple[str, list]:
    clauses, params = [], []
    if orateur:
        clauses.append(f"{alias}.orateur_norm LIKE %s")
        params.append(f"%{norm_name(orateur)}%")
    if periode:
        clauses.append(f"{alias}.date BETWEEN %s AND %s")
        params += list(periode)
    if sujet:
        clauses.append(f"unaccent_lower({alias}.sujet) LIKE %s")
        params.append(f"%{norm_name(sujet)}%")
    if types:
        clauses.append(f"{alias}.type = ANY(%s)")
        params.append(list(types))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def query(sql: str, params: list, s: Settings | None = None) -> list[tuple]:
    with connect(s) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.commit()
        return rows


def _periode_txt(periode) -> str:
    if not periode:
        return "toutes les séances"
    a, b = periode
    return f"le {a:%d/%m/%Y}" if a == b else f"du {a:%d/%m/%Y} au {b:%d/%m/%Y}"


# ---------------------------------------------------------------------------
# Requêtes exposées à l'agent (renvoient un texte court, prêt à être lu par le LLM)
# ---------------------------------------------------------------------------
def compter_prises_de_parole(orateur: str, date_ou_periode: str | None = None, sujet: str | None = None,
                             s: Settings | None = None) -> str:
    periode = lire_periode(date_ou_periode, s)
    where, params = _where(orateur, periode, sujet, ["discours", "interruption", "presidence"])
    rows = query(f"""
        SELECT i.orateur, i.date, i.type, count(*), sum(i.nb_mots)
        FROM silver_interventions i {where}
        GROUP BY i.orateur, i.date, i.type ORDER BY i.orateur, i.date""", params, s)
    if not rows:
        return (f"Aucune prise de parole trouvée pour « {orateur} » ({_periode_txt(periode)}"
                + (f", sujet « {sujet} »" if sujet else "") + "). Vérifier l'orthographe du nom ou la date.")
    by: dict[tuple, dict] = {}
    for nom, d, typ, n, mots in rows:
        e = by.setdefault((nom, d), {"discours": 0, "interruption": 0, "presidence": 0, "mots": 0})
        e[typ] = n
        e["mots"] += mots or 0
    lines = [f"Prises de parole de « {orateur} » ({_periode_txt(periode)}"
             + (f", sujet contenant « {sujet} »" if sujet else "") + "). Une intervention = un discours, même "
             "interrompu ; les interruptions sont les exclamations lancées depuis les bancs :"]
    tot = {"discours": 0, "interruption": 0, "presidence": 0, "mots": 0}
    for (nom, d), e in by.items():
        parts = [f"{e['discours']} intervention(s)"] if e["discours"] else []
        if e["interruption"]:
            parts.append(f"{e['interruption']} interruption(s) depuis les bancs")
        if e["presidence"]:
            parts.append(f"{e['presidence']} prise(s) de parole en tant que président(e) de séance")
        lines.append(f"- {nom}, séance du {d:%d/%m/%Y} : " + ", ".join(parts) + f" ; {e['mots']} mots au total")
        for k in tot:
            tot[k] += e[k]
    if len(by) > 1:
        lines.append(f"TOTAL : {tot['discours']} interventions, {tot['interruption']} interruptions, "
                     f"{tot['presidence']} prises de parole en présidence, {tot['mots']} mots")
    return "\n".join(lines)


def classement_orateurs(date_ou_periode: str | None = None, sujet: str | None = None, compter: str = "discours",
                        limite: int = 10, s: Settings | None = None) -> str:
    periode = lire_periode(date_ou_periode, s)
    where, params = _where(None, periode, sujet, ["interruption" if compter == "interruption" else "discours"])
    rows = query(f"""
        SELECT i.orateur, max(i.qualite), count(*) AS n FROM silver_interventions i {where}
        GROUP BY i.orateur ORDER BY n DESC, i.orateur""", params, s)  # ex aequo : ordre alphabétique, stable
    if not rows:
        return f"Aucune intervention trouvée ({_periode_txt(periode)}" + (f", sujet « {sujet} »)" if sujet else ")")
    seuil = rows[min(int(limite), len(rows)) - 1][2]
    rows = [r for r in rows if r[2] >= seuil]  # les ex aequo du dernier rang sont gardés
    quoi = {"interruption": "interruptions"}.get(compter, "interventions")
    lines = [f"Orateurs classés par nombre d'{quoi} ({_periode_txt(periode)}"
             + (f", sujet contenant « {sujet} »" if sujet else "") + "). Même rang = ex aequo :"]
    for nom, q, n in rows:
        rang = 1 + sum(r[2] > n for r in rows)
        lines.append(f"{rang}. {nom}" + (f" ({q})" if q else "") + f" : {n} {quoi}")
    for n in sorted({r[2] for r in rows}, reverse=True):
        noms = [r[0] for r in rows if r[2] == n]
        if len(noms) > 1:
            lines.append(f"Ex aequo avec {n} {quoi} : {' et '.join(noms)}.")
    if len(rows) > int(limite):
        lines.append(f"À cause des ex aequo, {len(rows)} orateurs sont classés au lieu de {limite}.")
    return "\n".join(lines)


def resultats_votes(numero_amendement: str | None = None, date_ou_periode: str | None = None, sujet: str | None = None,
                    resultat: str | None = None, s: Settings | None = None) -> str:
    periode = lire_periode(date_ou_periode, s)
    clauses, params = [], []
    if numero_amendement:
        clauses.append("%s = ANY(v.numeros)")
        params.append(re.sub(r"\D", "", str(numero_amendement)) or str(numero_amendement))
    if periode:
        clauses.append("v.date BETWEEN %s AND %s")
        params += list(periode)
    if sujet:
        clauses.append("unaccent_lower(v.sujet) LIKE %s")
        params.append(f"%{norm_name(sujet)}%")
    if resultat:
        r = norm_name(resultat)  # « rejetés », « non adopté », « repoussé »… -> valeur de la table
        if r.startswith(("rejet", "non adopt", "repouss")):
            resultat = "rejeté"
        elif r.startswith("adopt"):
            resultat = "adopté"
        elif r.startswith("retir"):
            resultat = "retiré"
        clauses.append("v.resultat = %s")
        params.append(resultat)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = query(f"""
        SELECT v.date, v.objet, v.numeros, v.resultat, v.scrutin_public, v.votants, v.pour, v.contre, v.sujet
        FROM silver_votes v {where} ORDER BY v.date, v.seance_uid, v.ordre""", params, s)
    if not rows:
        return "Aucun vote trouvé avec ces critères."
    counts: dict[str, int] = {}
    for r in rows:
        counts[f"{r[1]} {r[3]}"] = counts.get(f"{r[1]} {r[3]}", 0) + 1
    lines = [f"{len(rows)} vote(s) trouvé(s) : " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))]
    for d, objet, nums, res, pub, votants, pour, contre, sujet_v in rows[:25]:
        quoi = f"{objet} n° {', '.join(nums)}" if nums else objet
        sc = f" — scrutin public : {votants} votants, {pour} pour, {contre} contre" if pub else ""
        if pub and votants is None:  # motion de censure : seuls les « pour » sont comptés
            sc = f" — scrutin public : {pour} voix pour"
        lines.append(f"- {d:%d/%m/%Y} : {quoi} {res}{sc} (sujet : {sujet_v})")
    if len(rows) > 25:
        lines.append(f"(… {len(rows) - 25} autres votes non listés)")
    return "\n".join(lines)


def presidences_seances(periode, s: Settings | None = None) -> list[tuple[str, str, int]]:
    """Qui a présidé les séances : (nom, qualité, nombre de séances présidées au moins en partie), du plus au moins.
    « Mme X, vice-présidente » et « Mme X » (sommaire du compte rendu) sont la même personne."""
    where, params = ("WHERE date BETWEEN %s AND %s", list(periode)) if periode else ("", [])
    qualites: dict[str, str] = {}
    seances: dict[str, int] = {}
    for (pres,) in query(f"SELECT presidences FROM silver_seances {where}", params, s):
        noms = set()
        for p in pres or []:
            nom, _, qualite = (x.strip() for x in p.partition(","))
            qualites[nom] = qualites.get(nom) or qualite
            noms.add(nom)
        for nom in noms:
            seances[nom] = seances.get(nom, 0) + 1
    return sorted(((nom, qualites[nom], n) for nom, n in seances.items()), key=lambda t: (-t[2], t[0]))


def resultats_elections(date_ou_periode: str | None = None, s: Settings | None = None) -> str:
    """Élections et nominations (président, vice-présidents, questeurs, secrétaires, juges…), puis qui a présidé les
    séances. Les élus sont listés à part, en tête, pour que le LLM ne prenne pas les candidats pour des élus."""
    periode = lire_periode(date_ou_periode, s)
    where, params = ("WHERE date BETWEEN %s AND %s", list(periode)) if periode else ("", [])
    rows = query(f"""SELECT date, objet, tour, votants, majorite_absolue, candidats, elus FROM silver_elections {where}
                     ORDER BY date, seance_uid, ordre""", params, s)
    pres = presidences_seances(periode, s)
    if not rows and not pres:
        return f"Aucune élection ni nomination dans la base ({_periode_txt(periode)})."
    lines = []
    if rows:
        lines.append(f"ÉLUS ({_periode_txt(periode)}) — seules ces personnes ont été élues :")
        for d, objet, tour, _, _, _, elus in rows:
            if elus:
                n = f"{tour}{'er' if tour == 1 else 'e'}"
                lines.append(f"- {objet} : {', '.join(elus)} (le {d:%d/%m/%Y}, {n} tour)")
    else:
        lines.append(f"Aucune élection ni nomination dans la base ({_periode_txt(periode)}).")
    if pres:
        lines.append("")
        lines.append("PRÉSIDENCE DES SÉANCES (qui a présidé les séances ; ce n'est pas une élection) :")
        for nom, qualite, n in pres:
            lines.append(f"- {nom}{f' ({qualite})' if qualite else ''} : {n} séance(s) présidée(s), en tout ou partie")
    if rows:
        lines.append("")
        lines.append(f"DÉTAIL DES {len(rows)} TOUR(S) DE SCRUTIN (les candidats non élus n'ont PAS été élus) :")
        for d, objet, tour, votants, majorite, candidats, elus in rows:
            # les élus d'abord, avec leur nombre : en fin de longue ligne, le LLM n'en recopie qu'une partie
            issue = f"{len(elus)} élu(s) à ce tour : {', '.join(elus)}" if elus else "personne n'est élu à ce tour"
            maj = f", majorité absolue {majorite}" if majorite else ""
            voix = ", ".join(f"{c['nom']} {c['voix']}" for c in candidats)
            lines.append(f"- {d:%d/%m/%Y}, {objet}, {tour}{'er' if tour == 1 else 'e'} tour : {issue}. "
                         f"Candidats ({votants} votants{maj}) : {voix}")
    return "\n".join(lines)


def lister_seances(date_ou_periode: str | None = None, s: Settings | None = None) -> str:
    periode = lire_periode(date_ou_periode, s)
    where, params = ("WHERE se.date BETWEEN %s AND %s", list(periode)) if periode else ("", [])
    rows = query(f"""
        SELECT se.date, se.heure_debut, se.num_seance, se.presidences, se.nb_discours, se.nb_interruptions, se.nb_votes,
               (SELECT string_agg(DISTINCT split_part(i.sujet, ' > ', 1), ' ; ') FROM silver_interventions i
                WHERE i.seance_uid = se.uid AND i.type = 'discours')
        FROM silver_seances se {where} ORDER BY se.date, se.heure_debut LIMIT 40""", params, s)
    if not rows:
        return f"Aucune séance dans la base ({_periode_txt(periode)})."
    lines = [f"{len(rows)} séance(s) ({_periode_txt(periode)}) :"]
    for d, h, num, pres, nd, ni, nv, sujets in rows:
        lines.append(f"- {d:%d/%m/%Y} à {h} (séance n° {num}), présidence : {', '.join(pres or [])} ; "
                     f"{nd} interventions, {ni} interruptions, {nv} votes ; sujets : {sujets or '-'}")
    return "\n".join(lines)


def sujets_abordes(date_ou_periode: str | None = None, orateur: str | None = None, limite: int = 10,
                   s: Settings | None = None) -> str:
    """Points de l'ordre du jour les plus débattus (en nombre de mots prononcés), avec leurs séances. Les questions
    au Gouvernement comptent sujet par sujet (« Questions au Gouvernement > Effectifs de police à Marseille ») :
    regroupées, elles écraseraient tout le reste sans rien dire des thèmes."""
    periode = lire_periode(date_ou_periode, s)
    where, params = _where(orateur, periode, None, ["discours"])
    rows = query(rf"""
        WITH p AS (
            SELECT CASE WHEN i.sujet ILIKE 'questions au gouvernement%%' THEN split_part(i.sujet, ' > ', 2)
                        ELSE regexp_replace(split_part(i.sujet, ' > ', 1), '\s*\(suite\)$', '') END AS point,
                   i.sujet ILIKE 'questions au gouvernement%%' AS qag, i.seance_uid, i.date, i.nb_mots
            FROM silver_interventions i {where})
        SELECT point, qag, sum(nb_mots), count(*), count(DISTINCT seance_uid), min(date), max(date)
        FROM p WHERE point <> '' AND point !~* '^(suspension et reprise|rappels? au r|reprise de la discussion|fait personnel|ordre du jour|ouverture de la s|lev[ée]e de la s|clôture|modification de l.ordre du jour)'
        GROUP BY point, qag ORDER BY sum(nb_mots) DESC LIMIT %s""", params + [int(limite)], s)
    criteres = _periode_txt(periode) + (f", orateur « {orateur} »" if orateur else "")
    if not rows:
        return f"Aucun discours trouvé ({criteres})."
    lines = [f"Les {len(rows)} sujets les plus débattus ({criteres}), classés par nombre de mots prononcés :"]
    for k, (point, qag, mots, nb, nbs, d1, d2) in enumerate(rows, 1):
        quand = f"le {d1:%d/%m/%Y}" if d1 == d2 else f"du {d1:%d/%m/%Y} au {d2:%d/%m/%Y}"
        lines.append(f"{k}. {point}{' (question au Gouvernement)' if qag else ''} : {nb} interventions, "
                     f"{mots} mots, {nbs} séance(s), {quand}")
    return "\n".join(lines)


# Mots vides (sans accents, comparés à unaccent_lower(mot)) : outils grammaticaux, auxiliaires, formules d'adresse
MOTS_VIDES = sorted(set("""
les des que est vous nous qui pour une pas dans sur par plus cette notre aux sont mais avec ont ces nos avez votre
elle ete comme meme etre tout bien fait une son ses leur leurs lui ils elles eux cela ceci celui celle ceux celles
cet sans sous entre vers chez donc alors car aussi ainsi encore deja tres trop peu tous toutes toute autre autres
avons avait avaient aurait auraient sera seront serait seraient etait etaient etes fut faire font faut doit
doivent peut peuvent pouvons pourrait dont quand comment pourquoi quoi quel quelle quels quelles lorsque parce
puis apres avant depuis pendant jamais toujours ici oui non rien tant chaque meme memes sommes suis vos mes tes
mon ton ma ta sa moi toi soi uns aujourd hui avoir aura auront eu ait aient soit soient voici voila notamment
devons devez voulons voulez veux veut pouvez peux dit dire
monsieur madame mesdames messieurs cher chers chere cheres collegue collegues
""".split()))


def mots_frequents(date_ou_periode: str | None = None, orateur: str | None = None, sujet: str | None = None,
                   limite: int = 10, s: Settings | None = None) -> str:
    """Mots les plus employés dans les discours (pas les interruptions), sans les mots vides : ts_stat de
    PostgreSQL sur le texte découpé en mots (dictionnaire « simple » : ni racinisation, ni accents retirés)."""
    periode = lire_periode(date_ou_periode, s)
    where, params = _where(orateur, periode, sujet, ["discours"])
    with connect(s) as conn, conn.cursor() as cur:
        textes = cur.mogrify(f"SELECT to_tsvector('simple', i.texte) FROM silver_interventions i {where}", params)
        cur.execute("""SELECT word, nentry, ndoc FROM ts_stat(%s)
                       WHERE length(word) >= 3 AND word !~ '^[0-9]+$' AND unaccent_lower(word) <> ALL(%s)
                       ORDER BY nentry DESC, word LIMIT %s""", [textes.decode(), MOTS_VIDES, int(limite)])
        rows = cur.fetchall()
    criteres = _periode_txt(periode) + (f", orateur « {orateur} »" if orateur else "") + (
        f", sujet contenant « {sujet} »" if sujet else "")
    if not rows:
        return f"Aucun discours trouvé ({criteres})."
    lines = [f"Les {len(rows)} mots les plus employés dans les discours ({criteres}), hors mots vides (articles, "
             "pronoms, auxiliaires, « monsieur », « madame »…) :"]
    lines += [f"{1 + sum(r[1] > n for r in rows)}. « {mot} » : {n} fois, dans {nd} discours"  # même rang = ex aequo
              for mot, n, nd in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Résumé d'une séance : son plan, ses orateurs, ses élections et ses votes
# ---------------------------------------------------------------------------
# Titres de procédure : ni des sujets, ni des étapes du débat
_PROCEDURE = re.compile(r"^(Suspension et reprise|Rappels? au règlement|Reprise de la discussion|Fait personnel|"
                        r"Ordre du jour de la prochaine séance)")
RESUME_MAX_CARACTERES = 7000  # ~1 800 tokens : la fiche doit laisser de la place à la réponse du LLM


def _point(sujet: str) -> str:
    """« Projet de loi de finances pour 2025 (suite) > Article 2 » -> « Projet de loi de finances pour 2025 »."""
    return sujet.split(" > ")[0].removesuffix(" (suite)")


DECISIONS = "Décisions de la séance (résultats enregistrés au compte rendu) :"


def _fiche(uid: str, titre: str, s: Settings | None) -> str:
    """Fiche d'une séance en deux parties : le plan (points, sous-points, orateurs), à faire rédiger par le LLM,
    puis les décisions (élections, votes), à afficher telles quelles pour que les chiffres restent exacts."""
    points: dict[str, dict] = {}  # point de niveau 1 -> ce qui s'y rattache, dans l'ordre de la séance

    def point(sujet: str) -> dict:
        return points.setdefault(_point(sujet), {"sous": [], "orateurs": [], "decisions": [], "amendements": []})

    for niveau, t in query("SELECT niveau, titre FROM silver_points WHERE seance_uid = %s ORDER BY ordre", [uid], s):
        if _PROCEDURE.match(t):
            continue
        if niveau == 1:
            courant = point(t)
        elif points and t not in courant["sous"]:
            courant["sous"].append(t)
    for sujet, nom, qualite in query("""
            SELECT sujet, orateur, max(qualite) FROM silver_interventions WHERE seance_uid = %s AND type = 'discours'
            GROUP BY sujet, orateur ORDER BY sum(nb_mots) DESC, sujet, orateur""", [uid], s):
        point(sujet)["orateurs"].append(nom + (f" ({qualite.split(',')[0]})" if qualite else ""))
    for sujet, objet, tour, votants, candidats, elus in query("""
            SELECT sujet, objet, tour, votants, candidats, elus FROM silver_elections WHERE seance_uid = %s
            ORDER BY ordre""", [uid], s):
        # « l’élection des vice-présidents de l’Assemblée nationale » -> « élection des vice-présidents »
        objet = re.sub(r"^(l’|la |le |les )|\s*de l’Assemblée nationale$", "", objet or "élection")
        voix = ", ".join(f"{c['nom']} {c['voix']}" for c in candidats[:6])
        issue = f"élu(s) : {', '.join(elus)}" if elus else "personne n'est élu"
        point(sujet)["decisions"].append(
            f"{objet}, {tour}{'er' if tour == 1 else 'e'} tour ({votants} votants) : {voix} → {issue}")
    for sujet, objet, nums, res, pub, pour, contre in query("""
            SELECT sujet, objet, numeros, resultat, scrutin_public, pour, contre FROM silver_votes
            WHERE seance_uid = %s ORDER BY ordre""", [uid], s):
        if objet in ("amendement", "sous_amendement"):  # trop nombreux : seulement comptés
            point(sujet)["amendements"].append(res)
        else:
            decompte = f" ({pour} pour, {contre} contre)" if pub else ""
            point(sujet)["decisions"].append(f"{objet} {' '.join(nums)} {res}{decompte}")

    plan, decisions = [titre, "Ordre du jour :"], [DECISIONS]
    for k, (nom, p) in enumerate(points.items(), 1):
        plan.append(f"{k}. {nom}")
        if p["sous"]:
            plan.append("   - sous-points : " + " ; ".join(p["sous"][:20]))
        if p["orateurs"]:
            noms = list(dict.fromkeys(p["orateurs"]))
            plan.append(f"   - {len(noms)} orateurs, dont : " + ", ".join(noms[:4]))
        if p["amendements"]:
            a = p["amendements"]
            detail = ", ".join(f"{a.count(r)} {r}(s)" for r in dict.fromkeys(a))
            p["decisions"].append(f"amendements : {len(a)} ({detail})")
        if p["decisions"]:
            decisions += [f"- {nom} :"] + [f"   - {d}" for d in p["decisions"]]
    if len(decisions) == 1:
        decisions.append("- aucune élection ni aucun vote")
    return "\n".join(plan) + "\n\n" + "\n".join(decisions)


def resume_seance(date_ou_periode: str | None, s: Settings | None = None) -> str:
    periode = lire_periode(date_ou_periode, s)
    if not periode or periode[0] != periode[1]:
        return "Préciser UNE date de séance (ex. « 18 juillet 2024 »)."
    seances = query("""SELECT uid, date_libelle, heure_debut, heure_fin, presidences FROM silver_seances
                       WHERE date = %s ORDER BY heure_debut""", [periode[0]], s)
    if not seances:
        return f"Aucune séance le {periode[0]:%d/%m/%Y} dans la base."
    fiches = []
    for k, (uid, jour, debut, fin, presidences) in enumerate(seances, 1):
        titre = (f"=== SÉANCE {k} sur {len(seances)} : {jour}, de {debut} à {fin or '?'} — présidence : "
                 + ", ".join(dict.fromkeys(p.split(",")[0] for p in presidences or [])) + " ===")
        fiches.append(_fiche(uid, titre, s))
    return "\n\n".join(fiches)[:RESUME_MAX_CARACTERES]
