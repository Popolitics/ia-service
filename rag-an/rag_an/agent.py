"""Agent : le LLM choisit un outil et ses paramètres, Python l'exécute, puis le LLM rédige la réponse.

Routeur plutôt qu'appel de fonctions natif (FunctionAgent) : avec ce dernier, Mistral 7B écrit l'appel d'outil
dans sa réponse au lieu de le déclencher. Ici il remplit un JSON imposé par un schéma (sortie structurée
d'Ollama). Les chiffres viennent de requêtes SQL paramétrées (stats.py), jamais de SQL écrit par le LLM.
"""
import logging
import re
from typing import Literal

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from . import stats
from .config import Settings, get_settings
from .engine import RULES, SYSTEM_PROMPT, build_index, build_postprocessors, build_retriever

TOOL_GUIDE = (
    "- Question de chiffrage (combien, nombre de fois, qui a le plus parlé ou interrompu, classement, votes "
    "adoptés ou rejetés, liste des séances) : compter_prises_de_parole, classement_orateurs, resultats_votes "
    "ou lister_seances. Ces outils ne comptent que des prises de parole, des votes et des séances : un chiffre "
    "cité dans un discours (effectifs, montants, nombre de postes…) se cherche avec recherche_semantique.\n"
    "- Question sur les mots les plus employés, le vocabulaire d'une séance, d'une période ou d'un orateur : "
    "mots_frequents (limite = le nombre de mots demandé).\n"
    "- Question sur les sujets, thèmes ou textes les plus abordés ou débattus sur une période ou par un orateur "
    "(« de quoi a-t-on le plus parlé en 2025 ? ») : sujets_abordes. Les MOTS relèvent de mots_frequents, les "
    "SUJETS de sujets_abordes.\n"
    "- Question sur une élection ou une nomination (qui a été élu président, vice-président, questeur, secrétaire, "
    "juge…), ou sur la présidence (les présidents ou présidentes de l'Assemblée, qui préside ou a présidé les "
    "séances) : resultats_elections.\n"
    "- Question sur un terme exact (nom propre, numéro, sigle, expression entre guillemets) : recherche_lexicale.\n"
    "- Question sur le déroulé d'une séance entière à une date (que s'est-il passé, de quoi a-t-on parlé, ordre "
    "du jour, résumé de la séance) : resume_seance ; pour la séance la plus récente, date_ou_periode = « dernière "
    "séance ».\n"
    "- Question sur un sujet, une position, des arguments, ce qui a été dit sur un thème : recherche_semantique.\n"
    "- Si la question précise une date, une période ou un orateur, renseigne-les.\n"
    "- Question de suivi (« Et … ? », « Et lui ? ») : elle porte sur le sujet de la question précédente de "
    "l'historique ; choisis l'outil comme si elle était posée en entier."
)

ROUTER_PROMPT = PromptTemplate(
    "Tu orientes les questions posées sur les débats de l'Assemblée nationale vers le bon outil.\n\n"
    "Outils :\n"
    "- compter_prises_de_parole : combien de fois un orateur a parlé / interrompu (orateur obligatoire).\n"
    "- classement_orateurs : qui a le plus parlé ou le plus interrompu (compter = discours ou interruption).\n"
    "- resultats_votes : résultat d'un vote, amendements adoptés ou rejetés, décompte d'un scrutin.\n"
    "- resultats_elections : qui a été élu (président, vice-présidents, questeurs, secrétaires, juges…), voix "
    "par tour, et qui a présidé les séances.\n"
    "- lister_seances : quelles séances ont eu lieu à une date ou une période.\n"
    "- mots_frequents : mots les plus employés dans les discours (d'une séance, d'une période, d'un orateur).\n"
    "- sujets_abordes : sujets (points de l'ordre du jour, questions au Gouvernement) les plus débattus sur une "
    "période ou par un orateur.\n"
    "- resume_seance : déroulé d'une séance à une date précise (ordre du jour, élections, votes, orateurs).\n"
    "- recherche_lexicale : retrouver un terme exact (nom propre, numéro, sigle, expression).\n"
    "- recherche_semantique : sujet, position, arguments, ce qui a été dit sur un thème.\n\n"
    "Règles :\n" + TOOL_GUIDE + "\n"
    "- requete : pour une recherche, la question reformulée de façon autonome (pronoms remplacés grâce à "
    "l'historique) ; pour recherche_lexicale, uniquement les mots exacts à retrouver.\n"
    "- date_ou_periode : telle qu'écrite dans la question (« 18 juillet 2024 », « octobre 2024 », « du 21 au 25 "
    "octobre 2024 ») ou null.\n"
    "- limite : le nombre demandé (« les 5 mots », « le top 3 ») ou null.\n"
    "- Laisse à null tout paramètre que la question ne donne pas.\n\n"
    "Exemples :\n"
    "« Combien de fois Mme Panot est-elle intervenue en octobre 2024 ? » -> compter_prises_de_parole, "
    "orateur=Panot, date_ou_periode=octobre 2024\n"
    "« Qui a le plus interrompu le 22 octobre 2024 ? » -> classement_orateurs, compter=interruption, "
    "date_ou_periode=22 octobre 2024\n"
    "« Quels sont les 5 mots les plus utilisés le 1er juillet 2025 ? » -> mots_frequents, limite=5, "
    "date_ou_periode=1er juillet 2025\n"
    "« Quels mots M. Retailleau emploie-t-il le plus en octobre 2024 ? » -> mots_frequents, orateur=Retailleau, "
    "date_ou_periode=octobre 2024\n"
    "« Quels sont les sujets les plus abordés en 2025 ? » -> sujets_abordes, date_ou_periode=2025\n"
    "« De quoi a parlé la séance la plus récente ? » -> resume_seance, date_ou_periode=dernière séance\n"
    "« L'amendement 512 a-t-il été adopté ? » -> resultats_votes, numero_amendement=512\n"
    "« Qui a été élu vice-président le 19 juillet 2024 ? » -> resultats_elections, date_ou_periode=19 juillet 2024\n"
    "« Quels présidents ou présidentes de l'Assemblée trouve-t-on dans les documents ? » -> resultats_elections\n"
    "« Combien de logements sociaux le ministre a-t-il promis ? » -> recherche_semantique, requete=Combien de "
    "logements sociaux le ministre a-t-il promis ?\n"
    "Après une question sur le budget des armées : « Et pour la marine ? » -> recherche_semantique, requete=Que "
    "prévoit le budget des armées pour la marine ?\n"
    "« Que s'est-il passé à l'Assemblée le 19 juillet 2024 ? » -> resume_seance, date_ou_periode=19 juillet 2024\n"
    "« Que propose le RN sur les retraites ? » -> recherche_semantique, requete=Que propose le Rassemblement "
    "national sur les retraites ?\n\n"
    "Historique récent :\n{history}\n\nQuestion : {question}\n"
)

ANSWER_STATS_PROMPT = (
    "Résultat de la base de données des séances (fiable et complet pour les critères indiqués) :\n"
    "---------------------\n{result}\n---------------------\n"
    "Réponds à la question en une ou quelques phrases, en reprenant EXACTEMENT les chiffres et les noms "
    "ci-dessus, sans en calculer ni en inventer d'autres. Signale les ex aequo quand le résultat en indique (même "
    "rang) ; n'en invente pas. Pour un vote, donne sa date et, s'il y a eu un scrutin public, son décompte. Si le "
    "résultat ne permet pas de répondre, dis-le.\n\n"
    "Question : {question}"
)

ANSWER_RESUME_PROMPT = (
    "Plan d'UNE séance, tiré de la base des comptes rendus :\n---------------------\n{result}\n---------------------\n"
    "Résume ce plan en un paragraphe : la date, les horaires et la présidence de la séance, puis les points de "
    "l'ordre du jour dans l'ordre, avec leurs principaux orateurs. Rédige des phrases sans recopier les lignes. "
    "Ne donne AUCUN résultat d'élection ni de vote : ils sont affichés à part, après ton paragraphe. Le plan ne "
    "dit rien du contenu des débats : ne prête à personne des propos ou des arguments.\n\n"
    "Question : {question}"
)

ANSWER_SEARCH_PROMPT = (
    "Voici des extraits de documents de l'Assemblée nationale, numérotés [n] avec leur contexte (séance, "
    "sujet, orateurs).\n---------------------\n{result}\n---------------------\n" + RULES + "\n\nQuestion : {question}"
)

_MOIS = r"(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre)"
_JOUR = rf"\d{{1,2}}(?:er)?(?: {_MOIS})?(?: \d{{4}})?"
# Formes de dates comprises par stats.parse_periode : « du 21 au 25 octobre 2024 », « 18/07/2024 », « juillet 2024 »…
DATE_RE = re.compile(rf"\b(?:(?:semaine )?du {_JOUR} au {_JOUR}|entre (?:le )?{_JOUR} et (?:le )?{_JOUR}"
                     rf"|(?:semaine du )?\d{{1,2}}(?:er)? {_MOIS}(?: \d{{4}})?|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}"
                     rf"|\d{{4}}-\d{{2}}-\d{{2}}|{_MOIS} \d{{4}}|(?:printemps|été|automne|hiver) (?:\d{{4}}|dernier))\b",
                     re.I)
# Année seule : « en 2025 », « séances de 2025 », mais pas « loi de finances pour 2025 » (débattue en 2024)
ANNEE_RE = re.compile(r"\b(?:en|l'année|année|séances? de|débats? de|courant) (\d{4})\b", re.I)
# Textes dont le titre porte une année qui n'est pas celle des débats (budget, PLFSS pour 2025, débattus en 2024) :
# dans « Que prévoit le budget en 2025 ? », filtrer la recherche sur 2025 raterait le texte.
TEXTE_ANNEE_RE = re.compile(r"\b(?:budgets?|budgétaire|finances|financement|PLFS?S?|sécurité sociale|comptes|"
                            r"loi de programmation|LPM|programmation militaire)\b", re.I)
# … sauf si la question vise le moment des débats : « dans les séances de 2025 », « débattu en 2025 »
MOMENT_RE = re.compile(r"\b(?:séances?|débats?|débattue?s?|discussions?|discutée?s?|courant) (?:de |en |du )?"
                       r"(?:l'année )?\d{4}\b", re.I)
DERNIERE_RE = re.compile(r"\b(?:dernière séance|séance la plus récente|plus récente séance|dernière en date)\b",
                         re.I)
# Relance qui prolonge la question précédente (« Et les sujets ? », « a-t-il… ») : seule une relance hérite de
# sa date ; une nouvelle question courte ne doit pas être filtrée sur la date de la précédente.
RELANCE_RE = re.compile(r"^\s*(?:et|mais|plutôt|non|pareil|même chose|idem|je voudrais|je veux|maintenant|aussi)\b"
                        r"|-t-(?:il|elle|ils|elles)\b|\b(?:il|elle|lui|eux|ce jour-là|cette séance|ce mois-ci)\b",
                        re.I)

log = logging.getLogger(__name__)

# Quand le texte déborde de la fenêtre, Ollama ne signale rien : il retire les plus vieux messages, voire coupe le
# DÉBUT du dernier (consigne système et meilleurs extraits). L'agent fait donc ses comptes avant d'envoyer.
# Estimation prudente : 3 caractères par token (mesuré sur du français avec Mistral).
CARACTERES_PAR_TOKEN = 3
MARGE_TOKENS = 200  # balises de rôle et imprécision de l'estimation


def estimer_tokens(messages: list[ChatMessage]) -> int:
    return sum(len(str(m.content)) // CARACTERES_PAR_TOKEN + 4 for m in messages)


_EX_AEQUO = re.compile(r"ex[ -]?(?:a|æ)equo|ex[ -]?æquo|à égalité", re.I)

OUTILS = Literal["compter_prises_de_parole", "classement_orateurs", "resultats_votes", "resultats_elections",
                 "lister_seances", "mots_frequents", "sujets_abordes", "resume_seance", "recherche_lexicale", "recherche_semantique"]


class Choix(BaseModel):
    """Outil choisi et ses paramètres (rempli par le LLM, au format imposé)."""

    outil: OUTILS
    requete: str | None = Field(None, description="question reformulée, ou mots exacts pour recherche_lexicale")
    orateur: str | None = None
    date_ou_periode: str | None = None
    sujet: str | None = None
    numero_amendement: str | None = None
    resultat: Literal["adopté", "rejeté", "retiré"] | None = None
    compter: Literal["discours", "interruption"] | None = None
    limite: int | None = Field(None, description="nombre de résultats demandé (« les 5 mots », « le top 3 »)")


# --- nombres en toutes lettres -------------------------------------------------------------------------
# Mistral 7B lit mal les nombres en lettres (« soixante-quinze » lu 65) : les extraits envoyés au LLM portent le
# chiffre à côté, « soixante-quinze (75) ». Le texte stocké et indexé ne change pas.
_UNITES = {"zéro": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7,
           "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
           "seize": 16, "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60}
# « million(s) » et « milliard(s) » ne sont pas annotés : « 40 milliards » donnerait « 40 milliards (1 000 000 000) »
_MULT = {"cent": 100, "cents": 100, "mille": 1000}
_MOT_NB = "|".join(sorted([*_UNITES, *_MULT], key=len, reverse=True))
_ET = r"(?:-| )et(?:-| )(?=une?\b|onze\b)"  # « vingt et un », « soixante-et-onze », pas « cinq et six »
_NOMBRE_RE = re.compile(rf"(?<![\w-])(?:{_MOT_NB})(?:(?:-| |{_ET})(?:{_MOT_NB}))*(?![\w-])", re.I)


def _valeur(mots: list[str]) -> int:
    """[« quatre », « vingt », « dix », « sept »] -> 97."""
    total = courant = 0
    for i, m in enumerate(mots):
        if m in ("vingt", "vingts") and i and mots[i - 1] == "quatre":
            courant += 76  # quatre (4) déjà ajouté : 4 + 76 = 80
        elif m in _UNITES:
            courant += _UNITES[m]
        elif m in ("cent", "cents"):
            courant = (courant or 1) * 100
        else:  # mille
            total += (courant or 1) * 1000
            courant = 0
    return total + courant


def chiffres_en_clair(texte: str) -> str:
    """Ajoute le chiffre après les nombres en lettres de deux mots ou plus, ou valant au moins 20 (« vingt (20) »)."""

    def remplace(m: re.Match) -> str:
        mots = [w for w in re.split(r"[- ]", m.group(0).lower()) if w != "et"]
        if mots[0] in ("un", "une") and len(mots) > 1 and mots[1] not in _MULT:
            return m.group(0)  # « un deux » n'est pas un nombre
        v = _valeur(mots)
        if len(mots) < 2 and (v < 20 or mots[0] in _MULT):  # « trois », « neuf », « cent » ou « mille » seuls
            return m.group(0)
        return f"{m.group(0)} ({v:,})".replace(",", " ") if v >= 10_000 else f"{m.group(0)} ({v})"

    return _NOMBRE_RE.sub(remplace, texte)


def _llm_text(n: NodeWithScore) -> str:
    """Extrait tel que vu par le LLM, sans « source_n » (déjà en tête : [n]), sinon il cite « [1, source_n: 1] »."""
    node = n.node
    hidden = node.excluded_llm_metadata_keys
    added = "source_n" not in hidden
    if added:
        hidden.append("source_n")
    try:
        return chiffres_en_clair(node.get_content(metadata_mode=MetadataMode.LLM))
    finally:
        if added:
            hidden.remove("source_n")


class RagAgent:
    def __init__(self, s: Settings | None = None, llm=None):
        from llama_index.core import Settings as LISettings

        self.s = s or get_settings()
        self.index = build_index(self.s)
        self.llm = llm or LISettings.llm
        self.nodes: list[NodeWithScore] = []  # extraits fournis au LLM pour la question en cours

    # --- outils ---------------------------------------------------------------------------------
    def search(self, query: str, mode: str, date: str | None, orateur: str | None) -> str:
        """Recherche de passages (mode « hybride » ou « lexicale ») : extraits numérotés [n] pour le LLM."""
        retriever = build_retriever(self.index, self.s, mode=mode, date_ou_periode=date, orateur=orateur)
        qb = QueryBundle(query)
        known = {n.node.node_id for n in self.nodes}
        nodes = [n for n in retriever.retrieve(qb) if n.node.node_id not in known]
        for pp in build_postprocessors(self.s, numbering_start=len(self.nodes) + 1):
            nodes = pp.postprocess_nodes(nodes, qb)
        self.nodes += nodes
        if not nodes:
            filtres = [f"date « {date} »" if date else "", f"orateur « {orateur} »" if orateur else ""]
            filtres = ", ".join(f for f in filtres if f)
            if filtres:
                return f"Aucun extrait trouvé avec les filtres {filtres} (vérifier la date ou le nom)."
            return "Aucun extrait trouvé."
        return "\n\n".join(f"[{n.node.metadata['source_n']}] " + _llm_text(n) for n in nodes)

    @staticmethod
    def complete(c: Choix, question: str, history: list[ChatMessage] | None = None) -> Choix:
        """Corrige le choix du LLM : date oubliée, année d'un titre de texte, orateur manquant."""
        if not c.date_ou_periode:
            # date oubliée : cherchée dans la question, la requête reformulée, puis la question précédente (relance)
            textes = [question, c.requete or ""]
            precedentes = [str(m.content) for m in (history or []) if m.role == MessageRole.USER]
            if precedentes and len(question.split()) <= 12 and RELANCE_RE.search(question):
                textes.append(precedentes[-1])
            for texte in textes:
                if m := DATE_RE.search(texte) or ANNEE_RE.search(texte) or DERNIERE_RE.search(texte):
                    c = c.model_copy(update={"date_ou_periode": m.group(m.lastindex or 0)})
                    break
        if (c.outil.startswith("recherche") and c.date_ou_periode
                and re.fullmatch(r"(?:en |l'année |année |pour )?\d{4}", c.date_ou_periode.strip(), re.I)
                and TEXTE_ANNEE_RE.search(f"{question} {c.requete or ''}")
                and not MOMENT_RE.search(question)):
            c = c.model_copy(update={"date_ou_periode": None})  # année du texte, pas des débats
        if c.outil == "compter_prises_de_parole" and not c.orateur:  # sans orateur : on classe tout le monde
            c = c.model_copy(update={"outil": "classement_orateurs"})
        return c

    def call(self, c: Choix, question: str) -> tuple[str, dict]:
        """Exécute l'outil choisi ; renvoie son résultat et les paramètres utilisés."""
        date, s = c.date_ou_periode, self.s
        match c.outil:
            case "recherche_semantique" | "recherche_lexicale":
                args = {"requete": c.requete or question, "date": date, "orateur": c.orateur}
                mode = "lexicale" if c.outil == "recherche_lexicale" else "hybride"
                return self.search(args["requete"], mode, date, c.orateur), args
            case "compter_prises_de_parole":
                args = {"orateur": c.orateur, "date_ou_periode": date, "sujet": c.sujet}
                return stats.compter_prises_de_parole(c.orateur, date, c.sujet, s), args
            case "classement_orateurs":
                args = {"date_ou_periode": date, "sujet": c.sujet, "compter": c.compter or "discours",
                        "limite": c.limite or 10}
                return stats.classement_orateurs(date, c.sujet, args["compter"], args["limite"], s), args
            case "resultats_votes":
                args = {"numero_amendement": c.numero_amendement, "date_ou_periode": date, "sujet": c.sujet,
                        "resultat": c.resultat}
                return stats.resultats_votes(c.numero_amendement, date, c.sujet, c.resultat, s), args
            case "resultats_elections":
                return stats.resultats_elections(date, s), {"date_ou_periode": date}
            case "lister_seances":
                return stats.lister_seances(date, s), {"date_ou_periode": date}
            case "mots_frequents":
                args = {"date_ou_periode": date, "orateur": c.orateur, "sujet": c.sujet, "limite": c.limite or 10}
                return stats.mots_frequents(date, c.orateur, c.sujet, args["limite"], s), args
            case "sujets_abordes":
                args = {"date_ou_periode": date, "orateur": c.orateur, "limite": c.limite or 10}
                return stats.sujets_abordes(date, c.orateur, args["limite"], s), args
            case "resume_seance":
                return stats.resume_seance(date, s), {"date_ou_periode": date}

    # --- réponse --------------------------------------------------------------------------------
    def choisir(self, question: str, history: list[ChatMessage]) -> Choix:
        """Outil et paramètres choisis par Mistral, puis corrigés par complete()."""
        return self.complete(self.route(question, history), question, history)

    def route(self, question: str, history: list[ChatMessage]) -> Choix:
        hist = "\n".join(f"{m.role.value}: {str(m.content)[:300]}" for m in history) or "(aucun)"
        return self.llm.structured_predict(Choix, ROUTER_PROMPT, question=question, history=hist)

    def write(self, prompt: str, history: list[ChatMessage], on_token) -> str:
        """Fait rédiger le LLM, en transmettant chaque morceau de texte à on_token (affichage au fil de l'eau)."""
        n = self.s.history_max_chars
        history = [ChatMessage(role=m.role, content=str(m.content)[:n] + ("…" if len(str(m.content)) > n else ""))
                   for m in history]
        systeme = ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
        question = ChatMessage(role=MessageRole.USER, content=prompt)
        # on retire les plus vieux échanges plutôt que de laisser Ollama couper sans prévenir
        place = self.s.llm_context_window - self.s.llm_max_tokens - MARGE_TOKENS
        retires = 0
        while history and estimer_tokens([systeme, *history, question]) > place:
            history = history[2:] if len(history) >= 2 else []  # un échange = question + réponse
            retires += 2
        if retires:
            log.warning("Fenêtre de contexte : %d message(s) d'historique les plus anciens non transmis", retires)
        if (total := estimer_tokens([systeme, question])) > place:
            log.warning("Fenêtre de contexte : la demande seule fait ~%d tokens pour %d disponibles ; Ollama en "
                        "coupera le début", total, place)
        messages = [systeme, *history, question]
        out = []
        for chunk in self.llm.stream_chat(messages):
            if chunk.delta:
                out.append(chunk.delta)
                if on_token:
                    on_token(chunk.delta)
        return "".join(out)

    def write_stats(self, result: str, question: str, history: list[ChatMessage], on_token) -> str:
        """Réponse à partir d'un résultat chiffré, rédigée avant affichage pour contrôler les ex aequo.

        Mistral invente des ex aequo absents du résultat : la réponse est alors rédigée de nouveau avec un
        avertissement ; si l'invention persiste, les phrases qui la portent sont retirées.
        """
        prompt = ANSWER_STATS_PROMPT.format(result=result, question=question)
        answer = self.write(prompt, history, None)
        if _EX_AEQUO.search(answer) and not _EX_AEQUO.search(result):
            prompt += ("\n\nATTENTION : le résultat ne contient AUCUN ex aequo. N'écris pas « ex aequo » et ne "
                       "rapproche pas des personnes qui ont des chiffres différents.")
            answer = self.write(prompt, history, None)
            if _EX_AEQUO.search(answer):
                answer = " ".join(p for p in re.split(r"(?<=[.!?])\s+", answer) if not _EX_AEQUO.search(p))
        if on_token:
            on_token(answer)
        return answer

    def ask(self, question: str, history: list[ChatMessage] | None = None, on_token=None, on_tool=None
            ) -> tuple[str, list[NodeWithScore]]:
        """Réponse et extraits fournis au LLM. on_tool(nom, paramètres) est appelé pour chaque outil exécuté."""
        history = (history or [])[-self.s.history_messages:] if self.s.history_messages > 0 else []
        self.nodes = []
        c = self.choisir(question, history)
        try:
            result, args = self.call(c, question)
        except stats.DateInvalide as e:  # « 31 juin » : on le dit, plutôt que de chercher sans la date
            if on_tool:
                on_tool(c.outil, {"date_ou_periode": c.date_ou_periode})
            result = f"{e} Cette date n'existe pas : aucune séance n'a pu avoir lieu ce jour-là."
            return self.write(ANSWER_STATS_PROMPT.format(result=result, question=question), history, on_token), []
        if on_tool:
            on_tool(c.outil, args)
        if c.outil.startswith("recherche") and not self.nodes and (c.date_ou_periode or c.orateur):
            # filtre trop strict (date ou nom mal compris) : on retente sans filtre
            result, args = self.call(c.model_copy(update={"date_ou_periode": None, "orateur": None}), question)
            if on_tool:
                on_tool(c.outil, args)

        if c.outil in ("mots_frequents", "classement_orateurs", "sujets_abordes") or (
                c.outil == "resume_seance" and not result.startswith("=== SÉANCE")):
            # affichés tels quels : Mistral invente des ex aequo ou des rangs dans les listes chiffrées, et une séance
            # entière à partir d'un message comme « Aucune séance le … »
            if on_token:
                on_token(result)
            return result, self.nodes
        if c.outil.startswith("recherche"):
            return self.write(ANSWER_SEARCH_PROMPT.format(result=result, question=question), history, on_token), \
                self.nodes
        if c.outil != "resume_seance":
            return self.write_stats(result, question, history, on_token), self.nodes
        # Résumé : une rédaction par séance (Mistral 7B mélange les séances d'une même journée) ; il ne rédige que le
        # plan, les décisions (élections, votes) sont recopiées telles quelles pour garder des chiffres exacts.
        parts = []
        for fiche in re.split(r"\n\n(?==== SÉANCE)", result):
            plan, _, decisions = fiche.partition(stats.DECISIONS)
            texte = self.write(ANSWER_RESUME_PROMPT.format(result=plan.strip(), question=question), history, on_token)
            decisions = f"\n\n{stats.DECISIONS}{decisions}"
            if on_token:
                on_token(decisions + "\n\n")
            parts.append(texte.strip() + decisions)
        return "\n\n".join(parts), self.nodes
