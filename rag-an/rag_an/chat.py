"""Chat en ligne de commande sur l'index déjà construit (lancer `rag-ingest` d'abord).

Usage :
    uv run rag-chat                     # agent : choisit statistiques, recherche lexicale ou sémantique
    uv run rag-chat --session demo      # reprend la conversation « demo » stockée dans MongoDB
Commandes : /reset (efface la mémoire), /sources (détail des sources de la dernière réponse), /quitter
"""
from __future__ import annotations

import argparse
import logging
import time
import uuid

import httpx
from llama_index.core.llms import ChatMessage

from .agent import RagAgent
from .config import get_settings
from .engine import format_sources
from .models import configure, preload_llm
from .stores import ConversationStore, get_source_document


def print_sources(sources: list[dict]) -> None:
    if not sources:
        return
    print("\n\n📚 Sources :")
    for src in sources:
        print(f"  [{src['n']}] {src['label']}" + (f"\n        {src['scores']}" if src["scores"] else ""))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="identifiant de conversation (reprise de l'historique)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    s = get_settings()
    session = args.session or uuid.uuid4().hex[:8]
    conversations = ConversationStore(s)
    history = conversations.load(session)

    t0 = time.perf_counter()
    print(f"⏳ Chargement de {s.embed_model} et de {s.llm_model} (Ollama) en parallèle…")
    llm_ready = preload_llm(s) if s.llm_preload else None
    configure(s)
    agent = RagAgent(s)
    t_embed = time.perf_counter() - t0
    if llm_ready is not None:
        llm_ready.join()
    print(f"✅ Prêt en {time.perf_counter() - t0:.1f}s (embedding {t_embed:.1f}s). "
          f"Session « {session} »" + (f" ({len(history)} messages repris)" if history else ""))
    print("💬 Pose tes questions (/reset, /sources, /quitter)")

    last_sources: list[dict] = []
    while True:
        try:
            question = input("\n✍️  Ta question : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        cmd = question.lower()
        if cmd in ("/quitter", "quitter", "exit", "quit"):
            break
        if cmd in ("/reset", "reset", "nouveau", "clear"):
            history = []
            conversations.reset(session)
            print("🔄 Mémoire effacée.")
            continue
        if cmd == "/sources":
            for src in last_sources:
                doc = get_source_document(src["doc_id"], s) or {}
                print(f"\n[{src['n']}] {src['label']}\n{doc.get('text', '')[:1500]}")
            continue

        t0 = time.perf_counter()
        first = None

        def on_token(tok: str):
            nonlocal first
            if first is None:
                first = time.perf_counter()
                print("\n🤖 ", end="", flush=True)
            print(tok, end="", flush=True)

        def on_tool(name: str, kwargs: dict):
            params = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v not in (None, ""))
            print(f"🔧 {name}({params})", flush=True)

        try:
            # agent sans état : on lui passe toute la conversation, il n'en garde que les derniers messages
            answer, nodes = agent.ask(question, history, on_token, on_tool)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            print(f"\n⚠️  Ollama ne répond pas ({type(e).__name__}). Vérifie qu'il tourne ({s.ollama_url}) "
                  f"ou augmente RAG_LLM_TIMEOUT (actuel : {s.llm_timeout}s).")
            continue

        last_sources = format_sources(nodes)
        print_sources(last_sources)
        total = time.perf_counter() - t0
        ttft = (first - t0) if first else total
        print(f"⏱️  1er mot après {ttft:.1f}s, réponse complète en {total:.1f}s")

        history += [ChatMessage(role="user", content=question), ChatMessage(role="assistant", content=answer)]
        conversations.append(session, "user", question)
        conversations.append(session, "assistant", answer, sources=last_sources)

    print("👋 À bientôt !")


if __name__ == "__main__":
    main()
