# rag-an

Chatbot en ligne de commande sur 48 comptes rendus de séance de l'Assemblée nationale (juillet 2024 – septembre 2025).
Tout tourne en local : PostgreSQL + pgvector, MongoDB, Ollama (Mistral 7B).

## Prérequis

Docker (sans `sudo`), [uv](https://docs.astral.sh/uv/getting-started/installation/), environ 20 Go de disque.

## Installation

Dans le dossier `rag-an/` :

```bash
docker compose --profile ollama up -d            # sans --profile ollama si Ollama est déjà installé
docker compose exec ollama ollama pull mistral   # ou : ollama pull mistral
uv sync
```

## Utilisation

```bash
uv run rag-ingest docs/   # une fois (≈ 1 h 15 sur CPU)
uv run rag-chat           # /sources, /reset, /quitter
```

Un GPU NVIDIA est utilisé automatiquement pour l'indexation ; pour Mistral, décommenter `deploy` dans
`docker-compose.yml`. Réglages facultatifs : `cp .env.example .env`.
