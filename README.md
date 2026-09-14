# ia-service

Service IA, résumés automatiques, classification et analyse des données politiques.

Django + Django REST Framework, géré avec [uv](https://docs.astral.sh/uv/).

**Socle de base uniquement** — aucun provider IA n'est câblé pour l'instant
(intégration Mistral prévue, à faire séparément).

## Setup

```bash
uv sync
cp .env.example .env    # ajuster JWT_PUBLIC_KEY et le reste au besoin
git config core.hooksPath .githooks   # active les hooks locaux (une fois)
uv run python manage.py migrate
uv run python manage.py runserver
```

`GET /api/health/` → `{"status": "ok"}`

## Authentification

ia-service ne fait que **vérifier** les JWT émis par `auth-service` (RS256) :
il détient uniquement `JWT_PUBLIC_KEY` (jamais la clé privée). Voir
`auth-service/README.md` pour la génération de la paire de clés.
