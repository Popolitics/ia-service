"""Sources de l'Assemblée nationale -> Documents LlamaIndex à indexer.

- Comptes rendus XML (`CRSANR...xml`) ou JSON silver : découpés par chunking.py ; seuls les discours sont indexés.
- Déclarations politiques des groupes (`decl_<SIGLE>.txt`) : un Document par groupe, découpé à l'ingestion.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from llama_index.core import Document

from .chunking import chunk_seance
from .communs import hidden_metadata
from .silver import est_provisoire, load_seance

# Libellés AN des groupes, par sigle de fichier decl_<SIGLE>.txt
GROUPES = {
    "Dem": "Les Démocrates",
    "DR": "Droite Républicaine",
    "Ecolo": "Écologiste et Social",
    "EPR": "Ensemble pour la République",
    "GDR": "Gauche Démocrate et Républicaine",
    "Hor": "Horizons & Indépendants",
    "LFI": "La France insoumise - Nouveau Front Populaire",
    "LIOT": "Libertés, Indépendants, Outre-mer et Territoires",
    "PS": "Socialistes et apparentés",
    "RN": "Rassemblement National",
    "UDR": "Union des droites pour la République",
}


def make_document(text: str, metadata: dict, chunked: bool) -> Document:
    """chunked : déjà découpé (compte rendu), l'ingestion ne le redécoupe pas."""
    metadata = {k: v for k, v in metadata.items() if v is not None} | {"chunked": chunked}
    return Document(text=text, id_=metadata["doc_id"], metadata=metadata, **hidden_metadata(metadata))


def documents_from_silver(silver: dict, max_tokens: int, overlap_tokens: int,
                          count: Callable[[str], int] | None = None, mode: str = "taille",
                          embed: Callable | None = None) -> list[Document]:
    chunks = chunk_seance(silver, max_tokens=max_tokens, overlap_tokens=overlap_tokens, count=count, mode=mode,
                          embed=embed)
    return [make_document(c["text"], {**c["metadata"], "source": silver["seance"]["fichier"]}, chunked=True)
            for c in chunks]


def parse_declaration(path: Path) -> Document:
    sigle = path.stem.removeprefix("decl_")
    text = re.sub(r"\n\s*\n+", "\n\n", path.read_text(encoding="utf-8")).strip()
    return make_document(text, {
        "doc_id": f"declaration#{sigle}",
        "source": path.name,
        "type": "declaration_politique",
        "document": "Déclaration politique de groupe (XVIIe législature)",
        "groupe": f"{GROUPES.get(sigle, sigle)} ({sigle})",
    }, chunked=False)


def load_directories(directories: list[str | Path], max_tokens: int, overlap_tokens: int,
                     count: Callable[[str], int] | None = None, silver_out: str | Path | None = None,
                     mode: str = "taille", embed: Callable | None = None) -> tuple[list[Document], list[dict]]:
    """Documents à indexer + séances silver (pour les statistiques), lus dans un ou plusieurs dossiers.

    silver_out : si renseigné, y écrit le JSON silver de chaque compte rendu XML.
    mode, embed : découpage des discours longs (voir chunking.chunk_seance).
    """
    files = [p for d in map(Path, directories) for p in sorted(d.glob("*.xml")) + sorted(d.glob("*.json"))]
    docs: list[Document] = []
    seances: dict[str, dict] = {}
    for p in files:
        silver = load_seance(p)
        if est_provisoire(silver):
            print(f"⏭️  {p.name} : compte rendu provisoire, ignoré (la version définitive le remplacera)")
            continue
        uid = silver["seance"]["uid"]
        if uid in seances:  # même séance en XML et en JSON : on garde la première
            continue
        seances[uid] = silver
        if silver_out and p.suffix == ".xml":
            out = Path(silver_out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{p.stem}.json").write_text(json.dumps(silver, ensure_ascii=False, indent=1), encoding="utf-8")
        docs += documents_from_silver(silver, max_tokens, overlap_tokens, count, mode, embed)
    docs += [parse_declaration(p) for d in map(Path, directories) for p in sorted(d.glob("decl_*.txt"))]
    return docs, list(seances.values())
