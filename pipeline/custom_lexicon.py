"""Lexique piloté par les données : expressions et cas de tokenisation
ajoutés par un relecteur humain (pipeline/review_ui.py, workflow "aucune
cible n'existe encore") SANS éditer de code Python.

Avant ce module, ajouter une expression comme "smart ass" (absente de
idiomatch) ou un cas de tokenisation comme "e-mail" (que spaCy coupe en
3 tokens) exigeait d'éditer à la main pipeline/mwe.py::CUSTOM_IDIOMS et
pipeline/analyze.py::EMAIL_SPECIAL_CASES — voir le plan du 2026-08-27
"IHM de correction manuelle : plusieurs workflows, lexique piloté par les
données". data/custom_lexicon.jsonl (versionné, comme data/sense_fr.jsonl)
prend le relais : mwe.py et analyze.py fusionnent son contenu à leurs
listes en dur au chargement — celles-ci restent le socle (rien ne les
remplace), ce fichier ne fait qu'AJOUTER.

Deux types d'enregistrements, un par ligne :
    {"kind": "idiom", "lemma": "smart ass", "definition_en": "...",
     "added_at": "2026-08-27", "source": "review_ui"}
    {"kind": "tokenizer", "surfaces": ["e-mail", "e-mails", ...],
     "reason": "...", "added_at": "2026-08-27", "source": "review_ui"}

Lu par mwe.py (fusionné à CUSTOM_IDIOMS) et analyze.py (fusionné à
EMAIL_SPECIAL_CASES). Une entrée créée pendant une relecture profite donc
directement au PROCHAIN livre traité — pas besoin de rejouer S1-S5 pour
le livre courant, qui passe par data/manual_corrections.jsonl comme
d'habitude (voir sense_fr_commit.py).

Usage (bibliothèque, pas de CLI) :
    from pipeline import custom_lexicon
    custom_lexicon.load_idioms()               # -> format CUSTOM_IDIOMS
    custom_lexicon.load_tokenizer_surfaces()    # -> liste de chaînes
    custom_lexicon.add_idiom("smart ass", "A person who makes...")
    custom_lexicon.add_tokenizer_surfaces(["e-mail", "e-mails"], reason="...")
"""

from __future__ import annotations

import json
from datetime import date

from pipeline import config


def _load_entries() -> list[dict]:
    if not config.CUSTOM_LEXICON_PATH.exists():
        return []
    with config.CUSTOM_LEXICON_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_idioms() -> list[dict]:
    """Entrées `kind == "idiom"`, au format exact attendu par
    idiomatch.Idiomatcher.add_idioms() — mêmes clés que
    pipeline/mwe.py::CUSTOM_IDIOMS, pour une fusion directe."""
    return [
        {
            "etymology": None,
            "lemma": e["lemma"],
            "senses": [{"content": e.get("definition_en") or "", "examples": []}],
            "source": "custom_lexicon",
        }
        for e in _load_entries() if e.get("kind") == "idiom"
    ]


def load_tokenizer_surfaces() -> list[str]:
    """Surfaces `kind == "tokenizer"` à ajouter comme cas spéciaux du
    tokenizer spaCy — mêmes chaînes que pipeline/analyze.py::EMAIL_SPECIAL_CASES."""
    surfaces: list[str] = []
    for e in _load_entries():
        if e.get("kind") == "tokenizer":
            surfaces.extend(e.get("surfaces") or [])
    return surfaces


def _append(entry: dict) -> None:
    config.ensure_data_dir()
    with config.CUSTOM_LEXICON_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_idiom(lemma: str, definition_en: str, source: str = "review_ui") -> None:
    _append({
        "kind": "idiom", "lemma": lemma, "definition_en": definition_en,
        "added_at": date.today().isoformat(), "source": source,
    })


def add_tokenizer_surfaces(surfaces: list[str], reason: str = "", source: str = "review_ui") -> None:
    if not surfaces:
        return
    _append({
        "kind": "tokenizer", "surfaces": surfaces, "reason": reason,
        "added_at": date.today().isoformat(), "source": source,
    })
