"""Extrait, depuis pipeline_out/vocab.csv, toutes les phrases de
`contexte_en` partageant un `canonical_form` donné (tous POS/sens
confondus) — pour relire rapidement le contexte réel d'un mot ou d'une
MWE sans rouvrir tout vocab.csv.

Une cellule `contexte_en` contient plusieurs occurrences séparées par
" || " (voir pipeline/sense_fr.py::format_occurrences_en, même
convention partout où une traduction est exposée dans pipeline_out/) :
chaque occurrence extraite ici devient une ligne du fichier de sortie.

Usage :
    uv run python REVIEW_FIX_PIPELINE/dedup_tests/extract_context.py "be going to"
    uv run python REVIEW_FIX_PIPELINE/dedup_tests/extract_context.py "New York" --input REVIEW_FIX_PIPELINE/vocab_deduped.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path("C:/DOCS/_perso/vocab-filter")
IN_PATH = ROOT / "pipeline_out" / "vocab.csv"
OUT_DIR = ROOT / "REVIEW_FIX_PIPELINE" / "dedup_tests" / "extractions"

CONTEXT_SEPARATOR = " || "

# Caractères interdits dans un nom de fichier Windows — remplacés par
# "_" pour le seul nom de fichier ; le canonical_form lui-même (utilisé
# pour la recherche) n'est jamais modifié.
FORBIDDEN_FILENAME_CHARS = '\\/:*?"<>|'


def sanitize_filename(text: str) -> str:
    return "".join("_" if c in FORBIDDEN_FILENAME_CHARS else c for c in text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("canonical_form", help="canonical_form à extraire (comparaison insensible à la casse)")
    parser.add_argument("--input", default=str(IN_PATH), help=f"CSV source (défaut : {IN_PATH})")
    args = parser.parse_args()

    in_path = Path(args.input)
    target = args.canonical_form.casefold().strip()

    with in_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    matches = [r for r in rows if r["canonical_form"].casefold().strip() == target]
    if not matches:
        print(f"Aucune ligne avec canonical_form={args.canonical_form!r} dans {in_path}")
        return

    phrases: list[str] = []
    for r in matches:
        for phrase in (r.get("contexte_en") or "").split(CONTEXT_SEPARATOR):
            phrase = phrase.strip()
            if phrase:
                phrases.append(phrase)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"extract_{sanitize_filename(args.canonical_form)}.txt"
    out_path.write_text("\n".join(phrases) + "\n", encoding="utf-8")

    print(f"{len(matches)} ligne(s) {in_path.name} (pos/sens distincts) pour {args.canonical_form!r}")
    print(f"{len(phrases)} phrase(s) extraite(s) de contexte_en")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
