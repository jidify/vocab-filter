"""POC — fusionne les deux CSV d'analyse lexicale produits en aval du
pipeline mots/MWE (translate_word_context.py, translate_mwe_context.py) en un
seul CSV consultable et filtrable sur l'ensemble de leurs champs.

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production. Contrairement à ses deux sources, ce script
n'appelle aucun LLM — pure transformation CSV déterministe (stdlib
uniquement : argparse, csv, pathlib).

Entrées (schémas produits respectivement par translate_word_context.py et
translate_mwe_context.py, voir CSV_HEADER de ces deux scripts) :
  - mots : lemme, false_friend, sense, definition_en, translations, example
  - mwe  : extracted_form, lexicalized_form, mwe_type, compositionality,
    conventionality, difficulty_for_non_native, sense, definition_en,
    translations, example

Sortie : un CSV, une ligne par ligne d'entrée (aucune ligne ajoutée ni
supprimée), colonnes (ordre choisi par l'utilisateur) :
    type, lemme, extracted_form, lexicalized_form, mwe_type,
    sense, definition_en, translations,
    false_friend, compositionality, conventionality, difficulty_for_non_native,
    example

Règle de fusion des colonnes (décidée avec l'utilisateur) :
  - TOUTES les colonnes des deux entrées sont conservées, cellule vide quand
    le champ ne s'applique pas à la source de la ligne (ex. `extracted_form`
    vide sur une ligne `type=word`).
  - `lemme` et `extracted_form` restent des colonnes DISTINCTES malgré leur
    rôle voisin (identifiant de l'entrée) : noms différents dans les CSV
    source, on ne les fusionne pas.
  - Les 4 colonnes qui portent le MÊME nom et la MÊME sémantique dans les
    deux entrées (sense, definition_en, translations, example) fusionnent en
    une seule colonne chacune, toujours remplie quelle que soit l'origine.
  - Une colonne `type` (word/mwe) indique le fichier d'origine de la ligne.

Ordre des lignes : concaténation pure, jamais de tri — toutes les lignes du
CSV mots dans leur ordre d'origine, puis toutes les lignes du CSV mwe dans le
leur. Chaque ligne de sortie reste ainsi traçable vers sa ligne source.

Garde-fou schéma : la lecture de chaque CSV d'entrée vérifie la présence de
toutes ses colonnes attendues et échoue bruyamment (avec le nom du fichier et
de la colonne manquante) si l'une manque, plutôt que de produire
silencieusement un CSV fusionné tronqué — utile si l'un des deux scripts
amont fait évoluer son schéma de sortie.

Usage :
    uv run python POC/pipeline/stages/merge_word_and_mwe_analysis.py
    uv run python POC/pipeline/stages/merge_word_and_mwe_analysis.py \
        --word-in <word_analysis.csv> --mwe-in <mwe_analysis.csv> --out <fusionné.csv>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# POC/pipeline/stages/merge_word_and_mwe_analysis.py -> POC/ est le parent(2).
ROOT = Path(__file__).resolve().parents[2]

# Pointent vers les VRAIES sorties de translate_word_context.py /
# translate_mwe_context.py (et non les fixtures de tests/, comme c'était le
# cas avant — piège documenté dans le plan "Pipeline POC autonome" : lancer
# les scripts sans argument fusionnait silencieusement 25 lignes de test).
DEFAULT_WORD_IN = Path(__file__).parent / "word_analysis.csv"
DEFAULT_MWE_IN = Path(__file__).parent / "mwe_analysis.csv"
DEFAULT_OUT_PATH = Path(__file__).parent / "word_and_mwe_analysis.csv"

TYPE_COLUMN = "type"

# Copies exactes des CSV_HEADER de translate_word_context.py et
# translate_mwe_context.py — servent à la fois de contrat de validation en
# entrée (voir read_rows) et de source de vérité pour CSV_HEADER ci-dessous.
WORD_COLUMNS = ["lemme", "false_friend", "sense", "definition_en", "translations", "example"]
MWE_COLUMNS = [
    "extracted_form", "lexicalized_form", "mwe_type", "compositionality",
    "conventionality", "difficulty_for_non_native", "sense", "definition_en",
    "translations", "example",
]

# Colonnes homonymes des deux entrées (mêmes nom ET sémantique) : fusionnées
# en une seule colonne de sortie plutôt que dédoublées.
_SHARED_COLUMNS = ["sense", "definition_en", "translations", "example"]

# Schéma de sortie — ordre choisi par l'utilisateur : les identifiants
# d'entrée puis le typage/sens en tête (colonnes les plus lues en premier),
# le reste des colonnes propres à chaque source ensuite.
CSV_HEADER = [
    "type", "lemme", "extracted_form", "lexicalized_form", "mwe_type",
    "sense", "definition_en", "translations",
    "false_friend", "compositionality", "conventionality", "difficulty_for_non_native",
    "example",
]


# --------------------------------------------------------------------------
# Lecture des CSV d'entrée (avec garde-fou de schéma)
# --------------------------------------------------------------------------

def read_rows(path: Path, expected_columns: list[str], row_type: str) -> list[dict[str, str]]:
    """Lit un CSV d'analyse (mots ou mwe) et renvoie ses lignes, chacune
    limitée aux colonnes attendues et taguée `type=row_type`. Échoue
    bruyamment si une colonne attendue manque de l'en-tête — voir le
    docstring du module."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in expected_columns if c not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} : colonne(s) attendue(s) absente(s) de l'en-tête : {missing} "
                f"(en-tête lu : {fieldnames})"
            )
        rows = []
        for row in reader:
            merged_row = {TYPE_COLUMN: row_type}
            for column in expected_columns:
                merged_row[column] = row.get(column, "")
            rows.append(merged_row)
    return rows


# --------------------------------------------------------------------------
# Écriture du CSV fusionné
# --------------------------------------------------------------------------

def write_merged(rows: list[dict[str, str]], out_path: Path) -> None:
    """Écrit `rows` en une seule passe (mode "w", pas d'append) : le piège
    BOM-en-mode-append documenté dans translate_word_context.py::
    append_analyses_csv ne s'applique pas ici, "utf-8-sig" est donc correct
    et voulu pour que le fichier s'ouvre proprement dans Excel."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, restval="", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-in", dest="word_in_path", default=str(DEFAULT_WORD_IN),
                         help="Chemin du CSV d'analyse des mots (défaut : "
                              "word_analysis_test-batch.csv)")
    parser.add_argument("--mwe-in", dest="mwe_in_path", default=str(DEFAULT_MWE_IN),
                         help="Chemin du CSV d'analyse des MWE (défaut : "
                              "mwe_analysis_test-batch.csv)")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV fusionné en sortie (défaut : "
                              "word_and_mwe_analysis.csv)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    word_in_path = Path(args.word_in_path)
    mwe_in_path = Path(args.mwe_in_path)
    out_path = Path(args.out_path)

    if not word_in_path.exists():
        print(f"CSV d'entrée (mots) introuvable : {word_in_path}")
        return 1
    if not mwe_in_path.exists():
        print(f"CSV d'entrée (mwe) introuvable : {mwe_in_path}")
        return 1

    print(f"Entrée mots : {word_in_path}")
    print(f"Entrée mwe  : {mwe_in_path}")

    word_rows = read_rows(word_in_path, WORD_COLUMNS, "word")
    mwe_rows = read_rows(mwe_in_path, MWE_COLUMNS, "mwe")

    write_merged(word_rows + mwe_rows, out_path)

    print()
    print("=== Récapitulatif ===")
    print(f"Lignes word : {len(word_rows)}")
    print(f"Lignes mwe  : {len(mwe_rows)}")
    print(f"Total       : {len(word_rows) + len(mwe_rows)}")
    print()
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
