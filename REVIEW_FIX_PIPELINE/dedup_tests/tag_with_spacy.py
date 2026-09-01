"""Fait tourner spaCy sur un fichier produit par extract_context.py
(une ligne = un extrait de contexte_en) et écrit, pour chaque phrase
détectée (segmentation spaCy), un tableau token/lemme/POS/tag
fin/dépendance/tête + les entités nommées reconnues.

Modèle chargé SANS rien désactiver (`en_core_web_sm`, comme
pipeline/analyze.py::get_nlp() en production) et SANS le patch de
tokenizer "tiret après ponctuation fermante" (voir
TODO/tokenizer_dash_after_punctuation.md — connu, mesuré, mais pas
intégré en production) : ce script montre fidèlement ce que le
pipeline produit réellement, pas une version corrigée.

--filter LEMMA n'affiche que les lignes de tableau dont le LEMMA
correspond exactement (insensible à la casse) — une phrase sans aucun
token de ce lemme est omise entièrement. La sortie filtrée va dans un
fichier séparé (suffixe `_filter_<lemma>`), le fichier `_spacy.txt`
complet n'est jamais écrasé par un filtre.

Usage :
    uv run python REVIEW_FIX_PIPELINE/dedup_tests/tag_with_spacy.py extract_be going to.txt
    uv run python REVIEW_FIX_PIPELINE/dedup_tests/tag_with_spacy.py extract_be going to.txt --filter go
"""

from __future__ import annotations

import argparse
from pathlib import Path

import spacy

ROOT = Path("C:/DOCS/_perso/vocab-filter")
EXTRACTIONS_DIR = ROOT / "REVIEW_FIX_PIPELINE" / "dedup_tests" / "extractions"

SPACY_MODEL = "en_core_web_sm"

COLS = f"{'TOKEN':<20}{'LEMMA':<20}{'POS':<7}{'TAG':<7}{'DEP':<14}{'HEAD'}"

FORBIDDEN_FILENAME_CHARS = '\\/:*?"<>|'


def sanitize_filename(text: str) -> str:
    return "".join("_" if c in FORBIDDEN_FILENAME_CHARS else c for c in text)


def format_sentence(sent, filter_lemma: str | None = None) -> str | None:
    tokens = list(sent)
    if filter_lemma is not None:
        tokens = [t for t in tokens if t.lemma_.casefold() == filter_lemma.casefold()]
        if not tokens:
            return None
    lines = [COLS]
    for tok in tokens:
        lines.append(
            f"{tok.text:<20}{tok.lemma_:<20}{tok.pos_:<7}{tok.tag_:<7}{tok.dep_:<14}{tok.head.text}"
        )
    ents = [f"{e.text} ({e.label_})" for e in sent.ents]
    if ents:
        lines.append("ENTITÉS : " + ", ".join(ents))
    return "\n".join(lines)


def resolve_input(raw: str) -> Path:
    """Cherche le fichier tel quel, puis dans EXTRACTIONS_DIR (là où
    extract_context.py écrit par défaut) — pour pouvoir passer juste
    `extract_<canonical_form>.txt` sans chemin complet."""
    p = Path(raw)
    if p.exists():
        return p
    candidate = EXTRACTIONS_DIR / raw
    if candidate.exists():
        return candidate
    return p  # laisse l'erreur d'ouverture standard s'afficher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="fichier extract_<canonical_form>.txt à taguer")
    parser.add_argument("--filter", dest="filter_lemma", default=None,
                        help="n'affiche que les tokens dont le LEMMA correspond (insensible à la casse)")
    args = parser.parse_args()

    in_path = resolve_input(args.input)
    lines = [l.strip() for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(lines)} ligne(s) lue(s) depuis {in_path}")

    nlp = spacy.load(SPACY_MODEL)

    blocks = []
    for i, (line, doc) in enumerate(zip(lines, nlp.pipe(lines)), start=1):
        sent_blocks = [b for sent in doc.sents if (b := format_sentence(sent, args.filter_lemma)) is not None]
        if args.filter_lemma is not None and not sent_blocks:
            continue
        blocks.append(f"=== Ligne {i} ===\n{line}\n\n" + "\n\n".join(sent_blocks))

    if args.filter_lemma is not None:
        out_path = in_path.with_name(
            in_path.stem + "_spacy_filter_" + sanitize_filename(args.filter_lemma) + in_path.suffix
        )
        print(f"{len(blocks)} ligne(s) contenant le lemme {args.filter_lemma!r} sur {len(lines)}")
    else:
        out_path = in_path.with_name(in_path.stem + "_spacy" + in_path.suffix)

    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
