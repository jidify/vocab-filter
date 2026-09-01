"""POC : extraire des lemmes anglais filtres et leurs phrases d'usage.

Le script ne traite que les mots simples. Il ne depend d'aucun artefact du
pipeline principal et n'appelle aucun LLM.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import spacy
from wordfreq import zipf_frequency


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "books" / "The Humans - Stephen Karam.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "vocab_words.csv"
PREVALENCE_PATH = ROOT / "DATASETS" / "word-prevalence.txt"
CEFR_PATH = ROOT / "DATASETS" / "cefrj.csv"

SPACY_MODEL = "en_core_web_sm"
MIN_PKNOWN = 0.90
EXCLUDED_CEFR = {"A1", "A2"}
ZIPF_RESCUE_THRESHOLD = 4.5
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# Ordre de preference repris du test ayant produit rapport_filtrage.md.
UPOS_TO_CEFR_POS = {
    "NOUN": ["noun"],
    "PROPN": ["noun"],
    "VERB": ["verb"],
    "ADJ": ["adjective"],
    "ADV": ["adverb"],
}


@dataclass
class LemmaOccurrences:
    surfaces: set[str] = field(default_factory=set)
    upos: set[str] = field(default_factory=set)
    sentences: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrait les lemmes filtres et toutes leurs phrases d'usage."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"fichier texte anglais (defaut : {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV produit (defaut : {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def load_prevalence(path: Path) -> dict[str, float]:
    """Charge word-prevalence.txt : word,Pknown,Nobs,Prevalence,FreqZipfUS."""
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 2:
                continue
            try:
                scores[parts[0].strip().casefold()] = float(parts[1])
            except ValueError:
                continue
    return scores


def load_cefr(path: Path) -> dict[str, dict[str, set[str]]]:
    """Charge cefrj.csv sous la forme lemme -> POS -> niveaux CEFR."""
    by_word: dict[str, dict[str, set[str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pos_label = row["pos"].strip().casefold()
            level = row["CEFR"].strip().upper()
            if not level:
                continue
            for variant in row["headword"].split("/"):
                word = variant.strip().casefold()
                if word:
                    by_word.setdefault(word, {}).setdefault(pos_label, set()).add(level)
    return by_word


def cefr_levels_for(
    lemma: str,
    observed_upos: set[str],
    cefr_by_word: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Joint d'abord les POS observes, puis replie sur tous les POS connus."""
    pos_data = cefr_by_word.get(lemma)
    if not pos_data:
        return set()

    matched: set[str] = set()
    for upos in observed_upos:
        for candidate in UPOS_TO_CEFR_POS.get(upos, []):
            if candidate in pos_data:
                matched.update(pos_data[candidate])
                break

    if matched:
        return matched

    return set().union(*pos_data.values())


def lemma_passes_filter(
    lemma: str,
    observed_upos: set[str],
    pknown_by_lemma: dict[str, float],
    cefr_by_word: dict[str, dict[str, set[str]]],
    zipf_lookup: Callable[[str, str], float] = zipf_frequency,
) -> bool:
    """Applique Pknown, puis CEFR, puis le repechage Zipf du rapport."""
    pknown = pknown_by_lemma.get(lemma)
    if pknown is None or pknown <= MIN_PKNOWN:
        return False

    levels = cefr_levels_for(lemma, observed_upos, cefr_by_word)
    basic_only = bool(levels) and levels.issubset(EXCLUDED_CEFR)
    if basic_only and zipf_lookup(lemma, "en") >= ZIPF_RESCUE_THRESHOLD:
        return False

    return True


def extract_occurrences(text: str) -> dict[str, LemmaOccurrences]:
    """Lemmatise le texte et associe chaque lemme a ses phrases spaCy."""
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"Modele spaCy {SPACY_MODEL!r} introuvable ; installez les "
            "dependances du projet avec `uv sync`."
        ) from exc
    doc = nlp(text)
    occurrences: dict[str, LemmaOccurrences] = defaultdict(LemmaOccurrences)

    for sentence in doc.sents:
        sentence_text = " ".join(sentence.text.split())
        if not sentence_text:
            continue

        # Un lemme present plusieurs fois dans la meme phrase ne doit compter
        # cette phrase qu'une fois, tout en conservant toutes ses surfaces.
        lemmas_in_sentence: set[str] = set()
        for token in sentence:
            if (
                not token.is_alpha
                or token.is_stop
                or token.pos_ not in CONTENT_POS
            ):
                continue

            lemma = (token.lemma_ or token.text).casefold().strip()
            if len(lemma) < 3:
                continue

            entry = occurrences[lemma]
            entry.surfaces.add(token.text)
            entry.upos.add(token.pos_)
            lemmas_in_sentence.add(lemma)

        for lemma in lemmas_in_sentence:
            occurrences[lemma].sentences.append(sentence_text)

    return occurrences


def build_rows(
    occurrences: dict[str, LemmaOccurrences],
    pknown_by_lemma: dict[str, float],
    cefr_by_word: dict[str, dict[str, set[str]]],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for lemma in sorted(occurrences):
        entry = occurrences[lemma]
        if not lemma_passes_filter(
            lemma, entry.upos, pknown_by_lemma, cefr_by_word
        ):
            continue
        rows.append(
            {
                "canonical_form": lemma,
                "surface_forms": "/".join(
                    sorted(entry.surfaces, key=lambda value: (value.casefold(), value))
                ),
                "contexte_en": " || ".join(entry.sentences),
                "nombre_phrases": len(entry.sentences),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["canonical_form", "surface_forms", "contexte_en", "nombre_phrases"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} introuvable : {path}")


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    require_file(input_path, "Fichier d'entree")
    require_file(PREVALENCE_PATH, "Ressource Word Prevalence")
    require_file(CEFR_PATH, "Ressource CEFR")

    text = input_path.read_text(encoding="utf-8", errors="replace")
    occurrences = extract_occurrences(text)
    rows = build_rows(
        occurrences,
        load_prevalence(PREVALENCE_PATH),
        load_cefr(CEFR_PATH),
    )
    write_csv(rows, output_path)

    print(f"{len(occurrences)} lemmes candidats -> {len(rows)} lemmes conserves")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
