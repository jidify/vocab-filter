import csv
import statistics
from pathlib import Path

import spacy
from spacy.util import compile_infix_regex
from wordfreq import zipf_frequency

ROOT = Path("C:/DOCS/_perso/vocab-filter")

WORDLIST_PATH = ROOT / "DATASETS" / "wordlist.txt"

# BOOK_PATH = ROOT / "The Humans - Stephen Karam.txt"
# BOOK_PATH = ROOT / "books" / "Dark Matter - Blake Crouch.txt"
BOOK_PATH = ROOT / "books" / "James Graham Plays_ 1 (Contempo - James Graham.txt"
# BOOK_PATH = ROOT / "books" / "James Graham Plays 2 - James Graham.txt"
# BOOK_PATH = ROOT / "books" / "The History Boys - Alan Bennett.txt"
# BOOK_PATH = ROOT / "books" / "The Silent Patient - Alex Michaelides.txt"

AOA_PATH = ROOT / "DATASETS" / "kuperman-aoa.csv"
PREVALENCE_PATH = ROOT / "DATASETS" / "word-prevalence.txt"
CEFR_PATH = ROOT / "DATASETS" / "cefrj.csv"
OUT_PATH = ROOT / "REVIEW_FIX_PIPELINE" / "filter_tests" / "filter_book_vocab.csv"

# Même modèle que pipeline/analyze.py, pour cohérence avec le vrai pipeline.
SPACY_MODEL = "en_core_web_sm"


# --------------------------------------------------
# Réglage de la tranche AoA
#
# Exemple :
# 5.0 <= AoA < 6.0
# --------------------------------------------------

MIN_AOA = 0.0
MAX_AOA = 20.0


# --------------------------------------------------
# Réglage de la tranche de fréquence Zipf
#
# Exemple :
# 3.0 <= Zipf < 4.0
#
# Plus Zipf est faible, plus le mot est rare.
# --------------------------------------------------

MIN_ZIPF = 0.0
MAX_ZIPF = 200.0


# --------------------------------------------------
# Réglage de la tranche Pknown (Word Prevalence)
#
# Exemple :
# 0.95 <= Pknown < 1.00
#
# Pknown = part des natifs testés qui connaissent le mot.
# Un mot absent de word-prevalence.txt est écarté (pas de
# repli "conservé par défaut" ici, contrairement au pipeline).
# --------------------------------------------------

MIN_PKNOWN = 0.90
MAX_PKNOWN = 1.01


# --------------------------------------------------
# Réglage du filtre CEFR
#
# Un mot est écarté si TOUS ses niveaux CEFR connus (cefrj.csv), POUR
# LE POS RÉELLEMENT TAGUÉ PAR SPACY SUR CETTE OCCURRENCE, sont dans
# EXCLUDED_CEFR (mot trop basique) - sauf repêchage ci-dessous. Si le
# POS spaCy ne matche aucune entrée cefrj.csv pour ce mot (POS absent
# du référentiel), repli sur l'union de tous les POS connus. Un mot
# absent de cefrj.csv (niveau inconnu) est toujours conservé.
# --------------------------------------------------

EXCLUDED_CEFR = {"A1", "A2"}

# Un mot A1/A2 exclusif est repêché si sa fréquence Zipf est en
# dessous de ce seuil (mot basique mais rare -> probablement une
# vraie lacune L2, pas un mot trivial).
ZIPF_RESCUE_THRESHOLD = 4.5


# --------------------------------------------------
# UPOS spaCy -> catégories POS de cefrj.csv (en toutes lettres),
# par ordre de préférence (ex. un AUX est essayé comme be-verb/
# have-verb/do-verb/modal avant le générique "verb").
# --------------------------------------------------

UPOS_TO_CEFR_POS = {
    "NOUN": ["noun"],
    "PROPN": ["noun"],
    "VERB": ["verb"],
    "AUX": ["be-verb", "have-verb", "do-verb", "modal auxiliary", "verb"],
    "ADJ": ["adjective"],
    "ADV": ["adverb"],
    "DET": ["determiner"],
    "PRON": ["pronoun"],
    "ADP": ["preposition"],
    "CCONJ": ["conjunction"],
    "SCONJ": ["conjunction"],
    "NUM": ["number"],
    "INTJ": ["interjection"],
    "PART": ["infinitive-to"],
}


# --------------------------------------------------
# Charger wordlist.txt
# --------------------------------------------------

allowed_words = {
    line.strip().casefold()
    for line in WORDLIST_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
}


# --------------------------------------------------
# Charger les scores AoA
# --------------------------------------------------

aoa_scores = {}

with AOA_PATH.open(
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        word = row["Word"].strip().casefold()
        rating = row["Rating.Mean"].strip()

        if not rating:
            continue

        try:
            aoa_scores[word] = float(rating)
        except ValueError:
            continue


# --------------------------------------------------
# Charger les scores Pknown (Word Prevalence)
#
# word-prevalence.txt : word,Pknown,Nobs,Prevalence,FreqZipfUS
# sans en-tête.
# --------------------------------------------------

pknown_scores = {}

with PREVALENCE_PATH.open(
    encoding="utf-8",
    errors="replace",
) as f:

    for line in f:
        parts = line.rstrip("\n").split(",")

        if len(parts) < 5:
            continue

        word = parts[0].strip().casefold()

        try:
            pknown = float(parts[1])
        except ValueError:
            continue

        pknown_scores[word] = pknown


# --------------------------------------------------
# Charger les niveaux CEFR (cefrj.csv), par (mot, pos)
#
# headword,pos,CEFR,... - headword peut porter plusieurs variantes
# séparées par "/" (ex. "a.m./A.M./am/AM") : chacune indexée à part.
# --------------------------------------------------

cefr_by_word: dict[str, dict[str, set[str]]] = {}

with CEFR_PATH.open(
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        pos_label = row["pos"].strip().casefold()
        level = row["CEFR"].strip().upper()

        for variant in row["headword"].split("/"):
            headword = variant.strip().casefold()

            if not headword:
                continue

            cefr_by_word.setdefault(headword, {}).setdefault(pos_label, set()).add(level)


# --------------------------------------------------
# Fix tokenizer "tiret collé après ponctuation fermante" (voir
# TODO/tokenizer_dash_after_punctuation.md) : en_core_web_sm ne scinde
# un tiret cadratin/demi-cadratin qu'ENTRE deux caractères alphabétiques,
# jamais quand il est collé sans espace à une ponctuation fermante
# précédente ('around?—we' reste un seul token-poubelle). Copié depuis
# fix_pipeline/detection_benchmark/tokenizer_boundary_fix.py
# (patch_dash_after_punctuation) - pas encore intégré à pipeline/analyze.py
# en production (risques listés dans le TODO, propres à la chaîne S1->S6
# verrouillée), mais sans risque équivalent ici : ce script est un test
# jetable, pas un maillon d'une chaîne figée par inventory.py.
# --------------------------------------------------

DASH_AFTER_CLOSING_PUNCT = r"(?<=[?!.,;:'\")\]’”])(?:-|–|—|--|---)(?=\w)"
DASH_BEFORE_OPENING_PUNCT = r"(?<=\w)(?:-|–|—|--|---)(?=[\[\(‘“])"


def patch_dash_after_punctuation(nlp) -> None:
    """Mute en place le Tokenizer de nlp : ajoute les deux motifs d'infixe
    ci-dessus aux motifs par défaut du modèle (jamais de retrait)."""
    new_infixes = list(nlp.Defaults.infixes) + [
        DASH_AFTER_CLOSING_PUNCT,
        DASH_BEFORE_OPENING_PUNCT,
    ]
    nlp.tokenizer.infix_finditer = compile_infix_regex(new_infixes).finditer


def cefr_levels_for(lemma: str, upos: str) -> set[str]:
    """Niveaux CEFR pour ce lemme, en essayant d'abord le(s) POS cefrj.csv
    correspondant au tag spaCy (UPOS_TO_CEFR_POS), puis repli sur l'union
    de tous les POS connus pour ce mot si aucun ne matche."""

    pos_data = cefr_by_word.get(lemma)

    if not pos_data:
        return set()

    for candidate in UPOS_TO_CEFR_POS.get(upos, []):
        if candidate in pos_data:
            return pos_data[candidate]

    union: set[str] = set()

    for levels in pos_data.values():
        union |= levels

    return union


# --------------------------------------------------
# Charger le modèle spaCy et parser le livre
#
# parser/ner désactivés : seuls le tagger et le lemmatizer sont
# nécessaires ici (pas de découpage en phrases, pas d'entités).
# --------------------------------------------------

nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
patch_dash_after_punctuation(nlp)

book_text = BOOK_PATH.read_text(
    encoding="utf-8"
)

doc = nlp(book_text)


# --------------------------------------------------
# Filtrer le vocabulaire
# --------------------------------------------------

vocabulary = []
seen = set()


for token in doc:

    # Seuls les tokens purement alphabétiques sont considérés (contrairement
    # à l'ancienne extraction par regex, une contraction comme "don't" est
    # ici scindée par spaCy en deux tokens ("do" + "n't") ; "n't" n'est pas
    # alphabétique et est donc écarté ici).
    if not token.is_alpha:
        continue

    key = token.text.casefold()

    # Le mot doit être présent dans wordlist.txt
    if key not in allowed_words:
        continue

    upos = token.pos_
    lemma = token.lemma_.casefold()


    # --------------------------------------------------
    # Récupérer l'AoA : forme exacte d'abord, lemme spaCy sinon
    # (même logique que pipeline/lexicon.py::aoa_for_form).
    # --------------------------------------------------

    if key in aoa_scores:
        aoa = aoa_scores[key]
    elif lemma in aoa_scores:
        aoa = aoa_scores[lemma]
    else:
        continue


    # --------------------------------------------------
    # Filtre AoA
    # --------------------------------------------------

    if not (MIN_AOA <= aoa < MAX_AOA):
        continue


    # --------------------------------------------------
    # Éviter plusieurs occurrences du même (lemme, POS)
    # --------------------------------------------------

    dedup_key = (lemma, upos)

    if dedup_key in seen:
        continue


    # --------------------------------------------------
    # Calculer la fréquence Zipf
    # --------------------------------------------------

    zipf = zipf_frequency(
        lemma,
        "en"
    )


    # --------------------------------------------------
    # Filtre Zipf
    # --------------------------------------------------

    if not (MIN_ZIPF <= zipf < MAX_ZIPF):
        continue


    # --------------------------------------------------
    # Récupérer Pknown (même lemme que AoA/Zipf ci-dessus)
    # --------------------------------------------------

    pknown = pknown_scores.get(lemma)

    if pknown is None:
        continue


    # --------------------------------------------------
    # Filtre Pknown
    # --------------------------------------------------

    # Borne basse stricte, alignee sur rapport_filtrage.md ("Pknown > 0.90"),
    # contrairement aux autres tranches (AoA/Zipf) de ce script qui restent
    # inclusives en borne basse.
    if not (pknown > MIN_PKNOWN and pknown < MAX_PKNOWN):
        continue


    # --------------------------------------------------
    # Filtre CEFR (avec repêchage Zipf), jointure par POS spaCy
    # --------------------------------------------------

    cefr_levels = cefr_levels_for(lemma, upos)
    is_basic_only = bool(cefr_levels) and cefr_levels.issubset(EXCLUDED_CEFR)

    if is_basic_only and not (zipf < ZIPF_RESCUE_THRESHOLD):
        continue

    cefr_label = "/".join(sorted(cefr_levels)) if cefr_levels else ""


    vocabulary.append(
        (
            token.text,
            lemma,
            upos,
            aoa,
            zipf,
            pknown,
            cefr_label,
        )
    )

    seen.add(dedup_key)


# --------------------------------------------------
# Trier par fréquence
#
# Les mots les plus rares apparaissent en premier.
# --------------------------------------------------

vocabulary.sort(
    key=lambda item: item[4]
)


# --------------------------------------------------
# Écriture CSV
# --------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(
        ["word", "lemma", "pos", "aoa", "zipf", "pknown", "cefr"]
    )

    for word, lemma, upos, aoa, zipf, pknown, cefr_label in vocabulary:
        writer.writerow(
            [word, lemma, upos, f"{aoa:.2f}", f"{zipf:.2f}", f"{pknown:.2f}", cefr_label]
        )


# --------------------------------------------------
# Affichage
# --------------------------------------------------

print()

print(
    f"=== VOCABULARY ==="
)

print(
    f"AoA    : {MIN_AOA:.1f} -> {MAX_AOA:.1f}"
)

print(
    f"Zipf   : {MIN_ZIPF:.1f} -> {MAX_ZIPF:.1f}"
)

print(
    f"Pknown : {MIN_PKNOWN:.2f} -> {MAX_PKNOWN:.2f}"
)

print(
    f"CEFR   : exclu si {sorted(EXCLUDED_CEFR)} exclusif (POS spaCy), "
    f"repêché si Zipf < {ZIPF_RESCUE_THRESHOLD:.1f}"
)

print()


for word, lemma, upos, aoa, zipf, pknown, cefr_label in vocabulary:

    print(
        f"{word:<25} "
        f"lemma={lemma:<20} "
        f"pos={upos:<6} "
        f"AoA={aoa:>5.2f} "
        f"Zipf={zipf:>4.2f} "
        f"Pknown={pknown:>4.2f} "
        f"CEFR={cefr_label or '?':<5}"
    )


# --------------------------------------------------
# Statistiques
# --------------------------------------------------

print()
print("=" * 80)

print(
    f"Nombre total de mots : "
    f"{len(vocabulary)}"
)


if vocabulary:

    aoa_values = [
        item[3]
        for item in vocabulary
    ]

    zipf_values = [
        item[4]
        for item in vocabulary
    ]

    pknown_values = [
        item[5]
        for item in vocabulary
    ]

    print()

    print("=== STATISTIQUES AoA ===")

    print(
        f"Minimum : {min(aoa_values):.2f}"
    )

    print(
        f"Maximum : {max(aoa_values):.2f}"
    )

    print(
        f"Moyenne : {statistics.mean(aoa_values):.2f}"
    )

    print(
        f"Médiane : {statistics.median(aoa_values):.2f}"
    )


    print()

    print("=== STATISTIQUES ZIPF ===")

    print(
        f"Minimum : {min(zipf_values):.2f}"
    )

    print(
        f"Maximum : {max(zipf_values):.2f}"
    )

    print(
        f"Moyenne : {statistics.mean(zipf_values):.2f}"
    )

    print(
        f"Médiane : {statistics.median(zipf_values):.2f}"
    )


    print()

    print("=== STATISTIQUES PKNOWN ===")

    print(
        f"Minimum : {min(pknown_values):.2f}"
    )

    print(
        f"Maximum : {max(pknown_values):.2f}"
    )

    print(
        f"Moyenne : {statistics.mean(pknown_values):.2f}"
    )

    print(
        f"Médiane : {statistics.median(pknown_values):.2f}"
    )


    print()

    print("=== RÉPARTITION CEFR ===")

    cefr_counts: dict[str, int] = {}

    for item in vocabulary:
        label = item[6] or "?"
        cefr_counts[label] = cefr_counts.get(label, 0) + 1

    for label in sorted(cefr_counts):
        print(
            f"{label:<8} : {cefr_counts[label]}"
        )

print()
print(f"-> {OUT_PATH}")
