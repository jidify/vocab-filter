from pathlib import Path
import csv
import re
import statistics

from lemminflect import getLemma
from wordfreq import zipf_frequency


WORDLIST_PATH = Path("wordlist.txt")
BOOK_PATH = Path("The Humans - Stephen Karam.txt")
AOA_PATH = Path("kuperman-aoa.csv")

# Sources CEFR
CEFR_DIRECT_PATH = Path("cefr.csv")
CEFRJ_PATH = Path("cefrj.csv")
OCTANOVE_PATH = Path("octanove-c1c2.csv")


# --------------------------------------------------
# Réglage AoA
# --------------------------------------------------

MIN_AOA = 4.0
MAX_AOA = 5.0


# --------------------------------------------------
# Réglage Zipf
# --------------------------------------------------

MIN_ZIPF = 1.0
MAX_ZIPF = 5.0


# --------------------------------------------------
# Filtre CEFR optionnel
#
# None = aucun filtre
#
# Exemples :
#
# ALLOWED_CEFR = {"B1", "B2", "C1", "C2"}
# ALLOWED_CEFR = {"B2", "C1", "C2"}
# --------------------------------------------------

ALLOWED_CEFR = None


CEFR_ORDER = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}


# --------------------------------------------------
# Utilitaires
# --------------------------------------------------

def looks_inflected(word: str) -> bool:
    return (
        word.endswith("ing")
        or word.endswith("ed")
        or word.endswith("ies")
        or word.endswith("es")
        or word.endswith("s")
    )


def get_base_forms(word: str) -> list[str]:
    word = word.casefold()

    lemmas = set()

    for pos in ("NOUN", "VERB", "ADJ", "ADV"):
        results = getLemma(word, upos=pos)

        if results:
            for lemma in results:
                lemmas.add(lemma.casefold())

    return list(lemmas)


def expand_headword(headword: str) -> set[str]:
    """
    Certaines listes contiennent plusieurs variantes
    dans la même cellule :

        adviser/advisor
        analyze/analyse
        airplane/aeroplane

    On crée une entrée pour chaque variante.
    """

    headword = headword.strip().casefold()

    if not headword:
        return set()

    variants = {
        part.strip()
        for part in headword.split("/")
        if part.strip()
    }

    return variants


def add_cefr_entry(
    target: dict[str, set[str]],
    headword: str,
    level: str
):
    """
    Ajoute un niveau CEFR à toutes les variantes
    d'un headword.
    """

    level = level.strip().upper()

    if level not in CEFR_ORDER:
        return

    for word in expand_headword(headword):

        if word not in target:
            target[word] = set()

        target[word].add(level)


def highest_cefr(
    source: dict[str, set[str]],
    word: str
) -> str | None:
    """
    Si un mot apparaît à plusieurs niveaux / POS,
    retourne le niveau le plus élevé.
    """

    levels = source.get(
        word.casefold()
    )

    if not levels:
        return None

    return max(
        levels,
        key=lambda level: CEFR_ORDER[level]
    )


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
# Charger AoA
# --------------------------------------------------

aoa_scores = {}

with AOA_PATH.open(
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        word = (
            row["Word"]
            .strip()
            .casefold()
        )

        rating = (
            row["Rating.Mean"]
            .strip()
        )

        if not rating:
            continue

        try:
            aoa_scores[word] = float(rating)

        except ValueError:
            continue


# ==================================================
# CEFR SOURCE 1
#
# cefr.csv
# Priorité maximale
# ==================================================

cefr_direct = {}

with CEFR_DIRECT_PATH.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter=";"
    )

    for row in reader:

        headword = row["headword"]
        level = row["CEFR"]

        add_cefr_entry(
            cefr_direct,
            headword,
            level
        )


# ==================================================
# CEFR SOURCE 2
#
# CEFR-J Vocabulary Profile 1.5
# ==================================================

cefr_j = {}

with CEFRJ_PATH.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        headword = row["headword"]
        level = row["CEFR"]

        add_cefr_entry(
            cefr_j,
            headword,
            level
        )


# ==================================================
# CEFR SOURCE 3
#
# Octanove C1/C2
# ==================================================

cefr_octanove = {}

with OCTANOVE_PATH.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        headword = row["headword"]
        level = row["CEFR"]

        add_cefr_entry(
            cefr_octanove,
            headword,
            level
        )


# --------------------------------------------------
# Recherche CEFR avec priorité
#
# 1. direct
# 2. CEFR-J
# 3. Octanove
# --------------------------------------------------

def get_cefr(word: str) -> tuple[str | None, str]:
    key = word.casefold()

    # 1. CSV direct
    level = highest_cefr(
        cefr_direct,
        key
    )

    if level is not None:
        return level, "direct"

    # 2. CEFR-J
    level = highest_cefr(
        cefr_j,
        key
    )

    if level is not None:
        return level, "CEFR-J"

    # 3. Octanove C1/C2
    level = highest_cefr(
        cefr_octanove,
        key
    )

    if level is not None:
        return level, "Octanove"

    # Inconnu
    return None, "?"


# --------------------------------------------------
# Lire le livre
# --------------------------------------------------

book_text = BOOK_PATH.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# Extraire les mots
# --------------------------------------------------

words_in_book = re.findall(
    r"[A-Za-z]+(?:'[A-Za-z]+)?",
    book_text,
)


# --------------------------------------------------
# Filtrer
# --------------------------------------------------

vocabulary = []
seen_lemmas = set()


for word in words_in_book:

    key = word.casefold()


    # --------------------------------------------------
    # Le mot doit appartenir à wordlist.txt
    # --------------------------------------------------

    if key not in allowed_words:
        continue


    # --------------------------------------------------
    # AoA
    # --------------------------------------------------

    if key in aoa_scores:

        lemma = key
        aoa = aoa_scores[key]

    else:

        if not looks_inflected(key):
            continue

        candidates = get_base_forms(key)

        aoa_candidates = [
            (
                candidate,
                aoa_scores[candidate]
            )
            for candidate in candidates
            if candidate in aoa_scores
        ]

        if not aoa_candidates:
            continue

        lemma, aoa = min(
            aoa_candidates,
            key=lambda item: item[1]
        )


    # --------------------------------------------------
    # Filtre AoA
    # --------------------------------------------------

    if not (
        MIN_AOA <= aoa < MAX_AOA
    ):
        continue


    # --------------------------------------------------
    # Déduplication par lemme
    # --------------------------------------------------

    if lemma in seen_lemmas:
        continue


    # --------------------------------------------------
    # Zipf
    # --------------------------------------------------

    zipf = zipf_frequency(
        lemma,
        "en"
    )


    if not (
        MIN_ZIPF <= zipf < MAX_ZIPF
    ):
        continue


    # --------------------------------------------------
    # CEFR
    #
    # On essaie le lemme en premier.
    # Si absent, on essaie la forme exacte du livre.
    # --------------------------------------------------

    cefr, cefr_source = get_cefr(
        lemma
    )

    if cefr is None and lemma != key:

        cefr, cefr_source = get_cefr(
            key
        )


    # --------------------------------------------------
    # Filtre CEFR optionnel
    # --------------------------------------------------

    if ALLOWED_CEFR is not None:

        if cefr is None:
            continue

        if cefr not in ALLOWED_CEFR:
            continue


    vocabulary.append(
        (
            word,
            lemma,
            aoa,
            zipf,
            cefr,
            cefr_source
        )
    )

    seen_lemmas.add(lemma)


# --------------------------------------------------
# Trier par Zipf
# --------------------------------------------------

vocabulary.sort(
    key=lambda item: item[3]
)


# --------------------------------------------------
# Affichage
# --------------------------------------------------

print()

print("=== VOCABULARY ===")

print(
    f"AoA  : "
    f"{MIN_AOA:.1f} -> {MAX_AOA:.1f}"
)

print(
    f"Zipf : "
    f"{MIN_ZIPF:.1f} -> {MAX_ZIPF:.1f}"
)

print()


for (
    word,
    lemma,
    aoa,
    zipf,
    cefr,
    cefr_source
) in vocabulary:

    cefr_display = (
        cefr
        if cefr is not None
        else "?"
    )

    print(
        f"{word:<22} "
        f"lemma={lemma:<18} "
        f"AoA={aoa:>5.2f} "
        f"Zipf={zipf:>4.2f} "
        f"CEFR={cefr_display:<2} "
        f"source={cefr_source}"
    )


# --------------------------------------------------
# Statistiques
# --------------------------------------------------

print()
print("=" * 105)

print(
    f"Nombre total de mots : "
    f"{len(vocabulary)}"
)


if vocabulary:

    aoa_values = [
        item[2]
        for item in vocabulary
    ]

    zipf_values = [
        item[3]
        for item in vocabulary
    ]


    print()
    print("=== STATISTIQUES AoA ===")

    print(
        f"Minimum : "
        f"{min(aoa_values):.2f}"
    )

    print(
        f"Maximum : "
        f"{max(aoa_values):.2f}"
    )

    print(
        f"Moyenne : "
        f"{statistics.mean(aoa_values):.2f}"
    )

    print(
        f"Médiane : "
        f"{statistics.median(aoa_values):.2f}"
    )


    print()
    print("=== STATISTIQUES ZIPF ===")

    print(
        f"Minimum : "
        f"{min(zipf_values):.2f}"
    )

    print(
        f"Maximum : "
        f"{max(zipf_values):.2f}"
    )

    print(
        f"Moyenne : "
        f"{statistics.mean(zipf_values):.2f}"
    )

    print(
        f"Médiane : "
        f"{statistics.median(zipf_values):.2f}"
    )


    # --------------------------------------------------
    # Répartition CEFR
    # --------------------------------------------------

    print()
    print("=== RÉPARTITION CEFR ===")

    cefr_counts = {
        "A1": 0,
        "A2": 0,
        "B1": 0,
        "B2": 0,
        "C1": 0,
        "C2": 0,
        "?": 0,
    }

    source_counts = {
        "direct": 0,
        "CEFR-J": 0,
        "Octanove": 0,
        "?": 0,
    }


    for item in vocabulary:

        cefr = item[4]
        source = item[5]

        if cefr is None:
            cefr_counts["?"] += 1
        else:
            cefr_counts[cefr] += 1

        if source in source_counts:
            source_counts[source] += 1
        else:
            source_counts["?"] += 1


    for level in (
        "A1",
        "A2",
        "B1",
        "B2",
        "C1",
        "C2",
        "?"
    ):

        print(
            f"{level:<3} : "
            f"{cefr_counts[level]}"
        )


    print()
    print("=== SOURCES CEFR ===")

    print(
        f"Direct   : "
        f"{source_counts['direct']}"
    )

    print(
        f"CEFR-J   : "
        f"{source_counts['CEFR-J']}"
    )

    print(
        f"Octanove : "
        f"{source_counts['Octanove']}"
    )

    print(
        f"Absent   : "
        f"{source_counts['?']}"
    )