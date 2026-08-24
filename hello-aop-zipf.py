from pathlib import Path
import csv
import re
import statistics

from lemminflect import getLemma
from wordfreq import zipf_frequency


WORDLIST_PATH = Path("wordlist.txt")
BOOK_PATH = Path("The Humans - Stephen Karam.txt")
AOA_PATH = Path("kuperman-aoa.csv")


# --------------------------------------------------
# Réglage de la tranche AoA
#
# Exemple :
# 5.0 <= AoA < 6.0
# --------------------------------------------------

MIN_AOA = 4.0
MAX_AOA = 5.0


# --------------------------------------------------
# Réglage de la tranche de fréquence Zipf
#
# Exemple :
# 3.0 <= Zipf < 4.0
#
# Plus Zipf est faible, plus le mot est rare.
# --------------------------------------------------

MIN_ZIPF = 1.0
MAX_ZIPF = 5.0


def looks_inflected(word: str) -> bool:
    """
    On ne tente une lemmatisation que pour des formes
    qui ressemblent à des flexions anglaises.
    """
    return (
        word.endswith("ing")
        or word.endswith("ed")
        or word.endswith("ies")
        or word.endswith("es")
        or word.endswith("s")
    )


def get_base_forms(word: str) -> list[str]:
    """
    Retourne les lemmes possibles proposés par lemminflect.
    """
    word = word.casefold()

    lemmas = set()

    for pos in ("NOUN", "VERB", "ADJ", "ADV"):
        results = getLemma(word, upos=pos)

        if results:
            for lemma in results:
                lemmas.add(lemma.casefold())

    return list(lemmas)


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
# Lire le livre
# --------------------------------------------------

book_text = BOOK_PATH.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# Extraire les mots du livre
# --------------------------------------------------

words_in_book = re.findall(
    r"[A-Za-z]+(?:'[A-Za-z]+)?",
    book_text,
)


# --------------------------------------------------
# Filtrer le vocabulaire
# --------------------------------------------------

vocabulary = []
seen_lemmas = set()


for word in words_in_book:

    key = word.casefold()

    # Le mot doit être présent dans wordlist.txt
    if key not in allowed_words:
        continue


    # --------------------------------------------------
    # Cas 1 :
    # le mot exact existe dans le dataset AoA
    # --------------------------------------------------

    if key in aoa_scores:

        lemma = key
        aoa = aoa_scores[key]

    else:

        # --------------------------------------------------
        # Cas 2 :
        # essayer une lemmatisation uniquement
        # si la forme semble fléchie
        # --------------------------------------------------

        if not looks_inflected(key):
            continue

        candidates = get_base_forms(key)

        aoa_candidates = [
            (candidate, aoa_scores[candidate])
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

    if not (MIN_AOA <= aoa < MAX_AOA):
        continue


    # --------------------------------------------------
    # Éviter plusieurs formes du même lemme
    # --------------------------------------------------

    if lemma in seen_lemmas:
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


    vocabulary.append(
        (
            word,
            lemma,
            aoa,
            zipf
        )
    )

    seen_lemmas.add(lemma)


# --------------------------------------------------
# Trier par fréquence
#
# Les mots les plus rares apparaissent en premier.
# --------------------------------------------------

vocabulary.sort(
    key=lambda item: item[3]
)


# --------------------------------------------------
# Affichage
# --------------------------------------------------

print()

print(
    f"=== VOCABULARY ==="
)

print(
    f"AoA  : {MIN_AOA:.1f} -> {MAX_AOA:.1f}"
)

print(
    f"Zipf : {MIN_ZIPF:.1f} -> {MAX_ZIPF:.1f}"
)

print()


for word, lemma, aoa, zipf in vocabulary:

    print(
        f"{word:<25} "
        f"lemma={lemma:<20} "
        f"AoA={aoa:>5.2f} "
        f"Zipf={zipf:>4.2f}"
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