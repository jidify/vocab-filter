from pathlib import Path
from collections import defaultdict
import csv
import re

from lemminflect import getLemma


WORDLIST_PATH = Path("wordlist.txt")
BOOK_PATH = Path("The Humans - Stephen Karam.txt")
AOA_PATH = Path("kuperman-aoa.csv")

MIN_AOA = 5.0


def looks_inflected(word: str) -> bool:
    """
    Heuristique simple :
    on ne tente une lemmatisation que pour des formes
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
    # Cas 1 : le mot exact existe dans le dataset AoA
    # --------------------------------------------------

    if key in aoa_scores:
        lemma = key
        aoa = aoa_scores[key]

    else:
        # --------------------------------------------------
        # Cas 2 :
        # le mot exact n'existe pas dans AoA.
        # On ne tente la lemmatisation que s'il ressemble
        # à une forme fléchie.
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

        # Parmi les lemmes possibles, on prend celui
        # acquis le plus tôt.
        #
        # Cela évite de considérer artificiellement
        # une forme dérivée d'un mot courant comme difficile.
        lemma, aoa = min(
            aoa_candidates,
            key=lambda item: item[1]
        )

    # --------------------------------------------------
    # Appliquer le seuil AoA
    # --------------------------------------------------

    if aoa < MIN_AOA:
        continue

    # --------------------------------------------------
    # Éviter plusieurs formes du même lemme
    # --------------------------------------------------

    if lemma in seen_lemmas:
        continue

    vocabulary.append(
        (word, lemma, aoa)
    )

    seen_lemmas.add(lemma)


# --------------------------------------------------
# Regrouper par tranche d'âge
# --------------------------------------------------

groups = defaultdict(list)

for word, lemma, aoa in vocabulary:
    start = int(aoa)
    end = start + 1

    groups[(start, end)].append(
        (word, lemma, aoa)
    )


# --------------------------------------------------
# Affichage
# --------------------------------------------------

print("=== VOCABULARY PAR AoA ===")


for start, end in sorted(groups):

    entries = groups[(start, end)]

    print()
    print(
        f"=== AoA {start} -> {end} ==="
    )

    # Trier les mots à l'intérieur de chaque tranche
    # par score AoA croissant
    entries.sort(
        key=lambda item: item[2]
    )

    for word, lemma, aoa in entries:

        print(
            f"{word:<25} "
            f"lemma={lemma:<20} "
            f"AoA={aoa:.2f}"
        )

    print()
    print(
        f"Nombre de mots {start}-{end} : "
        f"{len(entries)}"
    )


# --------------------------------------------------
# Total
# --------------------------------------------------

print()
print("=" * 60)

print(
    f"Nombre total de mots : {len(vocabulary)}"
)