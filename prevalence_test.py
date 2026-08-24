from pathlib import Path
import csv
import re
import statistics


BOOK_PATH = Path("The Humans - Stephen Karam.txt")
PREVALENCE_PATH = Path("word-prevalence.txt")
CEFR_PATH = Path("cefr.csv")


# --------------------------------------------------
# Réglage Pknown
# --------------------------------------------------

MIN_PKNOWN = 0.99
MAX_PKNOWN = 1.01


# --------------------------------------------------
# Niveaux CEFR à exclure
#
# Le mot n'est exclu que si TOUS ses niveaux connus
# appartiennent à cet ensemble.
#
# Exemples :
# {"A1", "A2"}
# {"A1", "A2", "B1"}
# set()
#
# CEFR=? est toujours conservé.
# --------------------------------------------------

EXCLUDED_CEFR = {"A1", "A2"}


# --------------------------------------------------
# Ordre CEFR
# --------------------------------------------------

CEFR_ORDER = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}


# --------------------------------------------------
# Charger CEFR
#
# On conserve TOUS les niveaux trouvés
# pour chaque headword.
# --------------------------------------------------

cefr_scores = {}

with CEFR_PATH.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter=";"
    )

    for row in reader:

        word = (
            row["headword"]
            .strip()
            .casefold()
        )

        level = (
            row["CEFR"]
            .strip()
            .upper()
        )

        if level not in CEFR_ORDER:
            continue

        if word not in cefr_scores:
            cefr_scores[word] = set()

        cefr_scores[word].add(level)


# --------------------------------------------------
# Récupérer tous les niveaux CEFR d'un mot
# --------------------------------------------------

def get_cefr_levels(word: str) -> tuple[str, ...]:

    levels = cefr_scores.get(
        word.casefold()
    )

    if not levels:
        return ()

    return tuple(
        sorted(
            levels,
            key=lambda level: CEFR_ORDER[level]
        )
    )


# --------------------------------------------------
# Affichage CEFR
# --------------------------------------------------

def format_cefr(
    levels: tuple[str, ...]
) -> str:

    if not levels:
        return "?"

    return "/".join(levels)


# --------------------------------------------------
# Déterminer si un mot doit être exclu
#
# Exemple :
#
# EXCLUDED_CEFR = {"A1", "A2"}
#
# A1           -> exclu
# A1/A2        -> exclu
# A1/A2/B1     -> gardé
# B1           -> gardé
# ?            -> gardé
# --------------------------------------------------

def should_exclude_cefr(
    levels: tuple[str, ...]
) -> bool:

    # Aucun CEFR connu :
    # on conserve le mot.
    if not levels:
        return False

    return all(
        level in EXCLUDED_CEFR
        for level in levels
    )


# --------------------------------------------------
# Charger Word Prevalence
#
# Format :
# word,Pknown,Nobs,Prevalence,FreqZipfUS
# --------------------------------------------------

prevalence_scores = {}

with PREVALENCE_PATH.open(
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.reader(f)

    for row in reader:

        if len(row) < 5:
            continue

        word = (
            row[0]
            .strip()
            .casefold()
        )

        try:
            pknown = float(row[1])
            nobs = int(row[2])
            prevalence = float(row[3])
            freq_zipf_us = float(row[4])

        except ValueError:
            continue

        prevalence_scores[word] = {
            "pknown": pknown,
            "nobs": nobs,
            "prevalence": prevalence,
            "freq_zipf_us": freq_zipf_us,
        }


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
# Construire la liste des mots uniques du livre
# présents dans Word Prevalence
# --------------------------------------------------

book_words = []
seen = set()


for word in words_in_book:

    key = word.casefold()

    if key in seen:
        continue

    seen.add(key)

    data = prevalence_scores.get(key)

    if data is None:
        continue

    cefr_levels = get_cefr_levels(key)

    # Exclusion CEFR prudente :
    # uniquement si TOUS les niveaux sont exclus.
    if should_exclude_cefr(cefr_levels):
        continue

    book_words.append(
        (
            word,
            data["prevalence"],
            data["pknown"],
            data["nobs"],
            data["freq_zipf_us"],
            cefr_levels,
        )
    )


# --------------------------------------------------
# Distribution Pknown après filtre CEFR
# --------------------------------------------------

buckets = {
    "0.00-0.50": 0,
    "0.50-0.70": 0,
    "0.70-0.80": 0,
    "0.80-0.90": 0,
    "0.90-0.95": 0,
    "0.95-0.99": 0,
    "0.99-1.00": 0,
    "1.00": 0,
}


for (
    _,
    _,
    pknown,
    _,
    _,
    _
) in book_words:

    if pknown < 0.50:
        buckets["0.00-0.50"] += 1

    elif pknown < 0.70:
        buckets["0.50-0.70"] += 1

    elif pknown < 0.80:
        buckets["0.70-0.80"] += 1

    elif pknown < 0.90:
        buckets["0.80-0.90"] += 1

    elif pknown < 0.95:
        buckets["0.90-0.95"] += 1

    elif pknown < 0.99:
        buckets["0.95-0.99"] += 1

    elif pknown < 1.00:
        buckets["0.99-1.00"] += 1

    else:
        buckets["1.00"] += 1


# --------------------------------------------------
# Afficher distribution
# --------------------------------------------------

print()
print(
    "=== DISTRIBUTION PKNOWN "
    "APRÈS FILTRE CEFR ==="
)

print()

if EXCLUDED_CEFR:

    excluded_display = ", ".join(
        sorted(
            EXCLUDED_CEFR,
            key=lambda level: CEFR_ORDER[level]
        )
    )

else:
    excluded_display = "aucun"


print(
    f"CEFR exclus : {excluded_display}"
)

print(
    "Règle : exclusion seulement si "
    "tous les niveaux CEFR du mot sont exclus."
)

print()


for label, count in buckets.items():

    percent = (
        count / len(book_words) * 100
        if book_words
        else 0
    )

    print(
        f"{label:<12} : "
        f"{count:>5} mots "
        f"({percent:>5.1f}%)"
    )


print()

print(
    "Total de mots uniques après filtre CEFR : "
    f"{len(book_words)}"
)


# --------------------------------------------------
# Filtre Pknown
# --------------------------------------------------

results = []


for (
    word,
    prevalence,
    pknown,
    nobs,
    freq_zipf_us,
    cefr_levels
) in book_words:

    if not (
        MIN_PKNOWN <= pknown < MAX_PKNOWN
    ):
        continue

    results.append(
        (
            word,
            prevalence,
            pknown,
            nobs,
            freq_zipf_us,
            cefr_levels,
        )
    )


# --------------------------------------------------
# Trier par ZipfUS croissant
#
# Plus ZipfUS est faible,
# plus le mot est rare.
# --------------------------------------------------

results.sort(
    key=lambda item: item[4]
)


# --------------------------------------------------
# Affichage des résultats
# --------------------------------------------------

print()
print("=" * 125)
print()

print(
    "=== MOTS CONNUS PAR >= 99% DES NATIFS, "
    "TRIÉS PAR RARETÉ ==="
)

print()

print(
    f"Pknown : "
    f"{MIN_PKNOWN:.2f} -> {MAX_PKNOWN:.2f}"
)

print()


for (
    word,
    prevalence,
    pknown,
    nobs,
    freq_zipf_us,
    cefr_levels
) in results:

    unknown_percent = (
        1.0 - pknown
    ) * 100

    cefr_display = format_cefr(
        cefr_levels
    )

    print(
        f"{word:<25} "
        f"Pknown={pknown:>5.3f} "
        f"Unknown={unknown_percent:>5.1f}% "
        f"Prev={prevalence:>6.3f} "
        f"ZipfUS={freq_zipf_us:>5.2f} "
        f"CEFR={cefr_display:<14} "
        f"N={nobs}"
    )


# --------------------------------------------------
# Statistiques
# --------------------------------------------------

print()
print("=" * 125)

print(
    "Nombre total de mots après filtres : "
    f"{len(results)}"
)


if results:

    pknown_values = [
        item[2]
        for item in results
    ]

    prevalence_values = [
        item[1]
        for item in results
    ]

    zipf_values = [
        item[4]
        for item in results
    ]


    print()
    print("=== STATISTIQUES PKNOWN ===")

    print(
        f"Minimum : "
        f"{min(pknown_values):.3f}"
    )

    print(
        f"Maximum : "
        f"{max(pknown_values):.3f}"
    )

    print(
        f"Moyenne : "
        f"{statistics.mean(pknown_values):.3f}"
    )

    print(
        f"Médiane : "
        f"{statistics.median(pknown_values):.3f}"
    )


    print()
    print("=== STATISTIQUES PREVALENCE ===")

    print(
        f"Minimum : "
        f"{min(prevalence_values):.3f}"
    )

    print(
        f"Maximum : "
        f"{max(prevalence_values):.3f}"
    )

    print(
        f"Moyenne : "
        f"{statistics.mean(prevalence_values):.3f}"
    )

    print(
        f"Médiane : "
        f"{statistics.median(prevalence_values):.3f}"
    )


    print()
    print("=== STATISTIQUES ZIPF US ===")

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
    # Répartition des combinaisons CEFR
    # --------------------------------------------------

    print()
    print("=== RÉPARTITION CEFR FINALE ===")

    cefr_counts = {}


    for item in results:

        cefr_levels = item[5]

        cefr_display = format_cefr(
            cefr_levels
        )

        cefr_counts[cefr_display] = (
            cefr_counts.get(
                cefr_display,
                0
            )
            + 1
        )


    def cefr_sort_key(
        item
    ):

        label = item[0]

        if label == "?":
            return (999, label)

        levels = label.split("/")

        return (
            min(
                CEFR_ORDER[level]
                for level in levels
            ),
            label
        )


    for (
        cefr_display,
        count
    ) in sorted(
        cefr_counts.items(),
        key=cefr_sort_key
    ):

        print(
            f"{cefr_display:<15} : "
            f"{count}"
        )