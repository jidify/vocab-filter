from pathlib import Path
import csv


CEFR_PATH = Path("cefr.csv")


# --------------------------------------------------
# Mots à inspecter
# --------------------------------------------------

WORDS_TO_CHECK = [
    "take",
    "get",
    "set",
    "run",
    "bow",
    "mess",
    "shut",
]


# --------------------------------------------------
# Charger le CSV CEFR
# --------------------------------------------------

rows_by_word = {}

with CEFR_PATH.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter=";"
    )

    for row in reader:

        headword = (
            row["headword"]
            .strip()
            .casefold()
        )

        if headword not in rows_by_word:
            rows_by_word[headword] = []

        rows_by_word[headword].append(row)


# --------------------------------------------------
# Afficher les infos
# --------------------------------------------------

for word in WORDS_TO_CHECK:

    key = word.casefold()

    print()
    print("=" * 70)
    print(f"{word}")
    print("=" * 70)

    rows = rows_by_word.get(key)

    if not rows:
        print("Aucune entrée CEFR trouvée.")
        continue

    for row in rows:

        pos = (
            row.get("pos", "")
            .strip()
        )

        cefr = (
            row.get("CEFR", "")
            .strip()
        )

        print(
            f"POS={pos:<15} "
            f"CEFR={cefr}"
        )