from pathlib import Path
from collections import Counter
from nltk.corpus import wordnet as wn


# --------------------------------------------------
# Expression à comparer
# --------------------------------------------------

TARGET = "run away"

OMSTI_ROOT = Path(
    "one-million-sense-tagged-instances-wn30"
)


# --------------------------------------------------
# Convertir l'expression en nom de fichier OMSTI
#
# "run away" -> "run_away.key"
# --------------------------------------------------

OMSTI_NAME = TARGET.replace(" ", "_")

OMSTI_KEY = (
    OMSTI_ROOT
    / "verb"
    / f"{OMSTI_NAME}.key"
)


# --------------------------------------------------
# Trouver les sens WordNet correspondant à
# l'expression exacte.
#
# Exemple :
# run away -> lemma "run_away"
# --------------------------------------------------

def get_semcor_counts(expression: str):

    counts = Counter()

    lemma_name = expression.replace(
        " ",
        "_"
    ).casefold()

    # Chercher tous les synsets verbaux
    # et ne garder que les lemmes exacts.
    for synset in wn.all_synsets(
        pos=wn.VERB
    ):

        for lemma in synset.lemmas():

            if (
                lemma.name().casefold()
                == lemma_name
            ):

                count = lemma.count()

                if count > 0:
                    counts[
                        synset.name()
                    ] += count

    return counts


# --------------------------------------------------
# Compteurs OMSTI
# --------------------------------------------------

def get_omsti_counts(path: Path):

    counts = Counter()

    if not path.exists():

        print(
            f"Fichier OMSTI introuvable : {path}"
        )

        return counts

    with path.open(
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            sense_key = parts[2]

            try:
                lemma = wn.lemma_from_key(
                    sense_key
                )

                synset = lemma.synset()

            except Exception:
                continue

            counts[
                synset.name()
            ] += 1

    return counts


# --------------------------------------------------
# Charger les deux sources
# --------------------------------------------------

semcor_counts = get_semcor_counts(
    TARGET
)

omsti_counts = get_omsti_counts(
    OMSTI_KEY
)


semcor_total = sum(
    semcor_counts.values()
)

omsti_total = sum(
    omsti_counts.values()
)


# --------------------------------------------------
# Tous les synsets trouvés
# --------------------------------------------------

all_synsets = (
    set(semcor_counts)
    | set(omsti_counts)
)


# --------------------------------------------------
# Trier par fréquence OMSTI décroissante
# --------------------------------------------------

all_synsets = sorted(
    all_synsets,
    key=lambda s: omsti_counts.get(
        s,
        0
    ),
    reverse=True
)


# --------------------------------------------------
# Affichage
# --------------------------------------------------

print()
print("=" * 120)

print(
    f"{TARGET.upper()} — VERB"
)

print("=" * 120)

print(
    f"Fichier OMSTI : {OMSTI_KEY}"
)

print(
    f"SemCor total : {semcor_total}"
)

print(
    f"OMSTI total  : {omsti_total}"
)

print()

print(
    f"{'Synset':<22} "
    f"{'SemCor':>8} "
    f"{'SemCor %':>10} "
    f"{'OMSTI':>8} "
    f"{'OMSTI %':>10} "
    f"Definition"
)

print("-" * 120)


for synset_name in all_synsets:

    semcor_count = semcor_counts.get(
        synset_name,
        0
    )

    omsti_count = omsti_counts.get(
        synset_name,
        0
    )

    semcor_pct = (
        semcor_count
        / semcor_total
        * 100
        if semcor_total
        else 0
    )

    omsti_pct = (
        omsti_count
        / omsti_total
        * 100
        if omsti_total
        else 0
    )

    try:
        synset = wn.synset(
            synset_name
        )

        definition = (
            synset.definition()
        )

    except Exception:
        definition = "?"

    print(
        f"{synset_name:<22} "
        f"{semcor_count:>8} "
        f"{semcor_pct:>9.1f}% "
        f"{omsti_count:>8} "
        f"{omsti_pct:>9.1f}% "
        f"{definition}"
    )