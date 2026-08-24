from pathlib import Path
from collections import Counter
from nltk.corpus import wordnet as wn


# --------------------------------------------------
# Dossier OMSTI
# --------------------------------------------------

OMSTI_PATH = Path(
    "one-million-sense-tagged-instances-wn30"
)


# --------------------------------------------------
# Mots à tester
# --------------------------------------------------

WORDS_TO_CHECK = [
    "take",
    "run",
    "bow",
    "mess",
    "shut",
    "set",
]


# --------------------------------------------------
# Dossiers OMSTI -> POS WordNet
# --------------------------------------------------

POS_DIRS = {
    "noun": "n",
    "verb": "v",
    "adj": "a",
    "adv": "r",
}


# --------------------------------------------------
# Résoudre une sense key WordNet
#
# Exemple :
# take%2:41:04::
#
# -> lemma WordNet
# -> synset
# --------------------------------------------------

def sense_key_to_synset(sense_key):

    try:
        lemma = wn.lemma_from_key(
            sense_key
        )

        return lemma.synset()

    except Exception:
        return None


# --------------------------------------------------
# Lire un fichier OMSTI .key
# --------------------------------------------------

def read_key_file(path):

    counts = Counter()

    total = 0

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

            # Format :
            #
            # take take.750116 take%2:41:04::
            #
            lemma = parts[0]
            instance_id = parts[1]
            sense_key = parts[2]

            counts[sense_key] += 1
            total += 1

    return counts, total


# --------------------------------------------------
# Chercher un mot exact dans OMSTI
#
# On cherche :
#
# noun/take.key
# verb/take.key
# adj/take.key
# adv/take.key
#
# On NE prend PAS :
#
# take_off.key
# take_up.key
# take_over.key
# etc.
# --------------------------------------------------

def find_word_files(word):

    found = []

    filename = f"{word.casefold()}.key"

    for directory, wn_pos in POS_DIRS.items():

        path = (
            OMSTI_PATH
            / directory
            / filename
        )

        if path.exists():

            found.append(
                (
                    directory,
                    wn_pos,
                    path
                )
            )

    return found


# --------------------------------------------------
# Afficher les fréquences d'un mot/POS
# --------------------------------------------------

def inspect_key_file(
    word,
    pos_name,
    path
):

    counts, total = read_key_file(
        path
    )

    print()
    print("-" * 100)

    print(
        f"{pos_name.upper()} "
        f"— {path.name}"
    )

    print("-" * 100)

    print(
        f"Occurrences OMSTI : {total}"
    )

    print()


    if total == 0:

        print(
            "Aucune occurrence."
        )

        return


    # --------------------------------------------------
    # Construire les données de chaque sens
    # --------------------------------------------------

    senses = []

    for sense_key, count in counts.items():

        synset = sense_key_to_synset(
            sense_key
        )

        senses.append(
            {
                "sense_key": sense_key,
                "count": count,
                "synset": synset,
            }
        )


    # --------------------------------------------------
    # Trier par fréquence décroissante
    # --------------------------------------------------

    senses.sort(
        key=lambda item: item["count"],
        reverse=True
    )


    cumulative = 0.0


    for index, item in enumerate(
        senses,
        start=1
    ):

        count = item["count"]

        percentage = (
            count
            / total
            * 100
        )

        cumulative += percentage

        synset = item["synset"]


        print(
            f"Sens {index}"
        )

        print(
            f"  Count      : {count}"
        )

        print(
            f"  Fréquence  : "
            f"{percentage:5.1f}%"
        )

        print(
            f"  Cumul      : "
            f"{cumulative:5.1f}%"
        )

        print(
            f"  Sense key  : "
            f"{item['sense_key']}"
        )


        # --------------------------------------------------
        # Infos WordNet
        # --------------------------------------------------

        if synset is not None:

            print(
                f"  Synset     : "
                f"{synset.name()}"
            )

            lemmas = [
                lemma.name().replace(
                    "_",
                    " "
                )
                for lemma
                in synset.lemmas()
            ]

            print(
                "  EN         : "
                + ", ".join(
                    lemmas
                )
            )

            print(
                "  Definition : "
                + synset.definition()
            )

        else:

            print(
                "  Synset     : ?"
            )

            print(
                "  Definition : "
                "sense key non résolue"
            )


        print()


    print(
        f"Nombre de sens distincts : "
        f"{len(senses)}"
    )


# --------------------------------------------------
# Inspecter un mot
# --------------------------------------------------

def inspect_word(word):

    print()
    print()
    print("=" * 100)
    print(word.upper())
    print("=" * 100)

    files = find_word_files(
        word
    )


    if not files:

        print()
        print(
            "Aucun fichier OMSTI exact "
            "pour ce mot."
        )

        return


    for (
        pos_name,
        wn_pos,
        path
    ) in files:

        inspect_key_file(
            word,
            pos_name,
            path
        )


# --------------------------------------------------
# Programme principal
# --------------------------------------------------

for word in WORDS_TO_CHECK:

    inspect_word(
        word
    )