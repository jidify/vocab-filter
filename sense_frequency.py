from nltk.corpus import wordnet as wn


WORDS_TO_CHECK = [
    "run",
    "take",
    "bow",
    "mess",
    "shut",
    "set",
]


POS_TO_NAME = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}


def inspect_word(word: str):

    print()
    print("=" * 90)
    print(word.upper())
    print("=" * 90)

    # --------------------------------------------------
    # Récupérer tous les synsets contenant le mot
    # --------------------------------------------------

    synsets = wn.synsets(word)

    if not synsets:
        print("Aucun sens WordNet trouvé.")
        return

    senses = []

    # --------------------------------------------------
    # Pour chaque synset, retrouver le lemma correspondant
    # exactement au mot étudié.
    #
    # lemma.count() = fréquence du sens dans les corpus
    # annotés WordNet / SemCor.
    # --------------------------------------------------

    for synset in synsets:

        matching_lemmas = [
            lemma
            for lemma in synset.lemmas()
            if lemma.name().replace("_", " ").casefold()
            == word.casefold()
        ]

        if not matching_lemmas:
            continue

        # Normalement un seul lemma correspondant
        lemma = matching_lemmas[0]

        count = lemma.count()

        senses.append(
            {
                "synset": synset,
                "pos": synset.pos(),
                "count": count,
            }
        )

    # --------------------------------------------------
    # Grouper par POS
    # --------------------------------------------------

    pos_groups = {}

    for item in senses:

        pos = item["pos"]

        # WordNet distingue adjective satellite "s"
        # de adjective "a".
        # Pour notre affichage, on les regroupe.
        if pos == "s":
            pos = "a"

        pos_groups.setdefault(
            pos,
            []
        ).append(
            item
        )

    # --------------------------------------------------
    # Afficher chaque POS séparément
    # --------------------------------------------------

    for pos, items in pos_groups.items():

        pos_name = POS_TO_NAME.get(
            pos,
            pos
        )

        print()
        print("-" * 90)
        print(pos_name.upper())
        print("-" * 90)

        # Trier par fréquence décroissante
        items.sort(
            key=lambda x: x["count"],
            reverse=True
        )

        total_count = sum(
            item["count"]
            for item in items
        )

        cumulative = 0

        for index, item in enumerate(
            items,
            start=1
        ):

            synset = item["synset"]
            count = item["count"]

            if total_count > 0:

                percentage = (
                    count
                    / total_count
                    * 100
                )

            else:
                percentage = 0.0

            cumulative += percentage

            lemmas = [
                lemma.name().replace("_", " ")
                for lemma in synset.lemmas()
            ]

            print()
            print(
                f"Sens {index}"
            )

            print(
                f"  Count      : {count}"
            )

            print(
                f"  Fréquence  : {percentage:5.1f}%"
            )

            print(
                f"  Cumul      : {cumulative:5.1f}%"
            )

            print(
                f"  Synset     : {synset.name()}"
            )

            print(
                f"  EN         : "
                + ", ".join(lemmas)
            )

            print(
                f"  Definition : "
                f"{synset.definition()}"
            )

        print()

        print(
            f"Total occurrences annotées "
            f"pour {word}/{pos_name} : "
            f"{total_count}"
        )


for word in WORDS_TO_CHECK:
    inspect_word(word)