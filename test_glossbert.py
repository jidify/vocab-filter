from glossbert import GlossBERT


TESTS = [
    {
        "word": "view",
        "sentence": "I wish you had more of a view.",
    },
    {
        "word": "butts",
        "lemma": "butt",
        "sentence": "It's an alley full of cigarette butts.",
    },
    {
        "word": "alley",
        "sentence": "It's an alley full of cigarette butts.",
    },
    {
        "word": "access",
        "sentence": "He's the only one who has access.",
    },
    {
        "word": "standard",
        "sentence": "No, that's standard for a ground-floor apartment.",
    },
]


print("Chargement de GlossBERT...")
gloss = GlossBERT()
print("GlossBERT chargé.")
print()


def analyze(test):

    sentence = test["sentence"]
    surface = test["word"]
    target_word = test.get("lemma", surface)

    start_idx = sentence.casefold().find(
        surface.casefold()
    )

    if start_idx == -1:
        print(
            f'ERREUR : "{surface}" introuvable dans :'
        )
        print(sentence)
        return

    end_idx = start_idx + len(surface)

    print("=" * 100)
    print(f"TARGET   : {surface}")
    print(f"WORDNET  : {target_word}")
    print(f"SENTENCE : {sentence}")
    print(
        f"INDEX    : {start_idx}:{end_idx}"
    )
    print("=" * 100)

    results = gloss(
        sentence,
        start_idx,
        end_idx,
        target_word,
    )

    if not results:
        print("Aucun résultat.")
        print()
        return

    for rank, (
        score,
        synset
    ) in enumerate(
        results[:5],
        start=1
    ):

        print(
            f"{rank}. {synset.name()}"
        )

        print(
            f"   score      : {score:.6f}"
        )

        print(
            f"   definition : {synset.definition()}"
        )

        print(
            "   synonyms   : "
            + ", ".join(
                lemma.name().replace("_", " ")
                for lemma in synset.lemmas()
            )
        )

        print()

    best_score, best_synset = results[0]

    print(">>> BEST")
    print(best_synset.name())
    print(best_synset.definition())
    print(f"score = {best_score:.6f}")
    print()


for test in TESTS:
    analyze(test)