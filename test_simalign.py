import re
from simalign import SentenceAligner


# ============================================================
# TESTS
# ============================================================

TESTS = [
    {
        "target": "view",
        "english": "I wish you had more of a view.",
        "french": "J’aimerais que vous ayez une meilleure vue.",
    },
    {
        "target": "butts",
        "english": "It's an alley full of cigarette butts.",
        "french": "C’est une ruelle pleine de mégots de cigarettes.",
    },
    {
        "target": "alley",
        "english": "It's an alley full of cigarette butts.",
        "french": "C’est une ruelle pleine de mégots de cigarettes.",
    },
    {
        "target": "access",
        "english": "He's the only one who has access.",
        "french": "Il est le seul à avoir la clé.",
    },
    {
        "target": "standard",
        "english": "No, that's standard for a ground-floor apartment.",
        "french": "Non, c’est tout à fait normal pour un appartement au rez-de-chaussée.",
    },
]


# ============================================================
# TOKENISATION SIMPLE
#
# Suffisante pour notre test.
# On garde les apostrophes dans les mots.
# ============================================================

TOKEN_RE = re.compile(
    r"\w+(?:[’'-]\w+)*|[^\w\s]",
    re.UNICODE,
)


def tokenize(text):
    return TOKEN_RE.findall(text)


# ============================================================
# TROUVER LE MOT CIBLE
# ============================================================

def find_target_index(tokens, target):

    target_cf = target.casefold()

    for i, token in enumerate(tokens):

        if token.casefold() == target_cf:
            return i

    return None


# ============================================================
# AFFICHAGE D'UNE MÉTHODE D'ALIGNEMENT
# ============================================================

def show_alignment(
    method,
    pairs,
    src_tokens,
    trg_tokens,
    target_index
):

    print(f"\n--- {method} ---")

    # Toutes les paires
    for src_i, trg_i in pairs:

        print(
            f"{src_i:>2} {src_tokens[src_i]:<15}"
            f" ↔ "
            f"{trg_i:>2} {trg_tokens[trg_i]}"
        )

    # Alignements du mot cible uniquement
    aligned_targets = [
        trg_i
        for src_i, trg_i in pairs
        if src_i == target_index
    ]

    print()

    if not aligned_targets:

        print(
            "TARGET → aucun mot français aligné"
        )

        return []

    french_words = [
        trg_tokens[i]
        for i in aligned_targets
    ]

    print(
        "TARGET → "
        + " / ".join(french_words)
    )

    return french_words


# ============================================================
# CHARGEMENT SIMALIGN
# ============================================================

print("Chargement de SimAlign...")

aligner = SentenceAligner(
    model="bert",
    token_type="bpe",
    matching_methods="mai",
)

print("SimAlign chargé.")
print()


# ============================================================
# ANALYSE
# ============================================================

def analyze(test):

    target = test["target"]

    src_tokens = tokenize(
        test["english"]
    )

    trg_tokens = tokenize(
        test["french"]
    )

    target_index = find_target_index(
        src_tokens,
        target
    )

    print()
    print("=" * 100)
    print(f"TARGET : {target}")
    print(f"EN     : {test['english']}")
    print(f"FR     : {test['french']}")
    print("=" * 100)

    print()
    print("TOKENS EN:")

    for i, token in enumerate(src_tokens):
        marker = (
            "  <<< TARGET"
            if i == target_index
            else ""
        )

        print(
            f"{i:>2}: {token}{marker}"
        )

    print()

    print("TOKENS FR:")

    for i, token in enumerate(trg_tokens):
        print(
            f"{i:>2}: {token}"
        )

    if target_index is None:

        print()
        print(
            f'ERREUR : "{target}" '
            f"non trouvé dans la phrase."
        )

        return

    alignments = (
        aligner.get_word_aligns(
            src_tokens,
            trg_tokens
        )
    )

    results_by_method = {}

    for method, pairs in alignments.items():

        french_words = show_alignment(
            method,
            pairs,
            src_tokens,
            trg_tokens,
            target_index,
        )

        results_by_method[
            method
        ] = french_words


    # ========================================================
    # CONSENSUS
    # ========================================================

    votes = {}

    for method_words in (
        results_by_method.values()
    ):

        for word in method_words:

            key = word.casefold()

            votes[key] = (
                votes.get(key, 0)
                + 1
            )

    print()
    print(">>> CONSENSUS POUR LA CIBLE")

    if not votes:

        print(
            "Aucun alignement."
        )

        return

    ranked = sorted(
        votes.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    for french_word, count in ranked:

        print(
            f"{french_word:<20}"
            f"{count}/"
            f"{len(results_by_method)} méthodes"
        )


# ============================================================
# PROGRAMME
# ============================================================

for test in TESTS:
    analyze(test)