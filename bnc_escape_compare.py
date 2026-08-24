from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


BNC_PATH = Path(
    "spoken-bnc2014"
) / "spoken" / "tagged"


# --------------------------------------------------
# Expressions à comparer
#
# On peut en ajouter/enlever facilement.
# --------------------------------------------------

TARGETS = [
    ("run away", ["run", "away"]),
    ("get away", ["get", "away"]),
    ("flee", ["flee"]),
    ("escape", ["escape"]),
    ("take off", ["take", "off"]),
    ("bolt", ["bolt"]),
]


# --------------------------------------------------
# Nombre maximum d'exemples à conserver
# pour chaque expression
# --------------------------------------------------

MAX_EXAMPLES = 10


# --------------------------------------------------
# Résultats
# --------------------------------------------------

counts = Counter()

files_by_target = defaultdict(set)

examples = defaultdict(list)

total_words = 0
total_files = 0


# --------------------------------------------------
# Transformer une unité <u> en liste de tokens
# --------------------------------------------------

def get_tokens(utterance):

    tokens = []

    for element in utterance:

        # On ne prend que les mots <w>
        if element.tag != "w":
            continue

        surface = (
            element.text or ""
        ).strip()

        lemma = (
            element.attrib.get(
                "lemma",
                ""
            )
            .strip()
            .casefold()
        )

        pos = (
            element.attrib.get(
                "pos",
                ""
            )
        )

        word_class = (
            element.attrib.get(
                "class",
                ""
            )
        )

        usas = (
            element.attrib.get(
                "usas",
                ""
            )
        )

        if not surface:
            continue

        tokens.append(
            {
                "surface": surface,
                "surface_lower": surface.casefold(),
                "lemma": lemma,
                "pos": pos,
                "class": word_class,
                "usas": usas,
            }
        )

    return tokens


# --------------------------------------------------
# Reconstituer une phrase approximative
# --------------------------------------------------

def tokens_to_text(tokens):

    words = [
        token["surface"]
        for token in tokens
    ]

    text = " ".join(words)

    # nettoyage très léger de la ponctuation
    replacements = {
        " .": ".",
        " ,": ",",
        " ?": "?",
        " !": "!",
        " ;": ";",
        " :": ":",
        " n't": "n't",
        " 's": "'s",
        " 're": "'re",
        " 've": "'ve",
        " 'll": "'ll",
        " 'd": "'d",
        " 'm": "'m",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    return text


# --------------------------------------------------
# Détecter une cible dans une séquence
#
# On travaille sur les LEMMES.
#
# Ex:
# ran away
# running away
#
# deviennent tous les deux :
#
# run + away
# --------------------------------------------------

def find_target_positions(
    tokens,
    target_lemmas
):

    positions = []

    target_len = len(
        target_lemmas
    )

    token_lemmas = [
        token["lemma"]
        for token in tokens
    ]

    for i in range(
        len(token_lemmas)
        - target_len
        + 1
    ):

        chunk = token_lemmas[
            i:i + target_len
        ]

        if chunk == target_lemmas:

            positions.append(i)

    return positions


# --------------------------------------------------
# Parcourir tous les fichiers XML
# --------------------------------------------------

xml_files = list(
    BNC_PATH.glob(
        "*-tgd.xml"
    )
)

print(
    f"Fichiers XML trouvés : "
    f"{len(xml_files)}"
)

print()


for file_index, xml_path in enumerate(
    xml_files,
    start=1
):

    total_files += 1

    try:
        tree = ET.parse(
            xml_path
        )

    except ET.ParseError as exc:

        print(
            f"Erreur XML dans "
            f"{xml_path.name}: {exc}"
        )

        continue


    root = tree.getroot()


    # --------------------------------------------------
    # Une conversation contient plusieurs <u>
    # --------------------------------------------------

    for utterance in root.findall(
        ".//u"
    ):

        tokens = get_tokens(
            utterance
        )

        if not tokens:
            continue

        total_words += len(tokens)

        context_text = (
            tokens_to_text(
                tokens
            )
        )

        speaker = (
            utterance.attrib.get(
                "who",
                "?"
            )
        )


        # --------------------------------------------------
        # Tester chaque expression
        # --------------------------------------------------

        for (
            target_name,
            target_lemmas
        ) in TARGETS:

            positions = (
                find_target_positions(
                    tokens,
                    target_lemmas
                )
            )

            if not positions:
                continue


            # Chaque occurrence est comptée
            counts[target_name] += (
                len(positions)
            )


            # Diversité :
            # dans combien de fichiers/conversations ?
            files_by_target[
                target_name
            ].add(
                xml_path.name
            )


            # Garder quelques exemples seulement
            if (
                len(
                    examples[
                        target_name
                    ]
                )
                < MAX_EXAMPLES
            ):

                examples[
                    target_name
                ].append(
                    {
                        "file": xml_path.name,
                        "speaker": speaker,
                        "text": context_text,
                    }
                )


    if (
        file_index % 100 == 0
    ):

        print(
            f"{file_index}/"
            f"{len(xml_files)} "
            f"fichiers traités..."
        )


# --------------------------------------------------
# Affichage résultats
# --------------------------------------------------

print()
print("=" * 100)

print(
    "SPOKEN BNC2014 — "
    "COMPARAISON DES EXPRESSIONS"
)

print("=" * 100)

print()

print(
    f"Fichiers analysés : "
    f"{total_files}"
)

print(
    f"Tokens <w> analysés : "
    f"{total_words:,}"
)

print()


# --------------------------------------------------
# Trier par nombre d'occurrences
# --------------------------------------------------

sorted_targets = sorted(
    TARGETS,
    key=lambda item: counts[
        item[0]
    ],
    reverse=True
)


print(
    f"{'Expression':<20}"
    f"{'Occurrences':>12}"
    f"{'Conversations':>16}"
    f"{'par million':>15}"
)

print("-" * 65)


for target_name, _ in sorted_targets:

    count = counts[
        target_name
    ]

    conversation_count = len(
        files_by_target[
            target_name
        ]
    )

    per_million = (
        count
        / total_words
        * 1_000_000
        if total_words
        else 0
    )

    print(
        f"{target_name:<20}"
        f"{count:>12}"
        f"{conversation_count:>16}"
        f"{per_million:>15.2f}"
    )


# --------------------------------------------------
# Exemples
# --------------------------------------------------

print()
print()
print("=" * 100)

print(
    "EXEMPLES DE CONTEXTE"
)

print("=" * 100)


for target_name, _ in sorted_targets:

    print()
    print()
    print(
        f"### {target_name.upper()}"
    )

    print(
        f"Occurrences : "
        f"{counts[target_name]}"
    )

    print(
        f"Conversations : "
        f"{len(files_by_target[target_name])}"
    )

    print()

    target_examples = (
        examples[target_name]
    )

    if not target_examples:

        print(
            "Aucun exemple."
        )

        continue

    for i, example in enumerate(
        target_examples,
        start=1
    ):

        print(
            f"{i}. "
            f"[{example['file']} / "
            f"{example['speaker']}]"
        )

        print(
            f"   {example['text']}"
        )

        print()