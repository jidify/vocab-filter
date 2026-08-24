from pathlib import Path
import xml.etree.ElementTree as ET
import wn


WONEF_PATH = Path("wonef-fscore.xml")


WORDS_TO_CHECK = [
    "take",
    "bow",
    "mess",
    "shut",
    "run",
]


# --------------------------------------------------
# WordNet anglais basé sur WordNet 3.0
# --------------------------------------------------

EN = wn.Wordnet("omw-en:1.4")


# --------------------------------------------------
# Charger WoNeF
# --------------------------------------------------

tree = ET.parse(WONEF_PATH)
root = tree.getroot()


# --------------------------------------------------
# Construire :
#
# synset_id WordNet 3.0
# -> liste de candidats français
# --------------------------------------------------

wonef_by_id = {}


for synset in root.iter():

    tag = (
        synset.tag
        .split("}")[-1]
        .upper()
    )

    if tag != "SYNSET":
        continue

    synset_id = None
    literals = []

    for child in synset:

        child_tag = (
            child.tag
            .split("}")[-1]
            .upper()
        )

        if child_tag == "ID":

            if child.text:
                synset_id = (
                    child.text
                    .strip()
                )

        elif child_tag == "SYNONYM":

            for literal_element in child.iter():

                literal_tag = (
                    literal_element.tag
                    .split("}")[-1]
                    .upper()
                )

                if literal_tag != "LITERAL":
                    continue

                if not literal_element.text:
                    continue

                literal = (
                    literal_element.text
                    .strip()
                )

                if literal == "_EMPTY_":
                    continue

                if literal not in literals:
                    literals.append(literal)

    if synset_id:

        wonef_by_id[synset_id] = literals


print(
    f"Synsets WoNeF indexés : "
    f"{len(wonef_by_id)}"
)


# --------------------------------------------------
# Convertir ID synset wn -> ID WoNeF
#
# Exemple :
#
# omw-en-02880008-n
#
# ->
#
# eng-30-02880008-n
# --------------------------------------------------

def wn_id_to_wonef_id(synset):

    synset_id = synset.id

    parts = synset_id.split("-")

    if len(parts) < 2:
        return None

    pos = parts[-1]
    offset = parts[-2]

    if not offset.isdigit():
        return None

    return f"eng-30-{offset}-{pos}"


# --------------------------------------------------
# Nettoyage léger des candidats français
# --------------------------------------------------

def clean_french_candidates(
    french_candidates,
    english_lemmas
):

    cleaned = []

    english_keys = {
        lemma.casefold()
        for lemma in english_lemmas
    }

    for candidate in french_candidates:

        candidate = candidate.strip()

        if not candidate:
            continue

        if candidate == "_EMPTY_":
            continue

        # Enlever les candidats identiques
        # à des lemmes anglais du synset.
        if candidate.casefold() in english_keys:
            continue

        if candidate not in cleaned:
            cleaned.append(candidate)

    return cleaned


# --------------------------------------------------
# Récupérer les lemmes anglais d'un synset
# --------------------------------------------------

def get_english_lemmas(synset):

    lemmas = []

    try:
        words = synset.words()

    except Exception:
        words = []

    for word in words:

        try:
            lemma = (
                word.lemma()
                .replace("_", " ")
            )

        except Exception:
            continue

        if lemma not in lemmas:
            lemmas.append(lemma)

    return lemmas


# --------------------------------------------------
# Récupérer la définition anglaise
# --------------------------------------------------

def get_definition(synset):

    try:
        return synset.definition() or ""

    except Exception:
        return ""


# --------------------------------------------------
# Afficher les sens d'un mot
# --------------------------------------------------

def inspect_word(word):

    print()
    print()
    print("=" * 80)
    print(word.upper())
    print("=" * 80)

    synsets = EN.synsets(word)

    if not synsets:

        print(
            "Aucun synset WordNet trouvé."
        )

        return

    for index, synset in enumerate(
        synsets,
        start=1
    ):

        wonef_id = wn_id_to_wonef_id(
            synset
        )

        english_lemmas = (
            get_english_lemmas(
                synset
            )
        )

        definition = (
            get_definition(
                synset
            )
        )

        french_candidates = []

        if wonef_id:

            french_candidates = (
                wonef_by_id.get(
                    wonef_id,
                    []
                )
            )

        french_candidates = (
            clean_french_candidates(
                french_candidates,
                english_lemmas
            )
        )

        print()
        print("-" * 80)

        print(
            f"Sens {index}"
        )

        print(
            f"Synset : {wonef_id or '?'}"
        )

        print(
            f"POS    : {synset.pos}"
        )

        if english_lemmas:

            print(
                "EN     : "
                + ", ".join(
                    english_lemmas
                )
            )

        else:

            print(
                "EN     : ?"
            )

        if definition:

            print(
                f"DEF    : {definition}"
            )

        else:

            print(
                "DEF    : ?"
            )

        if french_candidates:

            print(
                "FR candidats : "
                + ", ".join(
                    french_candidates
                )
            )

        else:

            print(
                "FR candidats : ?"
            )


# --------------------------------------------------
# Programme principal
# --------------------------------------------------

for word in WORDS_TO_CHECK:

    inspect_word(word)