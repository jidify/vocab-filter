from pathlib import Path
import csv
import wn


CEFR_PATH = Path("cefr.csv")


# --------------------------------------------------
# Mots à inspecter
# --------------------------------------------------

WORDS_TO_CHECK = [
    "take",
    "bow",
    "mess",
    "shut",
    "set",
    "run",
]


# --------------------------------------------------
# Correspondance POS CEFR -> POS WordNet
# --------------------------------------------------

POS_TO_WN = {
    "noun": "n",
    "verb": "v",
    "adjective": "a",
    "adverb": "r",
}


# --------------------------------------------------
# Charger les WordNet
# --------------------------------------------------

EN = wn.Wordnet("omw-en:2.0")
FR = wn.Wordnet("omw-fr:2.0")


# --------------------------------------------------
# Charger CEFR par mot et POS
#
# Exemple :
#
# take:
#   noun -> B1
#   verb -> A1
# --------------------------------------------------

cefr_by_word = {}

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

        pos = (
            row["pos"]
            .strip()
            .casefold()
        )

        cefr = (
            row["CEFR"]
            .strip()
            .upper()
        )

        if not headword:
            continue

        if headword not in cefr_by_word:
            cefr_by_word[headword] = {}

        if pos not in cefr_by_word[headword]:
            cefr_by_word[headword][pos] = set()

        cefr_by_word[headword][pos].add(cefr)


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
# Formatter les niveaux CEFR
# --------------------------------------------------

def format_cefr(levels):

    if not levels:
        return "?"

    return "/".join(
        sorted(
            levels,
            key=lambda level: CEFR_ORDER.get(
                level,
                999
            )
        )
    )


# --------------------------------------------------
# Récupérer le lemma d'un sense
# --------------------------------------------------

def get_sense_lemma(sense):

    try:
        return (
            sense.word()
            .lemma()
            .replace("_", " ")
        )

    except Exception:
        return ""


# --------------------------------------------------
# Traduire uniquement le SENSE correspondant
# au mot anglais étudié
# --------------------------------------------------

def get_french_lemmas_for_word(
    synset,
    english_word
):

    translations = []

    english_key = (
        english_word
        .replace("_", " ")
        .casefold()
    )

    # --------------------------------------------------
    # Trouver les senses du synset correspondant
    # exactement au mot anglais étudié.
    #
    # Exemple :
    #
    # synset = bow + bowknot
    #
    # si on étudie "bow", on ne prend que
    # le sense lexical "bow".
    # --------------------------------------------------

    matching_senses = []

    try:
        senses = synset.senses()

    except Exception:
        senses = []

    for sense in senses:

        lemma = (
            get_sense_lemma(sense)
            .casefold()
        )

        if lemma == english_key:
            matching_senses.append(sense)


    # --------------------------------------------------
    # Traduire chaque sense anglais vers WOLF français
    # --------------------------------------------------

    for sense in matching_senses:

        try:
            french_senses = sense.translate(
                lexicon="omw-fr:2.0"
            )

        except Exception:
            continue

        for french_sense in french_senses:

            lemma = get_sense_lemma(
                french_sense
            )

            if not lemma:
                continue

            if lemma not in translations:
                translations.append(lemma)

    return translations


# --------------------------------------------------
# Récupérer les synonymes anglais
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
# Afficher les sens d'un mot pour un POS donné
# --------------------------------------------------

def show_senses(
    word,
    pos_name,
    cefr_levels
):

    wn_pos = POS_TO_WN.get(
        pos_name
    )

    print()
    print("-" * 70)

    print(
        f"{pos_name.upper()} "
        f"— CEFR={format_cefr(cefr_levels)}"
    )

    print("-" * 70)


    if wn_pos is None:

        print(
            "POS non pris en charge "
            "par WordNet dans ce script."
        )

        return


    # --------------------------------------------------
    # Chercher les synsets anglais du mot
    # pour ce POS
    # --------------------------------------------------

    try:
        synsets = EN.synsets(
            word,
            pos=wn_pos
        )

    except Exception as exc:

        print(
            f"Erreur WordNet : {exc}"
        )

        return


    if not synsets:

        print(
            "Aucun sens trouvé "
            "dans WordNet anglais."
        )

        return


    # --------------------------------------------------
    # Afficher chaque sens
    # --------------------------------------------------

    for index, synset in enumerate(
        synsets,
        start=1
    ):

        print()
        print(f"Sens {index}")


        # --------------------------------------------------
        # Définition anglaise
        # --------------------------------------------------

        try:
            definition = (
                synset.definition()
                or ""
            )

        except Exception:
            definition = ""


        if definition:

            print(
                f"  EN : {definition}"
            )

        else:

            print(
                "  EN : ?"
            )


        # --------------------------------------------------
        # Synonymes anglais
        # --------------------------------------------------

        english_lemmas = (
            get_english_lemmas(
                synset
            )
        )


        if english_lemmas:

            print(
                "  Synonymes EN : "
                + ", ".join(
                    english_lemmas
                )
            )

        else:

            print(
                "  Synonymes EN : ?"
            )


        # --------------------------------------------------
        # Traductions françaises basées sur
        # le sense anglais exact
        # --------------------------------------------------

        french_lemmas = (
            get_french_lemmas_for_word(
                synset,
                word
            )
        )


        if french_lemmas:

            print(
                "  FR : "
                + ", ".join(
                    french_lemmas
                )
            )

        else:

            print(
                "  FR : ?"
            )


# --------------------------------------------------
# Inspecter un mot
# --------------------------------------------------

def inspect_word(word):

    key = (
        word
        .strip()
        .casefold()
    )

    print()
    print()
    print("=" * 70)
    print(word.upper())
    print("=" * 70)


    pos_data = cefr_by_word.get(
        key
    )


    # --------------------------------------------------
    # Si le mot n'existe pas dans le fichier CEFR,
    # on essaie quand même les 4 POS WordNet.
    # --------------------------------------------------

    if not pos_data:

        print(
            "Aucune entrée CEFR trouvée."
        )

        for pos_name in (
            "noun",
            "verb",
            "adjective",
            "adverb"
        ):

            wn_pos = POS_TO_WN[
                pos_name
            ]

            try:
                synsets = EN.synsets(
                    key,
                    pos=wn_pos
                )

            except Exception:
                synsets = []


            if synsets:

                show_senses(
                    key,
                    pos_name,
                    set()
                )

        return


    # --------------------------------------------------
    # Sinon, afficher uniquement les POS
    # présents dans cefr.csv
    # --------------------------------------------------

    pos_order = {
        "noun": 1,
        "verb": 2,
        "adjective": 3,
        "adverb": 4,
    }


    sorted_pos = sorted(
        pos_data.items(),
        key=lambda item: pos_order.get(
            item[0],
            999
        )
    )


    for (
        pos_name,
        cefr_levels
    ) in sorted_pos:

        show_senses(
            key,
            pos_name,
            cefr_levels
        )


# --------------------------------------------------
# Programme principal
# --------------------------------------------------

for word in WORDS_TO_CHECK:

    inspect_word(
        word
    )