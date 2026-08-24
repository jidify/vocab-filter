import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

import wn
from nltk.corpus import wordnet as nwn
from lemminflect import getAllInflections, getAllInflectionsOOV

from glossbert import GlossBERT


# ============================================================
# RÉGLAGES
# ============================================================

FR_LEXICON = "omw-fr:2.0"
EN_LEXICON = "omw-en:2.0"

WONEF_PRECISION_PATH = "wonef-precision.xml"

# Poids de base du score de preuve française, avant pondération
# par pouvoir discriminant (voir compute_fr_scores).
FR_BASE_OMW = 1.0     # lemme FR trouvé via omw-fr (fiable)
FR_BASE_WONEF = 0.15  # repli WoNeF : nettement moins fiable, ne doit
                       # servir que de très léger tie-break — un match
                       # WoNeF "unique" sur un sens rare et implausible
                       # (ex : scene.n.08 pour "view") ne doit pas
                       # dominer un gloss_score nettement plus élevé
                       # obtenu par un autre candidat.

# Facteur appliqué quand le lemme FR est déjà "réclamé" par un
# autre mot anglais de la phrase.
FR_CLAIM_DISCOUNT = 0.15

POS_TO_UPOS = {
    "n": "NOUN",
    "v": "VERB",
    "a": "ADJ",
    "r": "ADV",
}

# Fichier source par défaut d'où extraire le contexte élargi
# (répliques voisines) quand une entrée de TESTS ne précise pas
# de champ "source".
DEFAULT_SOURCE = "The Humans - Stephen Karam.txt"

# Nombre de répliques voisines (avant et après) incluses dans le
# contexte élargi soumis à GlossBERT.
CONTEXT_WINDOW = 2


# ============================================================
# EXEMPLES
# ============================================================

TESTS = [
    {
        "word": "diss",
        "pos": "v",
        "english":
            "You guys better not be dissing my home—"
            "do you even get how special a place like this is?",
        "french":
            "Vous n’avez pas le droit de vous moquer de mon chez-moi. "
            "Vous comprenez au moins à quel point un endroit comme "
            "celui-ci est exceptionnel ?",
        "expected": "diss.v.01",
    },

    {
        "word": "duplex",
        "pos": "n",
        "english":
            "No New Yorkers have duplex apartments.",
        "french":
            "Aucun New-Yorkais n’a d’appartement en duplex.",
        "expected": "duplex_apartment.n.01",
    },

    {
        "word": "standard",
        "pos": "a",
        "english":
            "No, that's standard for a ground-floor apartment.",
        "french":
            "Non, c’est tout à fait normal pour un appartement "
            "au rez-de-chaussée.",
        "expected": "standard.a.01",
    },

    {
        "word": "super",
        "pos": "n",
        "english":
            "Uh, must be the super, he's the only one who has access.",
        "french":
            "Ah, ça doit être le concierge, "
            "il est le seul à avoir la clé.",
        "expected": "superintendent.n.02",
    },

    {
        "word": "access",
        "pos": "n",
        "english":
            "He's the only one who has access.",
        "french":
            "Il est le seul à avoir la clé.",
        "expected": "access.n.02",
    },

    {
        "word": "view",
        "pos": "n",
        "english":
            "I wish you had more of a view.",
        "french":
            "J’aimerais que vous ayez une meilleure vue.",
        "expected": "view.n.02",
    },

    {
        "word": "alley",
        "pos": "n",
        "english":
            "It's an alley full of cigarette butts.",
        "french":
            "C’est une ruelle pleine de mégots de cigarettes.",
        "expected": "alley.n.01",
    },

    {
        "word": "butt",
        "pos": "n",
        "english":
            "It's an alley full of cigarette butts.",
        "french":
            "C’est une ruelle pleine de mégots de cigarettes.",
        "expected": "butt.n.09",
    },

    {
        "word": "courtyard",
        "pos": "n",
        "english":
            "It's an interior courtyard.",
        "french":
            "C’est une cour intérieure.",
        "expected": "court.n.10",
    },

    {
        "word": "score",
        "pos": "n",
        "english":
            "I wanna check the score of the game.",
        "french":
            "Je veux vérifier le score du match.",
        "expected": "score.n.03",
    },

    {
        "word": "plow",
        "pos": "v",
        "english":
            "The roads are all plowed—",
        "french":
            "Les routes ont toutes été déneigées—",
        "expected": "plow.v.01",
    },

    {
        "word": "key",
        "pos": "n",
        "english":
            "That is a terrible key for me.",
        "french":
            "C’est une tonalité épouvantable pour moi.",
        "expected": "key.n.04",
    },

    {
        "word": "gig",
        "pos": "n",
        "english":
            "You have any gigs lined up?",
        "french":
            "Tu as des concerts de prévus ?",
        "expected": "gig.n.06",
    },

    {
        "word": "shake",
        "pos": "n",
        "english":
            "Are her shakes in the fridge?",
        "french":
            "Ses milk-shakes sont-ils dans le frigo ?",
        "expected": "milkshake.n.01",
    },
]


# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

print("Chargement de GlossBERT...")

gloss_model = GlossBERT()

print("GlossBERT chargé.")
print()

EN = wn.Wordnet(EN_LEXICON)


# ============================================================
# WORDNET (NLTK) — sélection des synsets candidats
# ============================================================

def normalize_pos(pos):
    """
    WordNet distingue :
      a = adjective
      s = adjective satellite
    """

    if pos == "a":
        return {"a", "s"}

    return {pos}


def get_synsets(word, pos):

    wanted_pos = normalize_pos(
        pos
    )

    results = []

    for synset in nwn.synsets(word):

        if synset.pos() not in wanted_pos:
            continue

        matching = False

        for lemma in synset.lemmas():

            lemma_name = (
                lemma.name()
                .replace("_", " ")
                .casefold()
            )

            if lemma_name == word.casefold():
                matching = True
                break

        if matching:
            results.append(
                synset
            )

    return results


def get_synonyms(synset):

    return [
        lemma.name()
        .replace("_", " ")
        for lemma
        in synset.lemmas()
    ]


def synset_offset(synset):

    return f"{synset.offset():08d}"


# ============================================================
# LOCALISATION DU MOT CIBLE DANS LA PHRASE ANGLAISE
#
# GlossBERT a besoin des indices [start:end] de la forme de
# surface exacte dans la phrase. On essaie le mot tel quel,
# puis ses flexions (lemminflect), du plus long au plus court
# pour éviter qu'une flexion courte matche une sous-chaîne
# d'une flexion plus longue.
# ============================================================

def candidate_surface_forms(word, pos):

    upos = POS_TO_UPOS.get(pos)

    forms = {word}

    if upos:

        inflections = getAllInflections(
            word,
            upos=upos
        )

        if not inflections:

            inflections = (
                getAllInflectionsOOV(
                    word,
                    upos=upos
                )
            )

        for tags in inflections.values():

            forms.update(tags)

    return sorted(
        forms,
        key=len,
        reverse=True
    )


def locate_target_word(word, pos, text, restrict=None):
    """
    Cherche le mot cible dans `text`. Si `restrict` est fourni
    (start, end), la recherche se limite à cette tranche de
    caractères — utile quand `text` est un contexte élargi et
    que le mot cible apparaît aussi ailleurs dans une réplique
    voisine (ex : "view" hors de la phrase de "alley").
    """

    if restrict is not None:
        window_start, window_end = restrict
    else:
        window_start, window_end = 0, len(text)

    search_zone = text[window_start:window_end].casefold()

    for form in candidate_surface_forms(word, pos):

        pattern = (
            r"\b"
            + re.escape(form.casefold())
            + r"\b"
        )

        match = re.search(
            pattern,
            search_zone
        )

        if match:

            start = window_start + match.start()
            end = window_start + match.end()

            return (
                start,
                end,
                text[start:end],
            )

    return None


# ============================================================
# CONTEXTE ÉLARGI (répliques voisines dans le texte source)
#
# Une seule phrase ne suffit souvent pas à désambiguïser (cas
# "view" : sans la réplique suivante, GlossBERT choisit le
# mauvais sens). On retrouve la phrase dans le texte source et
# on l'entoure de CONTEXT_WINDOW répliques de chaque côté.
# ============================================================

_source_units_cache = {}


def load_source_units(path):
    """
    Découpe le fichier source en répliques/didascalies : une
    ligne non vide par unité, en écartant les lignes de nom de
    personnage (tout en majuscules, courtes). Les didascalies
    ("still staring out the window") sont conservées : elles
    portent souvent le contexte décisif.
    """

    if path in _source_units_cache:
        return _source_units_cache[path]

    try:
        raw_text = open(
            path,
            encoding="utf-8",
            errors="replace"
        ).read()
    except FileNotFoundError:
        _source_units_cache[path] = []
        return []

    units = []

    for line in raw_text.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        letters = [
            char
            for char in stripped
            if char.isalpha()
        ]

        is_character_name = (
            letters
            and all(char.isupper() for char in letters)
            and len(stripped) < 40
        )

        if is_character_name:
            continue

        units.append(stripped)

    _source_units_cache[path] = units

    return units


def normalize_words(text):

    cleaned = text.replace(
        "’", "'"
    ).replace(
        "‘", "'"
    )

    cleaned = re.sub(
        r"[^a-zA-Z0-9']",
        " ",
        cleaned
    )

    return [
        word
        for word in cleaned.casefold().split()
        if word
    ]


def build_token_stream(units):
    """
    Concatène toutes les unités en un seul flux de mots, en
    gardant pour chaque mot l'index de l'unité d'origine — la
    phrase cherchée peut être coupée sur plusieurs lignes par
    une mise en page à deux colonnes.
    """

    stream = []
    owner = []

    for index, unit in enumerate(units):

        for word in normalize_words(unit):

            stream.append(word)
            owner.append(index)

    return stream, owner


def find_sentence_span(units, sentence):
    """
    Retourne (première_unité, dernière_unité) contenant la
    phrase, ou None. Repli : plus long préfixe de la phrase
    (au moins 4 mots) si la phrase entière n'est pas retrouvée
    telle quelle (ex : coupée par une didascalie insérée).
    """

    stream, owner = build_token_stream(units)

    target = normalize_words(sentence)

    n = len(target)

    if n == 0:
        return None

    for start in range(len(stream) - n + 1):

        if stream[start:start + n] == target:
            return owner[start], owner[start + n - 1]

    for prefix_len in range(n, 3, -1):

        prefix = target[:prefix_len]

        for start in range(len(stream) - prefix_len + 1):

            if stream[start:start + prefix_len] == prefix:
                return owner[start], owner[start + prefix_len - 1]

    return None


def build_wide_context(source_path, sentence, window=CONTEXT_WINDOW):
    """
    Retourne {"text", "sentence_start", "sentence_end"} où
    "text" est la phrase entourée de `window` répliques de
    chaque côté, et "sentence_start"/"sentence_end" délimitent
    en caractères la phrase elle-même dans "text" (pas les
    répliques voisines) — pour restreindre la recherche du mot
    cible à la bonne occurrence. None si la phrase n'a pas été
    retrouvée dans le fichier source.
    """

    units = load_source_units(source_path)

    if not units:
        return None

    span = find_sentence_span(units, sentence)

    if span is None:
        return None

    first_unit, last_unit = span

    window_start = max(0, first_unit - window)
    window_end = min(len(units), last_unit + window + 1)

    context_text = ""
    sentence_start = None
    sentence_end = None

    for index in range(window_start, window_end):

        if context_text:
            context_text += " "

        char_start = len(context_text)
        context_text += units[index]
        char_end = len(context_text)

        if index == first_unit:
            sentence_start = char_start

        if index == last_unit:
            sentence_end = char_end

    return {
        "text": context_text,
        "sentence_start": sentence_start,
        "sentence_end": sentence_end,
    }


# ============================================================
# PREUVE FRANÇAISE (omw-fr, avec repli WoNeF)
# ============================================================

def strip_accents(text):

    decomposed = unicodedata.normalize(
        "NFD",
        text
    )

    return "".join(
        char
        for char in decomposed
        if unicodedata.category(char) != "Mn"
    )


def fr_stem(word):
    """
    Radical français très simple : casefold, sans accents,
    sans un éventuel -s/-x final (pluriel), pour aligner
    "mégots"/"mégot" ou "cigarettes"/"cigarette".
    """

    normalized = strip_accents(
        word.casefold()
    ).strip()

    if len(normalized) > 3 and normalized[-1] in ("s", "x"):
        normalized = normalized[:-1]

    return normalized


def fr_tokens(text):

    cleaned = text.replace(
        "’",
        "'"
    )

    return [
        token
        for token in re.split(
            r"[^a-zA-Zà-ÿ]+",
            cleaned
        )
        if len(token) > 2
    ]


def normalize_fr_phrase(text):
    """
    Normalisation "large" d'un texte français, pour la recherche
    de lemmes composés par sous-chaîne : accents retirés, casse
    repliée, traits d'union et espaces unifiés en un seul espace.
    Ex : "clef de voûte" et "milk-shake" deviennent comparables
    à une phrase source normalisée de la même façon.
    """

    cleaned = text.replace(
        "’", "'"
    ).replace(
        "‘", "'"
    )

    cleaned = strip_accents(
        cleaned.casefold()
    )

    cleaned = re.sub(
        r"[-]+",
        " ",
        cleaned
    )

    cleaned = re.sub(
        r"[^a-z0-9' ]",
        " ",
        cleaned
    )

    return re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip()


def stems_of(text):
    """
    Radicaux de chaque mot d'un texte (simple ou composé) — sert
    à généraliser l'escompte "déjà réclamé" aux lemmes composés :
    un match sur "clef de voûte" est escompté si l'un de ses mots
    est déjà expliqué par un autre mot anglais de la phrase.
    """

    return {
        fr_stem(word)
        for word in text.split()
        if len(word) > 2
    }


def fr_lemmas_for_synset(word, synset_id_offset, pos):
    """
    Traductions françaises des SEULS senses du synset qui
    correspondent exactement au mot anglais étudié (comme
    dans word_senses.py::get_french_lemmas_for_word).
    """

    translations = []

    for wn_synset in EN.synsets(word, pos=pos):

        offset = wn_synset.id.split("-")[-2]

        if offset != synset_id_offset:
            continue

        for sense in wn_synset.senses():

            sense_lemma = None

            try:
                sense_lemma = (
                    sense.word()
                    .lemma()
                    .replace("_", " ")
                    .casefold()
                )
            except Exception:
                continue

            if sense_lemma != word.casefold():
                continue

            try:
                french_senses = sense.translate(
                    lexicon=FR_LEXICON
                )
            except Exception:
                continue

            for french_sense in french_senses:

                try:
                    lemma = (
                        french_sense
                        .word()
                        .lemma()
                        .replace("_", " ")
                    )
                except Exception:
                    continue

                if lemma and lemma not in translations:
                    translations.append(lemma)

    return translations


def all_fr_lemmas_for_word(word):
    """
    Toutes les traductions françaises de TOUS les sens (tous
    POS) d'un mot anglais — utilisé pour savoir quels lemmes
    français sont déjà "expliqués" par un autre mot de la
    phrase que le mot cible.
    """

    translations = []

    for wn_synset in EN.synsets(word):

        for sense in wn_synset.senses():

            try:
                french_senses = sense.translate(
                    lexicon=FR_LEXICON
                )
            except Exception:
                continue

            for french_sense in french_senses:

                try:
                    lemma = (
                        french_sense
                        .word()
                        .lemma()
                        .replace("_", " ")
                    )
                except Exception:
                    continue

                if lemma:
                    translations.append(lemma)

    return translations


def content_words(english, exclude):

    exclude_key = exclude.casefold()

    words = set()

    for token in re.findall(
        r"[A-Za-z']+",
        english
    ):

        if len(token) <= 2:
            continue

        if token.casefold() == exclude_key:
            continue

        words.add(
            token.casefold()
        )

    return words


def claimed_fr_stems(word, english):
    """
    Radicaux français déjà "réclamés", donc escomptés au lieu de
    trancher plein pot :

    - par les autres mots anglais de la phrase (ex : "cigarette"
      explique le radical "cigarett" indépendamment du sens de
      "butt") ;
    - par le mot cible lui-même, s'il existe en français sous une
      forme identique (emprunt, ex : "score", "duplex"). Toute la
      valeur du signal français vient de ce que le français
      découpe les sens autrement que l'anglais ; quand le mot
      français EST le mot anglais, cette valeur est nulle par
      construction — et c'est aussi là que l'alignement
      automatique de omw-fr est le moins fiable (vérifié sur
      "score" : "score" est attribué comme traduction FR de
      mark.n.01, pas de score.n.03, qui est le bon sens sportif).
    """

    claimed = {
        fr_stem(word)
    }

    for other_word in content_words(english, word):

        for lemma in all_fr_lemmas_for_word(other_word):

            claimed.add(
                fr_stem(lemma)
            )

    return claimed


_wonef_cache = None


def load_wonef_precision():

    global _wonef_cache

    if _wonef_cache is not None:
        return _wonef_cache

    import xml.etree.ElementTree as ET

    wonef_by_id = {}

    try:
        tree = ET.parse(WONEF_PRECISION_PATH)
    except (FileNotFoundError, ET.ParseError):
        _wonef_cache = {}
        return _wonef_cache

    for synset_element in tree.getroot().iter():

        tag = (
            synset_element.tag
            .split("}")[-1]
            .upper()
        )

        if tag != "SYNSET":
            continue

        synset_id = None
        literals = []

        for child in synset_element:

            child_tag = (
                child.tag
                .split("}")[-1]
                .upper()
            )

            if child_tag == "ID" and child.text:
                synset_id = child.text.strip()

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

                    literal = literal_element.text.strip()

                    if literal and literal != "_EMPTY_":
                        literals.append(literal)

        if synset_id:
            wonef_by_id[synset_id] = literals

    _wonef_cache = wonef_by_id

    return _wonef_cache


def wonef_lemmas_for_synset(word, offset, pos):

    wonef_by_id = load_wonef_precision()

    wonef_id = f"eng-30-{offset}-{pos}"

    literals = wonef_by_id.get(
        wonef_id,
        []
    )

    word_stem = fr_stem(word)

    cleaned = []

    for literal in literals:

        # Filtrer les artefacts d'alignement WoNeF (ex :
        # "butte" collé sur tous les sens de "butt") : on
        # rejette tout candidat trop proche du mot anglais
        # lui-même.
        if fr_stem(literal) == word_stem:
            continue

        cleaned.append(literal)

    return cleaned


def fr_lemma_match_key(lemma, fr_word_stems, normalized_french):
    """
    Teste si un lemme FR (simple ou composé) matche la phrase
    française, et retourne une "clé" identifiant ce match pour le
    calcul du pouvoir discriminant :

    - lemme simple ("mégot") : comparaison de radical, comme
      avant — la clé est le radical lui-même.
    - lemme composé ("milk-shake", "clef de voûte") : `fr_tokens`
      le découpe en morceaux qui ne peuvent jamais matcher un
      seul token de la phrase, donc on compare à la place par
      sous-chaîne sur la phrase entière normalisée (accents et
      séparateurs -/espace unifiés) — la clé est le lemme
      composé normalisé.

    None si aucun match.
    """

    if " " in lemma or "-" in lemma:

        normalized_lemma = normalize_fr_phrase(lemma)

        if normalized_lemma and normalized_lemma in normalized_french:
            return normalized_lemma

        return None

    stem = fr_stem(lemma)

    return stem if stem in fr_word_stems else None


def compute_fr_scores(word, pos, synsets, french, claimed):
    """
    Score de preuve française pour chaque synset candidat,
    pondéré par pouvoir discriminant : un lemme FR partagé par
    beaucoup de synsets candidats (ex : "vue" pour 7 des 10 sens
    de "view") ne doit quasiment rien apporter, alors qu'un
    lemme exclusif à un seul synset (ex : "mégot" pour
    butt.n.09) doit trancher.

        informativite(clé) = 1 / (nb de synsets candidats
                                   partageant cette clé, pour
                                   cette même source omw/wonef)
        fr_score(synset)   = base_source
                              × max sur ses clés appariées de
                                informativite(clé) × facteur_reclame

    "facteur_reclame" vaut FR_CLAIM_DISCOUNT dès qu'un des mots de
    la clé est déjà "réclamé" — par un autre mot anglais de la
    phrase, ou par le mot cible lui-même s'il s'agit d'un emprunt
    identique (voir claimed_fr_stems).

    Retourne {synset.name(): {...}}.
    """

    fr_word_stems = {
        fr_stem(token)
        for token in fr_tokens(french)
    }

    normalized_french = normalize_fr_phrase(french)

    # --------------------------------------------------------
    # 1) Lemmes FR + clés appariées par synset, par source.
    # --------------------------------------------------------

    per_synset = {}

    for synset in synsets:

        offset = synset_offset(synset)

        omw_lemmas = fr_lemmas_for_synset(
            word,
            offset,
            pos
        )

        omw_hit_keys = {
            key
            for lemma in omw_lemmas
            if (
                key := fr_lemma_match_key(
                    lemma,
                    fr_word_stems,
                    normalized_french
                )
            ) is not None
        }

        if omw_hit_keys:

            per_synset[synset.name()] = {
                "source": "omw-fr",
                "lemmas": omw_lemmas,
                "hit_keys": omw_hit_keys,
            }

            continue

        # Repli WoNeF si omw-fr ne couvre pas ce synset.
        wonef_lemmas = wonef_lemmas_for_synset(
            word,
            offset,
            pos
        )

        wonef_hit_keys = {
            key
            for lemma in wonef_lemmas
            if (
                key := fr_lemma_match_key(
                    lemma,
                    fr_word_stems,
                    normalized_french
                )
            ) is not None
        }

        per_synset[synset.name()] = {
            "source": "wonef" if wonef_hit_keys else None,
            "lemmas": wonef_lemmas if wonef_hit_keys else omw_lemmas,
            "hit_keys": wonef_hit_keys,
        }

    # --------------------------------------------------------
    # 2) Pouvoir discriminant de chaque clé, par source :
    #    combien de synsets candidats la partagent ?
    # --------------------------------------------------------

    key_counts = {"omw-fr": {}, "wonef": {}}

    for entry in per_synset.values():

        source = entry["source"]

        if source is None:
            continue

        for key in entry["hit_keys"]:

            key_counts[source][key] = (
                key_counts[source].get(key, 0) + 1
            )

    base_weight = {
        "omw-fr": FR_BASE_OMW,
        "wonef": FR_BASE_WONEF,
    }

    # --------------------------------------------------------
    # 3) Score final par synset.
    # --------------------------------------------------------

    results = {}

    for name, entry in per_synset.items():

        source = entry["source"]
        hit_keys = entry["hit_keys"]

        if source is None or not hit_keys:

            results[name] = {
                "score": 0.0,
                "source": None,
                "lemmas": entry["lemmas"],
                "hits": [],
                "best_key": None,
                "best_count": None,
            }

            continue

        best_score = 0.0
        best_key = None

        for key in hit_keys:

            count = key_counts[source][key]

            informativeness = 1.0 / count

            claim_factor = (
                FR_CLAIM_DISCOUNT
                if stems_of(key) & claimed
                else 1.0
            )

            score = (
                base_weight[source]
                * informativeness
                * claim_factor
            )

            if score > best_score or best_key is None:
                best_score = score
                best_key = key

        hits = [
            lemma
            for lemma in entry["lemmas"]
            if fr_lemma_match_key(
                lemma,
                fr_word_stems,
                normalized_french
            ) in hit_keys
        ]

        results[name] = {
            "score": best_score,
            "source": source,
            "lemmas": entry["lemmas"],
            "hits": hits,
            "best_key": best_key,
            "best_count": key_counts[source][best_key],
        }

    return results


# ============================================================
# GLOSSBERT (WSD anglais contextuel)
# ============================================================

def glossbert_scores(word, pos, context_text, restrict, synsets):
    """
    Retourne {synset.name(): score} à partir de GlossBERT,
    restreint aux synsets candidats. Score 0 si GlossBERT ne
    renvoie rien pour un synset (ex : mot cible introuvable
    dans le contexte). `restrict` limite la recherche du mot
    cible à l'empan de la phrase elle-même, pas des répliques
    voisines ajoutées au contexte.
    """

    located = locate_target_word(
        word,
        pos,
        context_text,
        restrict
    )

    if located is None:
        return {}, None

    start, end, surface = located

    raw_results = gloss_model(
        context_text,
        start,
        end,
        word,
    )

    wanted_names = {
        synset.name()
        for synset in synsets
    }

    scores = {
        synset.name(): float(score)
        for score, synset in raw_results
        if synset.name() in wanted_names
    }

    return scores, surface


# ============================================================
# ANALYSE D'UN TEST
# ============================================================

def analyze(test):

    word = test["word"]
    pos = test["pos"]

    english = (
        test["english"]
    )

    french = (
        test["french"]
    )

    source_path = test.get(
        "source",
        DEFAULT_SOURCE
    )


    print()
    print("=" * 110)

    print(
        f"{word.upper()} — POS={pos}"
    )

    print("=" * 110)

    print(
        f"EN : {english}"
    )

    print(
        f"FR : {french}"
    )

    print()


    # --------------------------------------------------------
    # WordNet
    # --------------------------------------------------------

    synsets = get_synsets(
        word,
        pos
    )

    if not synsets:

        print(
            "Aucun sens WordNet "
            "pour ce mot/POS."
        )

        return None


    # --------------------------------------------------------
    # Contexte élargi (répliques voisines)
    # --------------------------------------------------------

    wide_context = build_wide_context(
        source_path,
        english
    )

    if wide_context is None:

        print(
            "!!! Contexte élargi introuvable dans "
            f'"{source_path}" '
            "(repli sur la phrase seule) !!!"
        )

        context_text = english
        restrict = None

    else:

        print(
            "Contexte élargi trouvé "
            f"(±{CONTEXT_WINDOW} répliques) : "
            f'"{wide_context["text"]}"'
        )

        context_text = wide_context["text"]

        restrict = (
            wide_context["sentence_start"],
            wide_context["sentence_end"],
        )


    # --------------------------------------------------------
    # GlossBERT
    # --------------------------------------------------------

    gloss_scores, surface = glossbert_scores(
        word,
        pos,
        context_text,
        restrict,
        synsets
    )

    if surface is None:

        print(
            "!!! Mot cible introuvable dans le contexte "
            "(GlossBERT ignoré, seule la preuve FR décide) !!!"
        )

    print()


    # --------------------------------------------------------
    # Preuve française
    # --------------------------------------------------------

    claimed = claimed_fr_stems(
        word,
        english
    )

    fr_scores = compute_fr_scores(
        word,
        pos,
        synsets,
        french,
        claimed
    )

    results = []

    for synset in synsets:

        fr_entry = fr_scores[synset.name()]

        gloss_score = gloss_scores.get(
            synset.name(),
            0.0
        )

        results.append(
            {
                "synset": synset,
                "gloss_score": gloss_score,
                "fr_score": fr_entry["score"],
                "fr_source": fr_entry["source"],
                "fr_lemmas": fr_entry["lemmas"],
                "fr_hits": fr_entry["hits"],
                "fr_best_stem": fr_entry["best_key"],
                "fr_best_count": fr_entry["best_count"],
                "final_score": gloss_score + fr_entry["score"],
            }
        )

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )


    print(
        "=== CLASSEMENT (GlossBERT + preuve FR) ==="
    )

    print()

    for rank, item in enumerate(
        results,
        start=1
    ):

        synset = item["synset"]

        print(
            f"{rank}. {synset.name()}"
        )

        fr_detail = ""

        if item["fr_source"]:

            fr_detail = (
                f" via {item['fr_source']}"
                f", {item['fr_best_stem']}"
                f" partagé par {item['fr_best_count']} sens"
            )

        print(
            f"   Score final : {item['final_score']:.3f}"
            f"  (gloss={item['gloss_score']:.3f}"
            f", fr={item['fr_score']:.2f}"
            f"{fr_detail})"
        )

        print(
            f"   Definition  : {synset.definition()}"
        )

        if item["fr_lemmas"]:

            print(
                "   FR (synset) : "
                + ", ".join(item["fr_lemmas"])
                + (
                    f"  [match: {', '.join(item['fr_hits'])}]"
                    if item["fr_hits"]
                    else ""
                )
            )

        print(
            "   Synonymes   : "
            + ", ".join(
                get_synonyms(synset)
            )
        )

        print()


    # --------------------------------------------------------
    # Meilleur
    # --------------------------------------------------------

    best = results[0]

    print(
        ">>> MEILLEUR SENS"
    )

    print(
        best["synset"].name()
    )

    print(
        best["synset"]
        .definition()
    )

    print(
        "Synonymes : "
        + ", ".join(
            get_synonyms(
                best["synset"]
            )
        )
    )

    print(
        f"Score final : {best['final_score']:.3f}"
    )


    # --------------------------------------------------------
    # Marge
    # --------------------------------------------------------

    if len(results) >= 2:

        margin = (
            results[0]["final_score"]
            -
            results[1]["final_score"]
        )

        print(
            f"Marge sur le 2e : "
            f"{margin:.3f}"
        )

        if margin <= 0:

            print(
                "!!! Marge nulle ou négative : "
                "classement ambigu, à vérifier manuellement !!!"
            )

    return best["synset"].name()


# ============================================================
# PROGRAMME
#
# Sans argument : lance tous les cas de TESTS.
# Avec des arguments : ne lance que les cas dont "word" est dans
# la liste (comparaison insensible à la casse), ex :
#
#   python sense_in_context.py view
#   python sense_in_context.py view butt
# ============================================================

if len(sys.argv) > 1:

    wanted_words = {
        arg.casefold()
        for arg in sys.argv[1:]
    }

    selected_tests = [
        test
        for test in TESTS
        if test["word"].casefold() in wanted_words
    ]

    unknown = wanted_words - {
        test["word"].casefold()
        for test in TESTS
    }

    if unknown:

        print(
            "Mots inconnus dans TESTS (ignorés) : "
            + ", ".join(sorted(unknown))
        )

else:

    selected_tests = TESTS


recap = []

for test in selected_tests:

    obtained = analyze(
        test
    )

    recap.append(
        (test, obtained)
    )


# ------------------------------------------------------------
# Récapitulatif OK/ÉCHEC — n'a de sens que par rapport au sens
# attendu ("expected") ; les cas sans "expected" sont juste
# listés sans verdict.
# ------------------------------------------------------------

print()
print("=" * 110)
print("RÉCAPITULATIF")
print("=" * 110)

if len(selected_tests) < len(TESTS):

    print(
        f"(sous-ensemble filtré : {len(selected_tests)}/{len(TESTS)} cas)"
    )

failures = 0
checked = 0

for test, obtained in recap:

    expected = test.get("expected")

    if expected is None:

        print(
            f"?    {test['word']:12s} obtenu={obtained}"
            "  (pas de sens attendu déclaré)"
        )

        continue

    checked += 1

    ok = obtained == expected

    if not ok:
        failures += 1

    print(
        f"{'OK  ' if ok else 'ÉCHEC'} {test['word']:12s} "
        f"attendu={expected}  obtenu={obtained}"
    )

print()

print(
    f"{checked - failures}/{checked} OK"
    + (
        f"  ({len(recap) - checked} sans sens attendu)"
        if len(recap) > checked
        else ""
    )
)
