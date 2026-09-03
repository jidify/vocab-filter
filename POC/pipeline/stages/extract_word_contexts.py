"""POC — extrait, pour les MOTS SIMPLES uniquement, les lemmes qui passent le
filtrage documenté dans REVIEW_FIX_PIPELINE/RAPPORT/rapport_filtrage.md, puis
pour chaque lemme retenu la liste des phrases du livre qui le contiennent
(sous une de ses formes fléchies) et leur nombre.

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production (même statut que REVIEW_FIX_PIPELINE/filter_tests/
filter_book_vocab.py, dont il reprend les seuils et le chargement des
ressources afin de rester cohérent avec le rapport).

Chaîne de filtres reproduite (ordre et seuils identiques au rapport), précédée
d'un filtre 0 hors rapport (voir plus bas) :
  1. Pknown (DATASETS/word-prevalence.txt)      : absent -> exclu ; garder si Pknown > 0.90
  2. CEFR (DATASETS/cefrj.csv), jointure par POS : exclu si niveaux connus ⊆ {A1, A2}
  3. Repêchage Zipf (wordfreq)                   : seulement pour les exclus du filtre 2,
                                                    réintégrés si zipf_frequency < 4.5

Filtres supplémentaires (hors rapport, ajoutés pour ce script) :
  0. Exclusion par UPOS, avant même le filtre 1 : PROPN (nom propre), INTJ
     (interjection) et PUNCT (ponctuation) sont écartés dès la lecture des
     tokens — ni comptés dans l'entonnoir (tokens_alpha et au-delà), ni
     jamais candidats à la chaîne 1-5. PUNCT est de toute façon déjà
     exclu par `token.is_alpha` ; PROPN/INTJ ne l'étaient pas.
  0bis. Cognats/faux-amis EN-FR, juste après le filtre 0 (donc avant même
     Pknown) :
       - COGNAT (DATASETS/cognet_en_fr.csv, sous-ensemble EN-FR filtré de
         CogNet v0 — Batsuren/Bella/Giunchiglia, ACL 2019, CC BY-NC-SA 4.0,
         https://github.com/kbatsuren/CogNet, paire (lemme EN, mot FR)
         partageant un synset WordNet ET jugée cognat, ~94% précision
         mesurée par les auteurs) : si le lemme y figure, il est retiré
         IMMÉDIATEMENT, avant tout le reste de la chaîne (jamais dans
         seen_types/by_lemma) — jamais dans la liste finale. Chaque
         suppression est journalisée dans un CSV séparé (voir
         write_cognates_removed_csv), avec le mot FR cognat et le
         concept_id WordNet correspondants.
       - FAUX-AMI (false_friends_en_fr_seed.csv, à côté de ce script —
         liste seed constituée à la main, ~35 paires classiques
         actually/actuellement, library/librairie, etc. — PAS un dataset
         académique : aucune source téléchargeable trouvée pour le
         lexique de faux-amis Uban & Dinu, LREC 2020, malgré recherche ;
         à étendre/remplacer si une meilleure source est trouvée) : si le
         lemme y figure, il est protégé de TOUTE exclusion ultérieure —
         filtres 1-3 (passes_filter court-circuité, considéré comme
         passant toujours), filtre 4 (A1/A2) et filtre 5 (MWE) — et
         apparaît donc toujours dans la liste finale, quel que soit son
         Pknown/CEFR/Zipf ou son appartenance à une MWE détectée.
  4. Exclusion A1/A2 stricte, tous POS confondus : si, pour un lemme, au
     moins un des niveaux CEFR connus dans cefrj.csv (n'importe quel POS,
     pas seulement celui matché par le filtre 2) est A1 ou A2, le lemme
     est retiré de la liste finale — même s'il a par ailleurs des niveaux
     supérieurs (B1+) ou des entrées sans niveau pour d'autres POS. Cette
     règle est appliquée en dernier, après le filtre 3 : un lemme repêché
     par le filtre 3 avait par construction au moins un niveau A1/A2 (le
     filtre 2 ne l'a exclu que parce que TOUS ses niveaux matchés
     l'étaient), donc le filtre 4 l'exclut à nouveau — le repêchage Zipf
     ne produit plus de survivants avec ce filtre actif, mais le code du
     filtre 3 est conservé tel quel (issu du rapport) à titre de
     diagnostic dans l'entonnoir affiché en fin d'exécution.
  5. Exclusion MWE : pour chaque lemme retenu, chacune de ses phrases est
     repassée au détecteur MWE DE PROD (pipeline/mwe.py::get_matcher,
     l'objet idiomatch chargé par S2 — mêmes portes de validation S2 que
     dans mwe_gates.classify, même alignement des membres que dans
     mwe_alignment.align_members). Dès qu'une occurrence du lemme dans
     une de ses phrases tombe dans le span d'une MWE acceptée par ces
     portes, le lemme entier est retiré de la liste finale — pas
     seulement cette phrase. Ne couvre que la source idiomatch : VPC et
     rules_plus (les deux autres sources de S2, voir pipeline/mwe.py)
     lisent des artefacts précalculés par analyze.py sur un corpus
     entier déjà passé par S1, hors de portée d'un appel phrase par
     phrase isolé comme ici.

Contexte de phrase courte (hors chaîne de filtrage ci-dessus) : si la phrase
contenant l'occurrence du lemme mesure moins de MIN_SENTENCE_WORDS mots, la
phrase précédente et la phrase suivante de la même ligne du livre (quand
elles existent) sont incluses avec elle dans le texte exporté (colonne
`phrases`), pour donner assez de matière au lecteur. N'affecte que le texte
de contexte affiché, pas la détection MWE (filtre 5, rejouée sur ce texte
étendu) ni aucun autre filtre.

Pas de filtre AoA (le rapport le documente comme informatif, sans seuil
tranché) et pas de wordlist.txt (absent de cette chaîne) — mais AoA
(DATASETS/kuperman-aoa.csv, colonne Rating.Mean) est exporté à titre
informatif dans les CSV de sortie (colonne `aoa`), comme `cefr` et `zipf`
(wordfreq.zipf_frequency) et `false_friend` (appartenance à
false_friends_en_fr_seed.csv) — aucune de ces 4 colonnes ne filtre quoi que
ce soit, elles ne font qu'annoter les lignes déjà décidées par la chaîne
ci-dessus.

Usage :
    uv run python POC/traitement_word/claude/extract_word_contexts.py
    uv run python POC/traitement_word/claude/extract_word_contexts.py --book "books/Dark Matter - Blake Crouch.txt"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import spacy
from wordfreq import zipf_frequency

# POC/traitement_word/claude/extract_word_contexts.py -> POC/ est le parent(2).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_pipeline.corpus import is_hors_oeuvre  # noqa: E402
from poc_pipeline import mwe as mwe_module  # noqa: E402
from poc_pipeline import mwe_alignment, mwe_gates  # noqa: E402
from poc_pipeline.tokenizer_boundary_fix import (  # noqa: E402
    patch_dash_after_punctuation,
)

DEFAULT_BOOK_PATH = ROOT / "books" / "The Humans - Stephen Karam.txt"
DEFAULT_OUT_PATH = Path(__file__).parent / "word_contexts.csv"
DEFAULT_MWE_EXCLUSIONS_OUT_PATH = Path(__file__).parent / "mwe_exclusions.csv"
DEFAULT_COGNATES_REMOVED_OUT_PATH = Path(__file__).parent / "cognates_removed.csv"
DEFAULT_PKNOWN_CEFR_EXCLUDED_OUT_PATH = Path(__file__).parent / "pknown_cefr_excluded.csv"
DEFAULT_BASIC_LEVEL_EXCLUDED_OUT_PATH = Path(__file__).parent / "basic_level_excluded.csv"

AOA_PATH = ROOT / "poc_datasets" / "kuperman-aoa.csv"  # non utilisé (rapport : AoA informatif seulement)
PREVALENCE_PATH = ROOT / "poc_datasets" / "word-prevalence.txt"
CEFR_PATH = ROOT / "poc_datasets" / "cefrj.csv"
COGNET_PATH = ROOT / "poc_datasets" / "cognet_en_fr.csv"
FALSE_FRIENDS_PATH = Path(__file__).parent / "false_friends_en_fr_seed.csv"

SPACY_MODEL = "en_core_web_sm"

# --------------------------------------------------------------------------
# Seuils du filtrage — identiques à REVIEW_FIX_PIPELINE/RAPPORT/rapport_filtrage.md
# et à filter_tests/filter_book_vocab.py (section "Pour reproduire / ajuster").
# --------------------------------------------------------------------------

MIN_PKNOWN = 0.90
EXCLUDED_CEFR = {"A1", "A2"}
ZIPF_RESCUE_THRESHOLD = 4.5

# Phrase courte (hors chaîne de filtrage ci-dessus) : si la phrase contenant
# le lemme mesure moins de ce nombre de mots, la phrase précédente et la
# phrase suivante (dans la même ligne du livre) sont incluses avec elle dans
# le contexte exporté, pour donner assez de matière au lecteur.
MIN_SENTENCE_WORDS = 5

# Filtre 0 (hors rapport) : UPOS écartés avant toute autre étape.
EXCLUDED_UPOS = {"PROPN", "INTJ", "PUNCT"}

# UPOS spaCy -> catégories POS de cefrj.csv (en toutes lettres), par ordre de
# préférence — copié tel quel de filter_book_vocab.py.
UPOS_TO_CEFR_POS = {
    "NOUN": ["noun"],
    "PROPN": ["noun"],
    "VERB": ["verb"],
    "AUX": ["be-verb", "have-verb", "do-verb", "modal auxiliary", "verb"],
    "ADJ": ["adjective"],
    "ADV": ["adverb"],
    "DET": ["determiner"],
    "PRON": ["pronoun"],
    "ADP": ["preposition"],
    "CCONJ": ["conjunction"],
    "SCONJ": ["conjunction"],
    "NUM": ["number"],
    "INTJ": ["interjection"],
    "PART": ["infinitive-to"],
}


# --------------------------------------------------------------------------
# Chargement des ressources (copié de filter_book_vocab.py)
# --------------------------------------------------------------------------

def load_pknown_scores() -> dict[str, float]:
    """word-prevalence.txt : word,Pknown,Nobs,Prevalence,FreqZipfUS, sans en-tête."""
    scores: dict[str, float] = {}
    with PREVALENCE_PATH.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 5:
                continue
            word = parts[0].strip().casefold()
            try:
                pknown = float(parts[1])
            except ValueError:
                continue
            scores[word] = pknown
    return scores


def load_cefr_by_word() -> dict[str, dict[str, set[str]]]:
    """cefrj.csv : headword,pos,CEFR,... — headword peut porter plusieurs
    variantes séparées par '/' (ex. "a.m./A.M./am/AM"), chacune indexée à part."""
    cefr_by_word: dict[str, dict[str, set[str]]] = {}
    with CEFR_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos_label = row["pos"].strip().casefold()
            level = row["CEFR"].strip().upper()
            for variant in row["headword"].split("/"):
                headword = variant.strip().casefold()
                if not headword:
                    continue
                cefr_by_word.setdefault(headword, {}).setdefault(pos_label, set()).add(level)
    return cefr_by_word


def load_cognates() -> dict[str, dict]:
    """DATASETS/cognet_en_fr.csv : word_en,word_fr,concept_id — sous-ensemble
    EN-FR de CogNet v0 (filtré depuis CogNet-v0.tsv, voir docstring du
    module). Plusieurs entrées possibles par lemme EN (concepts distincts) :
    on garde la première rencontrée, l'usage ici est binaire (cognat oui/non)
    donc le choix entre plusieurs mots FR cognats n'affecte rien d'autre que
    le rapport écrit par write_cognates_removed_csv."""
    cognates: dict[str, dict] = {}
    with COGNET_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lemma = row["word_en"].strip().casefold()
            cognates.setdefault(lemma, {
                "word_fr": row["word_fr"], "concept_id": row["concept_id"],
            })
    return cognates


def load_false_friends() -> set[str]:
    """false_friends_en_fr_seed.csv : liste seed écrite à la main (voir
    docstring du module) — aucun dataset académique téléchargeable trouvé
    pour ce couple de langues au moment de l'écriture de ce script."""
    lemmas: set[str] = set()
    with FALSE_FRIENDS_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lemmas.add(row["word_en"].strip().casefold())
    return lemmas


def load_aoa_scores() -> dict[str, float]:
    """kuperman-aoa.csv : Word,...,Rating.Mean,... — âge d'acquisition,
    colonne informative uniquement (voir docstring du module), même
    chargement que REVIEW_FIX_PIPELINE/filter_tests/filter_book_vocab.py."""
    scores: dict[str, float] = {}
    with AOA_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            word = row["Word"].strip().casefold()
            rating = row["Rating.Mean"].strip()
            if not rating:
                continue
            try:
                scores[word] = float(rating)
            except ValueError:
                continue
    return scores


def cefr_levels_for(cefr_by_word: dict, lemma: str, upos: str) -> set[str]:
    """Niveaux CEFR pour ce lemme : d'abord le(s) POS cefrj.csv correspondant
    au tag spaCy (UPOS_TO_CEFR_POS), puis repli sur l'union de tous les POS
    connus pour ce mot si aucun ne matche."""
    pos_data = cefr_by_word.get(lemma)
    if not pos_data:
        return set()
    for candidate in UPOS_TO_CEFR_POS.get(upos, []):
        if candidate in pos_data:
            return pos_data[candidate]
    union: set[str] = set()
    for levels in pos_data.values():
        union |= levels
    return union


def all_cefr_levels_for_lemma(cefr_by_word: dict, lemma: str) -> set[str]:
    """Union de tous les niveaux CEFR connus pour ce lemme, tous POS
    cefrj.csv confondus — colonne informative `cefr` des CSV de sortie,
    même périmètre que lemma_has_basic_level mais sans réduire à un booléen."""
    pos_data = cefr_by_word.get(lemma)
    if not pos_data:
        return set()
    union: set[str] = set()
    for levels in pos_data.values():
        union |= levels
    return union


def lemma_has_basic_level(
    cefr_by_word: dict, lemma: str, cache: dict[str, bool],
) -> bool:
    """Filtre supplémentaire 4 : True si, tous POS confondus dans cefrj.csv,
    au moins un niveau connu pour ce lemme est A1 ou A2 — un niveau
    supérieur ou une entrée sans niveau pour un autre POS ne rachète pas
    le lemme."""
    cached = cache.get(lemma)
    if cached is not None:
        return cached
    result = bool(all_cefr_levels_for_lemma(cefr_by_word, lemma) & EXCLUDED_CEFR)
    cache[lemma] = result
    return result


# --------------------------------------------------------------------------
# Filtre supplémentaire 5 : exclusion des lemmes pris dans une MWE, via le
# détecteur idiomatch de prod (pipeline/mwe.py) rejoué phrase par phrase.
# --------------------------------------------------------------------------

def mwe_member_spans(
    phrase_text: str, matcher, cache: dict[str, list],
) -> list[tuple[int, int, str]]:
    """Spans caractère (bornes dans `phrase_text`, plus l'idiome associé) de
    tous les membres MWE acceptés (mêmes portes S2 que
    pipeline/mwe.py::find_candidates : n_tokens_span >= 2,
    mwe_gates.classify) détectés par le matcher de prod dans cette phrase.
    Un alignement ambigu (mwe_alignment.align_members) n'a pas de membres
    sûrs : on retient alors l'enveloppe entière du match par prudence
    (mieux vaut exclure un lemme à tort que le garder à tort dans une MWE
    incertaine)."""
    cached = cache.get(phrase_text)
    if cached is not None:
        return cached

    spans: list[tuple[int, int, str]] = []
    doc = matcher.nlp(phrase_text)
    for m in matcher(doc):
        _match_id, start, end = m["meta"]
        if end - start < 2:
            continue
        span = doc[start:end]
        if mwe_gates.classify(m["idiom"], list(span), matcher.nlp, matcher.n) is not None:
            continue
        alignment = mwe_alignment.align_members(m["idiom"], list(span), matcher.nlp, matcher.n)
        if alignment.ambiguous:
            spans.append((span.start_char, span.end_char, m["idiom"]))
            continue
        for i in sorted(alignment.member_indices):
            tok = span[i]
            spans.append((tok.idx, tok.idx + len(tok.text), m["idiom"]))

    cache[phrase_text] = spans
    return spans


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def lemma_mwe_idioms(entry: dict, matcher, cache: dict[str, list]) -> set[str]:
    """Idiomes dont une occurrence a chevauché au moins un `token_span` de
    ce lemme dans l'une de ses phrases — ensemble vide si le lemme n'est
    jamais pris dans une MWE. Un lemme avec un résultat non vide est exclu
    en entier (pas seulement les phrases concernées), voir le filtre 5 dans
    le docstring du module."""
    idioms: set[str] = set()
    for phrase in entry["phrases"].values():
        mwe_spans = mwe_member_spans(phrase["text"], matcher, cache)
        if not mwe_spans:
            continue
        for token_span in phrase["token_spans"]:
            for start, end, idiom in mwe_spans:
                if _spans_overlap(token_span, (start, end)):
                    idioms.add(idiom)
    return idioms


# --------------------------------------------------------------------------
# Filtrage — chaîne du rapport, décision mémoïsée par (lemma, upos)
# --------------------------------------------------------------------------

def passes_filter(
    lemma: str, upos: str, pknown_scores: dict, cefr_by_word: dict,
    cache: dict[tuple[str, str], bool],
) -> bool:
    key = (lemma, upos)
    cached = cache.get(key)
    if cached is not None:
        return cached

    pknown = pknown_scores.get(lemma)
    if pknown is None or not (pknown > MIN_PKNOWN):
        cache[key] = False
        return False

    levels = cefr_levels_for(cefr_by_word, lemma, upos)
    is_basic_only = bool(levels) and levels.issubset(EXCLUDED_CEFR)
    if is_basic_only:
        zipf = zipf_frequency(lemma, "en")
        if not (zipf < ZIPF_RESCUE_THRESHOLD):
            cache[key] = False
            return False

    cache[key] = True
    return True


# --------------------------------------------------------------------------
# Extraction + agrégation
# --------------------------------------------------------------------------

def iter_book_lines(book_path: Path, skip_lines: int = 0):
    """Lignes non vides du livre, hors-œuvre exclu (mêmes règles que
    poc_pipeline/corpus.py::is_hors_oeuvre, réutilisé tel quel), plus les
    `skip_lines` premières lignes de tête (1-indexées, lignes vides
    comprises) — même sémantique que corpus.py::load_segments(skip_lines=),
    pour que ce script voie exactement le même texte que
    extract_mwe_contexts.py sur le même livre."""
    raw_text = book_path.read_text(encoding="utf-8", errors="replace")
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if line_no <= skip_lines:
            continue
        stripped = line.strip()
        if not stripped or is_hors_oeuvre(stripped):
            continue
        yield stripped


def build_word_contexts(
    book_path: Path, skip_lines: int = 0,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], dict[tuple[str, str], dict],
           dict[str, dict], dict[str, int]]:
    pknown_scores = load_pknown_scores()
    cefr_by_word = load_cefr_by_word()
    cognates = load_cognates()
    false_friends = load_false_friends()
    filter_cache: dict[tuple[str, str], bool] = {}
    lemma_basic_cache: dict[str, bool] = {}

    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    patch_dash_after_punctuation(nlp)

    stats = {
        "tokens_alpha": 0,
        "types_seen": 0,
        "types_after_pknown": 0,
        "types_after_cefr": 0,
        "types_rescued_zipf": 0,
        "lemmas_excluded_basic_level_any_pos": 0,
        "lemmas_excluded_mwe_member": 0,
        "lemmas_removed_cognate": 0,
        "false_friend_lemmas_seen": 0,
    }
    seen_types: set[tuple[str, str]] = set()
    passed_types: set[tuple[str, str]] = set()
    rescued_types: set[tuple[str, str]] = set()
    # (lemma, upos) -> formes de surface rencontrées, quel que soit le sort
    # du type (retenu ou non) — utilisé uniquement par l'audit du filtre 1-3
    # ci-dessous (pknown_cefr_zipf_excluded), voir sa docstring.
    type_surfaces: dict[tuple[str, str], set[str]] = {}

    # lemma -> {"surfaces": set[str], "phrases": dict[phrase_key, text]}
    by_lemma: dict[str, dict] = {}
    # lemma -> {"surfaces": set[str], "count": int} — filtre 0bis, cognats
    # retirés avant même d'atteindre by_lemma (voir write_cognates_removed_csv)
    cognates_removed: dict[str, dict] = {}
    false_friend_lemmas_seen: set[str] = set()

    lines = list(iter_book_lines(book_path, skip_lines=skip_lines))
    for line_idx, doc in enumerate(nlp.pipe(lines, batch_size=64)):
        sents = list(doc.sents)
        sent_idx_by_start = {s.start_char: i for i, s in enumerate(sents)}
        for token in doc:
            if not token.is_alpha:
                continue
            if token.pos_ in EXCLUDED_UPOS:
                continue

            lemma = token.lemma_.casefold()
            upos = token.pos_

            # Faux-ami vérifié EN PREMIER : protège de l'exclusion cognat
            # juste en dessous (cas réel : "assist" est à la fois dans
            # CogNet — partage un synset avec "assister" — ET dans la liste
            # faux-amis — "assist"=aider, "assister"=to attend). Un mot dans
            # les deux listes n'est donc jamais retiré comme cognat.
            is_false_friend = lemma in false_friends
            if is_false_friend:
                false_friend_lemmas_seen.add(lemma)

            # Filtre 0bis — cognat : retiré immédiatement, avant tout le
            # reste de la chaîne (jamais compté dans tokens_alpha/seen_types,
            # jamais dans by_lemma). Journalisé séparément pour
            # write_cognates_removed_csv.
            if not is_false_friend and lemma in cognates:
                removed = cognates_removed.setdefault(
                    lemma, {"surfaces": set(), "upos": set(), "count": 0, **cognates[lemma]}
                )
                removed["surfaces"].add(token.text)
                removed["upos"].add(upos)
                removed["count"] += 1
                continue

            stats["tokens_alpha"] += 1

            type_key = (lemma, upos)
            type_surfaces.setdefault(type_key, set()).add(token.text)

            if type_key not in seen_types:
                seen_types.add(type_key)
                stats["types_seen"] += 1
                pknown = pknown_scores.get(lemma)
                if pknown is not None and pknown > MIN_PKNOWN:
                    stats["types_after_pknown"] += 1
                    levels = cefr_levels_for(cefr_by_word, lemma, upos)
                    is_basic_only = bool(levels) and levels.issubset(EXCLUDED_CEFR)
                    if not is_basic_only:
                        stats["types_after_cefr"] += 1
                    elif zipf_frequency(lemma, "en") < ZIPF_RESCUE_THRESHOLD:
                        rescued_types.add(type_key)
                        stats["types_rescued_zipf"] += 1

            if not is_false_friend and not passes_filter(
                lemma, upos, pknown_scores, cefr_by_word, filter_cache
            ):
                continue
            passed_types.add(type_key)

            sent = token.sent
            sent_idx = sent_idx_by_start[sent.start_char]

            # Phrase courte (< MIN_SENTENCE_WORDS mots) : on l'étend avec la
            # phrase précédente et la phrase suivante de la même ligne, si
            # elles existent (voir MIN_SENTENCE_WORDS).
            if len(sent.text.split()) < MIN_SENTENCE_WORDS:
                span_start_char = sents[sent_idx - 1].start_char if sent_idx > 0 else sent.start_char
                span_end_char = (
                    sents[sent_idx + 1].end_char if sent_idx < len(sents) - 1 else sent.end_char
                )
            else:
                span_start_char = sent.start_char
                span_end_char = sent.end_char

            raw_sent_text = doc.text[span_start_char:span_end_char]
            phrase_text = raw_sent_text.strip()
            if not phrase_text:
                continue
            phrase_key = (line_idx, sent.start_char)

            # Offset du token dans `phrase_text` (après strip), pour pouvoir
            # tester plus tard le chevauchement avec les spans MWE détectés
            # sur ce même texte (filtre 5) — voir mwe_member_spans.
            lstrip_offset = len(raw_sent_text) - len(raw_sent_text.lstrip())
            token_start = (token.idx - span_start_char) - lstrip_offset
            token_end = token_start + len(token.text)

            entry = by_lemma.setdefault(lemma, {"surfaces": set(), "upos": set(), "phrases": {}})
            entry["surfaces"].add(token.text)
            entry["upos"].add(upos)
            phrase_entry = entry["phrases"].setdefault(
                phrase_key, {"text": phrase_text, "token_spans": []}
            )
            phrase_entry["token_spans"].append((token_start, token_end))

    # Audit filtres 1-3 (Pknown / CEFR basique / repêchage Zipf) : jusqu'ici
    # ces trois filtres ne laissaient AUCUNE trace écrite (seuls des
    # compteurs agrégés en console), contrairement aux filtres 0bis et 5 —
    # voir write_pknown_cefr_zipf_excluded_csv. Un type (lemma, upos) VU
    # dans le texte (seen_types) mais jamais RETENU (passed_types) a été
    # écarté soit au filtre 1 (Pknown absent ou <= MIN_PKNOWN), soit au
    # filtre 2 (CEFR A1/A2 exclusif pour ce POS) faute d'avoir été repêché
    # par le filtre 3 (Zipf < ZIPF_RESCUE_THRESHOLD) — passes_filter() est
    # la seule source de vérité pour distinguer les deux cas, rejouée ici
    # à l'identique (même cache, donc gratuite).
    pknown_cefr_zipf_excluded: dict[tuple[str, str], dict] = {}
    for lemma, upos in seen_types - passed_types:
        pknown = pknown_scores.get(lemma)
        reason = (
            "pknown_absent_or_low" if pknown is None or not (pknown > MIN_PKNOWN)
            else "cefr_basic_not_rescued_by_zipf"
        )
        pknown_cefr_zipf_excluded[(lemma, upos)] = {
            "reason": reason,
            "pknown": pknown,
            "surfaces": type_surfaces.get((lemma, upos), set()),
        }

    # Filtre supplémentaire 4 : exclusion finale des lemmes ayant un niveau
    # A1/A2 pour n'importe quel POS dans cefrj.csv (voir lemma_has_basic_level).
    # Un faux-ami (filtre 0bis) est protégé : jamais exclu ici.
    excluded_lemmas = [
        lemma for lemma in by_lemma
        if lemma not in false_friends
        and lemma_has_basic_level(cefr_by_word, lemma, lemma_basic_cache)
    ]
    # Détails capturés AVANT suppression de by_lemma (voir
    # write_basic_level_excluded_csv) — même principe d'audit que ci-dessus.
    basic_level_excluded: dict[str, dict] = {
        lemma: {
            "upos": set(by_lemma[lemma]["upos"]),
            "surfaces": set(by_lemma[lemma]["surfaces"]),
        }
        for lemma in excluded_lemmas
    }
    for lemma in excluded_lemmas:
        del by_lemma[lemma]
    stats["lemmas_excluded_basic_level_any_pos"] = len(excluded_lemmas)

    # Filtre supplémentaire 5 : exclusion des lemmes dont au moins une
    # occurrence retenue tombe dans une MWE détectée par le matcher de prod
    # (voir lemma_mwe_idioms / mwe_member_spans). Un faux-ami est protégé :
    # jamais exclu ici (mais on ne calcule même pas lemma_mwe_idioms pour lui,
    # inutile puisque le résultat serait ignoré).
    print("Chargement du détecteur MWE de prod (idiomatch, pipeline/mwe.py)...")
    matcher = mwe_module.get_matcher()
    mwe_phrase_cache: dict[str, list] = {}
    mwe_exclusions: dict[str, dict] = {}
    for lemma, entry in by_lemma.items():
        if lemma in false_friends:
            continue
        idioms = lemma_mwe_idioms(entry, matcher, mwe_phrase_cache)
        if idioms:
            mwe_exclusions[lemma] = {"idioms": idioms, "upos": entry["upos"]}
    for lemma in mwe_exclusions:
        del by_lemma[lemma]
    stats["lemmas_excluded_mwe_member"] = len(mwe_exclusions)

    stats["lemmas_removed_cognate"] = len(cognates_removed)
    stats["false_friend_lemmas_seen"] = len(false_friend_lemmas_seen)

    final_lemmas = set(by_lemma)
    stats["lemmas_retained"] = len(by_lemma)
    stats["types_retained"] = sum(1 for lemma, _upos in passed_types if lemma in final_lemmas)
    return (
        by_lemma, mwe_exclusions, cognates_removed,
        pknown_cefr_zipf_excluded, basic_level_excluded, stats,
    )


# --------------------------------------------------------------------------
# Écriture CSV
# --------------------------------------------------------------------------

def lemma_annotations(
    lemma: str, cefr_by_word: dict, aoa_scores: dict[str, float], false_friends: set[str],
) -> tuple[str, str, str, str]:
    """Colonnes informatives communes aux 3 CSV de sortie — n'affectent
    aucun filtre, voir docstring du module. Renvoie (false_friend, cefr,
    zipf, aoa) déjà formatées en chaînes, chaîne vide si inconnu."""
    is_false_friend = "true" if lemma in false_friends else "false"
    cefr = "/".join(sorted(all_cefr_levels_for_lemma(cefr_by_word, lemma)))
    zipf = f"{zipf_frequency(lemma, 'en'):.2f}"
    aoa = aoa_scores.get(lemma)
    aoa_str = f"{aoa:.2f}" if aoa is not None else ""
    return is_false_friend, cefr, zipf, aoa_str


def write_csv(
    by_lemma: dict[str, dict], out_path: Path, max_phrases: int,
    cefr_by_word: dict, aoa_scores: dict[str, float], false_friends: set[str],
) -> None:
    rows = []
    for lemma, entry in by_lemma.items():
        phrases = [p["text"] for p in entry["phrases"].values()]
        nb_phrases = len(phrases)
        shown = phrases if max_phrases <= 0 else phrases[:max_phrases]
        is_false_friend, cefr, zipf, aoa = lemma_annotations(
            lemma, cefr_by_word, aoa_scores, false_friends
        )
        rows.append(
            (
                lemma,
                "/".join(sorted(entry["upos"])),
                is_false_friend,
                cefr,
                zipf,
                aoa,
                "/".join(sorted(entry["surfaces"])),
                " || ".join(shown),
                nb_phrases,
            )
        )

    rows.sort(key=lambda r: (-r[8], r[0]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["lemma", "pos", "false_friend", "cefr", "zipf", "aoa",
             "surface_forms", "phrases", "nb_phrases"]
        )
        writer.writerows(rows)


def write_mwe_exclusions_csv(
    mwe_exclusions: dict[str, dict], out_path: Path,
    cefr_by_word: dict, aoa_scores: dict[str, float], false_friends: set[str],
) -> None:
    """lemme exclu par le filtre 5 -> POS, idiome(s) responsables (voir
    lemma_mwe_idioms) — un lemme peut apparaître dans plusieurs idiomes
    différents selon ses phrases, listés triés et joints par '/'."""
    rows = sorted(
        (lemma, "/".join(sorted(data["upos"])),
         *lemma_annotations(lemma, cefr_by_word, aoa_scores, false_friends),
         "/".join(sorted(data["idioms"])))
        for lemma, data in mwe_exclusions.items()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "pos", "false_friend", "cefr", "zipf", "aoa", "idioms"])
        writer.writerows(rows)


def write_pknown_cefr_zipf_excluded_csv(
    excluded: dict[tuple[str, str], dict], out_path: Path, cefr_by_word: dict,
) -> None:
    """(lemma, POS) rencontré dans le texte mais jamais entré dans by_lemma —
    filtre 1 (Pknown absent ou <= MIN_PKNOWN) ou filtre 2/3 (CEFR A1/A2
    exclusif pour ce POS, non repêché par ZIPF_RESCUE_THRESHOLD), voir
    passes_filter() et le calcul de `pknown_cefr_zipf_excluded` dans
    build_word_contexts(). Contrairement aux filtres 0bis/5, cette
    exclusion n'avait jusqu'ici aucune trace écrite (seuls des compteurs
    agrégés en console) — d'où ce fichier, ajouté après coup sur un cas
    concret (le verbe "draw", A1 mais trop fréquent — Zipf 4.81 — pour être
    repêché)."""
    rows = sorted(
        (
            lemma, upos, data["reason"],
            f"{data['pknown']:.2f}" if data["pknown"] is not None else "",
            "/".join(sorted(cefr_levels_for(cefr_by_word, lemma, upos))),
            f"{zipf_frequency(lemma, 'en'):.2f}",
            "/".join(sorted(data["surfaces"])),
        )
        for (lemma, upos), data in excluded.items()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "pos", "reason", "pknown", "cefr", "zipf", "surface_forms"])
        writer.writerows(rows)


def write_basic_level_excluded_csv(
    excluded: dict[str, dict], out_path: Path, cefr_by_word: dict,
) -> None:
    """Lemme exclu par le filtre 4 (lemma_has_basic_level) : au moins un
    niveau A1/A2 connu pour ce lemme, TOUS POS confondus dans cefrj.csv —
    même quand le(s) POS effectivement rencontré(s) dans le texte (colonne
    `pos`, remplie AVANT suppression de by_lemma) n'était pas lui-même
    A1/A2 (sinon le lemme aurait déjà été écarté au filtre 2/3, voir
    write_pknown_cefr_zipf_excluded_csv ci-dessus). `cefr` liste ici
    l'union tous-POS (all_cefr_levels_for_lemma), pas seulement celui du
    POS rencontré."""
    rows = sorted(
        (
            lemma,
            "/".join(sorted(data["upos"])),
            "/".join(sorted(all_cefr_levels_for_lemma(cefr_by_word, lemma))),
            "/".join(sorted(data["surfaces"])),
        )
        for lemma, data in excluded.items()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "pos", "cefr", "surface_forms"])
        writer.writerows(rows)


def write_cognates_removed_csv(
    cognates_removed: dict[str, dict], out_path: Path,
    cefr_by_word: dict, aoa_scores: dict[str, float], false_friends: set[str],
) -> None:
    """lemme cognat (filtre 0bis) -> POS, mot FR cognat, concept_id WordNet
    (DATASETS/cognet_en_fr.csv), formes de surface rencontrées et nombre
    d'occurrences retirées avant même d'entrer dans le reste de la chaîne.
    `false_friend` vaut toujours "false" ici par construction (voir
    docstring du module : un faux-ami ne peut jamais être retiré comme
    cognat, le check faux-ami passe en premier)."""
    rows = sorted(
        (
            lemma,
            "/".join(sorted(data["upos"])),
            *lemma_annotations(lemma, cefr_by_word, aoa_scores, false_friends),
            data["word_fr"],
            data["concept_id"],
            "/".join(sorted(data["surfaces"])),
            data["count"],
        )
        for lemma, data in cognates_removed.items()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["lemma", "pos", "false_friend", "cefr", "zipf", "aoa",
             "word_fr", "concept_id", "surface_forms", "occurrences_removed"]
        )
        writer.writerows(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default=str(DEFAULT_BOOK_PATH),
                         help="Chemin du livre .txt (défaut : The Humans)")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV de sortie")
    parser.add_argument("--mwe-exclusions-out", default=str(DEFAULT_MWE_EXCLUSIONS_OUT_PATH),
                         help="Chemin du CSV des lemmes exclus par le filtre 5 (MWE) "
                              "et de leur(s) idiome(s)")
    parser.add_argument("--cognates-removed-out", default=str(DEFAULT_COGNATES_REMOVED_OUT_PATH),
                         help="Chemin du CSV des lemmes retirés par le filtre 0bis (cognats)")
    parser.add_argument("--pknown-cefr-excluded-out",
                         default=str(DEFAULT_PKNOWN_CEFR_EXCLUDED_OUT_PATH),
                         help="Chemin du CSV des (lemme, POS) écartés par le filtre 1 (Pknown) "
                              "ou le filtre 2/3 (CEFR A1/A2 non repêché par le Zipf)")
    parser.add_argument("--basic-level-excluded-out",
                         default=str(DEFAULT_BASIC_LEVEL_EXCLUDED_OUT_PATH),
                         help="Chemin du CSV des lemmes écartés par le filtre 4 "
                              "(A1/A2 pour au moins un POS, tous POS cefrj.csv confondus)")
    parser.add_argument("--max-phrases", type=int, default=0,
                         help="Plafond de phrases affichées par lemme dans la colonne "
                              "'phrases' (0 = toutes, défaut). Ne change pas nb_phrases.")
    parser.add_argument("--skip-lines", type=int, default=0,
                         help="Nombre de lignes de tête (hors-œuvre : copyright, sommaire, "
                              "distribution...) à ignorer en plus de la détection par motifs "
                              "(0 = aucune, défaut ; 182 pour le livre complet The Humans).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    book_path = Path(args.book)
    out_path = Path(args.out)
    mwe_exclusions_out_path = Path(args.mwe_exclusions_out)
    cognates_removed_out_path = Path(args.cognates_removed_out)
    pknown_cefr_excluded_out_path = Path(args.pknown_cefr_excluded_out)
    basic_level_excluded_out_path = Path(args.basic_level_excluded_out)

    if not book_path.exists():
        print(f"Livre introuvable : {book_path}")
        return 1

    print(f"Livre : {book_path}")
    (
        by_lemma, mwe_exclusions, cognates_removed,
        pknown_cefr_zipf_excluded, basic_level_excluded, stats,
    ) = build_word_contexts(book_path, skip_lines=args.skip_lines)

    # Rechargés ici pour les colonnes informatives des CSV (false_friend/
    # cefr/zipf/aoa) — fichiers légers, cohérent avec le style du script
    # (chaque fonction d'écriture reste autonome sur ses dépendances).
    cefr_by_word = load_cefr_by_word()
    aoa_scores = load_aoa_scores()
    false_friends = load_false_friends()

    write_csv(by_lemma, out_path, args.max_phrases, cefr_by_word, aoa_scores, false_friends)
    write_mwe_exclusions_csv(
        mwe_exclusions, mwe_exclusions_out_path, cefr_by_word, aoa_scores, false_friends
    )
    write_cognates_removed_csv(
        cognates_removed, cognates_removed_out_path, cefr_by_word, aoa_scores, false_friends
    )
    write_pknown_cefr_zipf_excluded_csv(
        pknown_cefr_zipf_excluded, pknown_cefr_excluded_out_path, cefr_by_word
    )
    write_basic_level_excluded_csv(
        basic_level_excluded, basic_level_excluded_out_path, cefr_by_word
    )

    print()
    print("=== Entonnoir (au niveau type lemme+POS) ===")
    print(f"Lemmes retirés filtre 0bis (cognats) : {stats['lemmas_removed_cognate']}")
    print(f"Lemmes faux-amis rencontrés (protégés) : {stats['false_friend_lemmas_seen']}")
    print(f"Tokens alphabétiques vus       : {stats['tokens_alpha']}")
    print(f"Types (lemme, POS) distincts   : {stats['types_seen']}")
    print(f"Après filtre 1 (Pknown > {MIN_PKNOWN}) : {stats['types_after_pknown']}")
    print(f"Après filtre 2 (CEFR {sorted(EXCLUDED_CEFR)} exclusif) : {stats['types_after_cefr']}")
    print(f"Repêchés filtre 3 (Zipf < {ZIPF_RESCUE_THRESHOLD}) : {stats['types_rescued_zipf']}")
    print(f"Lemmes exclus filtre 4 (A1/A2 tous POS) : {stats['lemmas_excluded_basic_level_any_pos']}")
    print(f"Lemmes exclus filtre 5 (membre de MWE) : {stats['lemmas_excluded_mwe_member']}")
    print(f"Total types retenus            : {stats['types_retained']}")
    print(f"Lemmes distincts retenus       : {stats['lemmas_retained']}")
    print()
    print(f"-> {out_path}")
    print(f"-> {mwe_exclusions_out_path} ({len(mwe_exclusions)} lemme(s) exclu(s) par le filtre 5)")
    print(f"-> {cognates_removed_out_path} ({len(cognates_removed)} lemme(s) retiré(s) par le filtre 0bis)")
    print(f"-> {pknown_cefr_excluded_out_path} "
          f"({len(pknown_cefr_zipf_excluded)} type(s) écarté(s) par le filtre 1/2/3)")
    print(f"-> {basic_level_excluded_out_path} "
          f"({len(basic_level_excluded)} lemme(s) écarté(s) par le filtre 4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
