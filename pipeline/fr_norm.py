"""Normalisation et comparaison de candidats de traduction française —
étape 1 du dispositif d'arbitrage sans relecture humaine (voir le plan
"Valider / corriger suggested_fr et suggested_fr_alt").

Constat qui motive ce module, mesuré sur les 210 lignes `pending` de
`pipeline_out/sense_fr_review.csv` (toutes en statut `frontier_desaccord`) :
`senses.fr_stem` (retrait naïf de `-s`/`-x` final, comparaison de chaînes
ENTIÈRES) déclare "désaccord" dans 98 cas sur 210 (47%) alors que la
proposition du modèle et celle de la ressource désignent le même sens —
`care.v.01` "se soucier" vs omw "soucier" (pronom réfléchi), `foot.n.01`
"pied (membre)" vs omw "pied" (parenthèse), `thump.n.01` "coup sourd" vs
WoNeF "bruit sourd" (recouvrement partiel sur les mots de contenu).

`senses.fr_stem` N'EST PAS modifié ici : il est utilisé en amont dans le
scoring (fr_score/fr_hits, pipeline/senses.py) et le modifier déplacerait
des scores déjà calibrés. Ce module est un comparateur DÉDIÉ à l'arbitrage
des candidats de traduction, plus tolérant :

- une parenthèse disambiguatrice ("pied (membre)") est séparée de la
  traduction principale et conservée à part (`fr_gloss`) plutôt que de
  polluer la comparaison ET `meaning_fr_official` à l'export ;
- une parenthèse qui est un simple marqueur grammatical ("capable (de)")
  est retirée sans être promue en glose ;
- pronoms réfléchis, articles, prépositions de tête sont ignorés ;
- la radicalisation utilise `nltk.stem.snowball.FrenchStemmer` (règles
  de dérivation françaises) plutôt que le retrait `-s/-x` ;
- la comparaison se fait par ENSEMBLE de mots de contenu (mot-tête commun,
  ou indice de Jaccard >= seuil), jamais par égalité de chaîne entière.
"""

from __future__ import annotations

import re
import unicodedata

from nltk.stem.snowball import FrenchStemmer

_stemmer = FrenchStemmer()

# ------------------------------------------------------------------
# Séparation traduction / glose disambiguatrice
# ------------------------------------------------------------------

_PAREN_RE = re.compile(r"\(([^)]*)\)")

# Contenu de parenthèse qui est un marqueur GRAMMATICAL (régime du verbe,
# construction) plutôt qu'une glose disambiguatrice — à retirer sans être
# conservé. Comparé après normalisation (accents/casse retirés).
_TAIL_MARKER_WORDS = {
    "de", "d'", "a", "à", "au", "aux", "du", "des",
    "qqn", "qqch", "qqn.", "qqch.",
    "de qqch", "de qqn", "a qqn", "a qqch", "à qqn", "à qqch",
    "qqn de qqch", "se", "s'",
}


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _normalize_marker(text: str) -> str:
    return strip_accents(text.casefold()).strip(" .")


def split_gloss(text: str) -> tuple[str, str | None]:
    """Sépare une parenthèse de la traduction principale.

    - parenthèse = marqueur grammatical connu (`_TAIL_MARKER_WORDS`) ->
      retirée, pas de glose ("capable (de)" -> ("capable", None)) ;
    - parenthèse = autre chose -> traitée comme une glose disambiguatrice,
      extraite dans le second élément du tuple
      ("pied (membre)" -> ("pied", "membre")) ;
    - pas de parenthèse -> (texte tel quel, None).

    Seule la PREMIÈRE parenthèse est traitée (les traductions du magasin
    n'en portent jamais plus d'une en pratique)."""
    match = _PAREN_RE.search(text)
    if not match:
        return text.strip(), None
    inner = match.group(1).strip()
    main = (text[:match.start()] + text[match.end():]).strip()
    main = re.sub(r"\s+", " ", main)
    if _normalize_marker(inner) in _TAIL_MARKER_WORDS:
        return main, None
    return main, inner or None


# ------------------------------------------------------------------
# Radicalisation
# ------------------------------------------------------------------


def stem(word: str) -> str:
    """Radical snowball d'un mot déjà en minuscules/sans accents attendu
    en entrée typique, mais accepte aussi une forme brute (normalise elle-
    même). Volontairement permissif : un radical vide ou une exception du
    stemmer renvoie le mot normalisé tel quel plutôt que de lever."""
    normalized = strip_accents(word.casefold()).strip()
    if not normalized:
        return normalized
    try:
        stemmed = _stemmer.stem(normalized)
    except Exception:
        return normalized
    return stemmed or normalized


# ------------------------------------------------------------------
# Décomposition en mots de contenu
# ------------------------------------------------------------------

# Un candidat peut grouper plusieurs synonymes ("tenir à ; se soucier de")
# — comme dans les listes fr_alt/omw/wonef existantes, mais aussi parfois
# à l'intérieur d'une même cellule CSV relue depuis le magasin.
_SPLIT_RE = re.compile(r"[;/,]")

_HEAD_MARKERS = {
    "se", "s", "le", "la", "les", "l", "un", "une",
    "de", "du", "des", "d", "a", "à", "au", "aux", "en",
}

_STOPWORDS = _HEAD_MARKERS | {
    "etre", "avoir", "que", "qui", "ce", "cette", "ces", "son", "sa", "ses",
    "qqn", "qqch",
}


def _clean_tokens(text: str) -> list[str]:
    cleaned = strip_accents(text.casefold())
    cleaned = re.sub(r"[-']", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned)
    return [t for t in cleaned.split() if t]


def content_tokens(candidate: str) -> list[str]:
    """Mots de contenu (radicaux), dans l'ordre, pour UNE portion de
    candidat (déjà éclatée sur ;/,). Ignore articles/pronoms/prépositions
    de tête et la glose entre parenthèses (retirée par split_gloss)."""
    main, _gloss = split_gloss(candidate)
    tokens = [t for t in _clean_tokens(main) if t not in _STOPWORDS and len(t) > 2]
    return [stem(t) for t in tokens]


def content_word_set(candidate: str) -> set[str]:
    """Ensemble des radicaux de mots de contenu de TOUT le candidat, en
    éclatant d'abord sur ;/, (plusieurs synonymes dans une même cellule)."""
    words: set[str] = set()
    for part in _SPLIT_RE.split(candidate):
        words.update(content_tokens(part))
    return words


def readable_content_words(candidate: str) -> list[str]:
    """Mots de contenu d'un candidat, SANS radicalisation — accents et
    casse d'origine préservés. Sert aux tests d'attestation externes
    (wordfreq) qui ont besoin de la forme fléchie réelle, un radical
    snowball n'étant pas forcément un mot français attesté."""
    words: list[str] = []
    for part in _SPLIT_RE.split(candidate):
        main, _gloss = split_gloss(part)
        for tok in re.split(r"[-'\s]+", main):
            tok_clean = tok.strip(" .,;:!?\"'()")
            if not tok_clean:
                continue
            key = strip_accents(tok_clean.casefold())
            if key in _STOPWORDS or len(key) <= 2:
                continue
            words.append(tok_clean)
    return words


def head_stem(candidate: str) -> str | None:
    """Radical du premier mot de contenu de la PREMIÈRE portion du
    candidat (avant le premier ;/,) — sert au critère "mot-tête commun"."""
    first_part = _SPLIT_RE.split(candidate)[0]
    tokens = content_tokens(first_part)
    return tokens[0] if tokens else None


# ------------------------------------------------------------------
# Comparaison
# ------------------------------------------------------------------


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def candidates_match(a: str, b: str, jaccard_threshold: float = 0.5) -> bool:
    """Concordance entre deux candidats : mot-tête commun (fort — un seul
    mot de recouvrement au bon endroit suffit), OU indice de Jaccard sur
    l'ensemble des mots de contenu >= seuil (recouvrement partiel mais
    substantiel, p.ex. "coup sourd" / "bruit sourd")."""
    head_a, head_b = head_stem(a), head_stem(b)
    if head_a is not None and head_a == head_b:
        return True
    words_a, words_b = content_word_set(a), content_word_set(b)
    if not words_a or not words_b:
        return False
    if words_a & words_b:
        return True
    return jaccard(words_a, words_b) >= jaccard_threshold


def any_match(candidates: list[str], resources: list[str], jaccard_threshold: float = 0.5) -> bool:
    """Vrai si au moins un candidat (p.ex. [fr] + fr_alt) concorde avec au
    moins une ressource (p.ex. omw_fr + wonef) au sens de `candidates_match`."""
    for c in candidates:
        if not c:
            continue
        for r in resources:
            if not r:
                continue
            if candidates_match(c, r, jaccard_threshold=jaccard_threshold):
                return True
    return False
