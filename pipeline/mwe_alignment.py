"""Lot 1 — Dérivation exacte des membres d'un candidat idiomatch.

Corrige le défaut A (plan Partie 2, point D) : `is_covered()` réservait
jusqu'ici l'ENVELOPPE entière d'un match idiomatch, avalant les mots
intercalés (`turn off` avalait `lantern` dans "turned the lantern off").
L'heuristique de repli envisagée un temps ("intersecter avec les mots-
lemmes de l'idiome, ignorer someone/one's") est fausse dans le cas général
— vérifié sur idiomatch 0.2.14 : un pronom réel comme "him" dans
"let him go" VALIDE l'ancre `{POS: PRON}` de l'idiome, ce n'est pas du
bruit à écarter.

Vérifié sur `idiomatch.builders` (openslot/hyphenated/default, tous
passés par `slop()`) : chaque motif spaCy Matcher est une séquence stricte
d'éléments dont un seul type sert de remplissage arbitraire — celui que
`slop()` insère ENTRE deux éléments réels du motif, reconnaissable
exactement par `{"TEXT": {"REGEX": idiomatch.configs.WILDCARD}, ...}`.
Tout le reste — y compris un élément optionnel comme la virgule ou le
trait d'union propres à l'idiome (`{"TEXT": ",", "OP": "?"}`,
`{"TEXT": "-", "OP": "?"}`, produits par openslot()/hyphenated()) — est un
composant réel de l'idiome : présent dans l'occurrence, il compte comme
membre, même s'il est marqué optionnel (`OP`). La présence d'un `OP` ne
distingue donc PAS à elle seule membre et remplissage — seul le motif
WILDCARD exact le fait.

Méthode : pour chaque candidat, rejouer déterministiquement (recherche
avec retour arrière, mémoïsée — les spans sont courts, quelques tokens)
chaque alternative de motif enregistrée pour cet idiome contre les tokens
du span déjà reconnu par idiomatch. Si toutes les alternatives, tous
comptes faits, ne s'accordent que sur UN SEUL ensemble d'indices membres,
il est retenu. Sinon (aucune solution, ou plusieurs ensembles distincts
possibles), le candidat est marqué `ambiguous_alignment` et ne réserve
aucun token — jamais un mot supprimé sur une hypothèse incertaine.
"""

from __future__ import annotations

import json
import re

from idiomatch.builders import build as _idiomatch_build
from idiomatch.configs import WILDCARD
from idiomatch.idiomatcher import RESOURCES_DIR

_PATTERNS_BY_N: dict[int, dict[str, list]] = {}
_CUSTOM_PATTERN_CACHE: dict[tuple[str, int], list] = {}


def _load_pretrained_patterns(n: int) -> dict[str, list]:
    if n not in _PATTERNS_BY_N:
        path = RESOURCES_DIR / f"slop_{n}.json"
        with path.open(encoding="utf-8") as f:
            _PATTERNS_BY_N[n] = json.load(f)
    return _PATTERNS_BY_N[n]


def patterns_for_idiom(idiom: str, nlp, n: int) -> list[list[dict]]:
    """Les alternatives de motif spaCy Matcher enregistrées pour `idiom`
    sous un Idiomatcher(n=n) — lues depuis le fichier pré-entraîné
    (`slop_{n}.json`) si l'idiome y figure, sinon reconstruites via
    `idiomatch.builders.build` (la MÊME fonction qu'utilise
    `Idiomatcher.add_idioms`), pour couvrir aussi les idiomes custom
    (`crack open`, `smart ass`, `data/custom_lexicon.jsonl`) absents du
    fichier pré-entraîné."""

    pretrained = _load_pretrained_patterns(n)
    if idiom in pretrained:
        return pretrained[idiom]
    key = (idiom, n)
    if key not in _CUSTOM_PATTERN_CACHE:
        built = _idiomatch_build([idiom], nlp, n)
        _CUSTOM_PATTERN_CACHE[key] = built.get(idiom, [])
    return _CUSTOM_PATTERN_CACHE[key]


def _is_filler(spec: dict) -> bool:
    """Vrai seulement pour le remplissage générique inséré par slop()
    entre deux éléments réels du motif — jamais pour un composant
    optionnel de l'idiome lui-même (virgule, trait d'union), qui porte
    aussi un OP mais avec un TEXT littéral, pas le regex WILDCARD."""

    text = spec.get("TEXT")
    return isinstance(text, dict) and text.get("REGEX") == WILDCARD


def _op_bounds(spec: dict) -> tuple[int, int]:
    op = spec.get("OP")
    if op is None:
        return (1, 1)
    if op == "?":
        return (0, 1)
    m = re.fullmatch(r"\{(\d+),(\d+)\}", op)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (0, 1)  # repli prudent si idiomatch introduit un OP inconnu


def _spec_matches_token(spec: dict, token) -> bool:
    """`re.search`, PAS `re.match` : vérifié empiriquement sur une
    occurrence réelle du livre ("I've done" avec apostrophe courbe ’ —
    segment 445) que le Matcher spaCy accepte le token "’ve" comme
    remplissage WILDCARD alors que le regex générique
    (`[a-zA-Z0-9,\\-\\'\\"]+`, sans ancres ^/$) échoue en `re.match`
    (la courbe ’ n'est pas dans la classe de caractères, en position 0)
    mais réussit en `re.search` (trouve "ve" plus loin dans le token).
    Les motifs LEMMA/TAG d'idiomatch incluent déjà leurs propres ancres
    `^...$` dans la chaîne de regex, donc `re.search` s'y comporte comme
    `re.match` pour eux — seul le remplissage générique, sans ancres, est
    concerné par la différence."""

    if "LEMMA" in spec:
        return re.search(spec["LEMMA"]["REGEX"], token.lemma_) is not None
    if "TAG" in spec:
        return token.tag_ == spec["TAG"]
    if "POS" in spec:
        return token.pos_ == spec["POS"]
    if "TEXT" in spec:
        val = spec["TEXT"]
        if isinstance(val, dict) and "REGEX" in val:
            return re.search(val["REGEX"], token.text) is not None
        return token.text == val
    return False


def _iter_assignments(spec_list: list[dict], tokens: list) -> list[tuple[tuple[int, int], ...]]:
    """Comme `_all_alignments` ci-dessous, mais chaque solution est la
    liste COMPLÈTE des paires (indice de spec, indice de token) pour les
    tokens membres, dans l'ordre du motif — pas seulement l'ensemble nu des
    indices de tokens. Nécessaire à `pipeline/mwe_gates.py`, qui doit savoir
    QUEL spec (slot ou ancre) a consommé quel token, information que
    l'ensemble seul ne porte pas.

    Même mémoïsation que l'original : le résultat pour un état (si, ti) ne
    dépend que des specs restantes et des tokens à partir de ti, jamais du
    chemin emprunté pour y arriver — donc réutilisable tel quel."""

    n_specs, n_tokens = len(spec_list), len(tokens)
    memo: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {}

    def rec(si: int, ti: int) -> list[tuple[tuple[int, int], ...]]:
        key = (si, ti)
        if key in memo:
            return memo[key]
        if si == n_specs:
            result: list[tuple[tuple[int, int], ...]] = [()] if ti == n_tokens else []
            memo[key] = result
            return result

        spec = spec_list[si]
        lo, hi = _op_bounds(spec)
        is_member = not _is_filler(spec)
        results: list[tuple[tuple[int, int], ...]] = []
        for k in range(0, hi + 1):
            if ti + k > n_tokens:
                break
            if k > 0 and not _spec_matches_token(spec, tokens[ti + k - 1]):
                break  # ce token ne matche plus le motif : k plus grand ne marchera pas
            if k < lo:
                continue
            consumed = tuple((si, ti + j) for j in range(k)) if is_member else ()
            for rest in rec(si + 1, ti + k):
                results.append(consumed + rest)
        memo[key] = results
        return results

    return rec(0, 0)


def _all_alignments(spec_list: list[dict], tokens: list) -> set[frozenset[int]]:
    """Toutes les façons d'aligner `spec_list` sur TOUTE la séquence
    `tokens` (bornes exactes : rien avant, rien après). Chaque solution
    est l'ensemble des indices (dans `tokens`) considérés membres réels de
    l'idiome — le remplissage générique (`_is_filler`) n'y figure jamais,
    qu'il ait été consommé ou non.

    Réduction de `_iter_assignments` : c'est le nombre d'ensembles de
    membres DISTINCTS qui compte ici, pas le nombre de dérivations —
    plusieurs affectations complètes peuvent produire le même ensemble."""

    return {
        frozenset(ti for _, ti in assignment)
        for assignment in _iter_assignments(spec_list, tokens)
    }


class AlignmentResult:
    __slots__ = ("member_indices", "ambiguous")

    def __init__(self, member_indices: frozenset[int] | None, ambiguous: bool):
        self.member_indices = member_indices
        self.ambiguous = ambiguous


def align_members(idiom: str, span_tokens: list, nlp, n: int) -> AlignmentResult:
    """Rejoue toutes les alternatives de motif de `idiom` contre les
    tokens déjà reconnus par idiomatch (`span_tokens`, typiquement
    `list(doc[start:end])`). Retourne l'ensemble unique d'indices membres
    s'il existe, sinon marque `ambiguous=True` (zéro ou plusieurs
    ensembles distincts possibles)."""

    alternatives = patterns_for_idiom(idiom, nlp, n)
    solutions: set[frozenset[int]] = set()
    for spec_list in alternatives:
        solutions |= _all_alignments(spec_list, span_tokens)

    if len(solutions) != 1:
        return AlignmentResult(member_indices=None, ambiguous=True)
    return AlignmentResult(member_indices=next(iter(solutions)), ambiguous=False)
