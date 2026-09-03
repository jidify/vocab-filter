"""S2 — Portes de validation post-idiomatch (voir
`fix_pipeline/s2_fix/bug_idiomatch_slot_overmatch.md` et le plan qui l'a
corrigé). `idiomatch` sur-généralise trois façons distinctes, toutes
propres à sa dépendance et sans recouvrement avec les phrasal verbs
(`pipeline/vpc`, `pipeline/rules_plus.py`) :

- **slots ouverts** (`know someone`) — `idiomatch/builders.py::openslot_passive`
  réordonne les tokens ET rappelle `slop()` sur un motif déjà slopé ; le
  motif "passif" qui en résulte accepte n'importe quel pronom à distance de
  l'ancre, sans lien syntaxique réel. Voir `_slot_gate`.
- **idiomes tout-grammaticaux** (`I do`) — les ancres LEMMA d'un idiome
  composé uniquement de mots grammaticaux/verbes légers matchent leurs
  emplois auxiliaires courants (`I've done`, `Did you see it`). Voir
  `_grammatical_gate`.
- **ancre lexicale corrompue** (`wing it`) — traité séparément dans
  `repair_corrupt_anchor_lemmas`, appliqué au chargement du matcher
  (`mwe.get_matcher`), pas via `classify` : le motif buggé ne doit jamais
  être proposé comme candidat, pas être filtré après coup.

Ces portes ne s'appliquent QU'à la source `idiomatch` (`pipeline/mwe.py`) et
ne modifient jamais VPC/rules_plus. Elles ne couvrent pas la sur-fusion par
remplissage lexical (`go to` ← `Go home to`, Correction S2-1 du plan
d'action) — périmètre distinct, plus risqué pour les vrais phrasal verbs,
volontairement laissé de côté ici."""

from __future__ import annotations

import json

from idiomatch.builders import default as _idiomatch_default_pattern

from poc_pipeline import mwe_alignment
from poc_pipeline.rules_plus import PARTICLE_CLOSED_CLASS

GRAMMATICAL_POS = {
    "PRON", "DET", "ADP", "PART", "AUX", "CCONJ", "SCONJ", "ADV", "NUM",
    "PUNCT", "INTJ",
}
LIGHT_VERB_LEMMAS = {"do", "be", "have"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}

# Porte D — "wing it" est le seul cas confirmé dans idioms.yml (4961 entrées
# auditées manuellement au moment de ce correctif) dont l'ancre LEMMA vient
# d'une lemmatisation spaCy erronée : nlp("wing it") étiquette "wing" en
# VBG et lui applique la règle -ing -> -e (lemme "we"), donc idiomatch
# compile le motif sur "we" et matche n'importe quel "we ... it" ("We all
# love it", "We get it"). nlp("winging it") (le gérondif réel) lemmatise
# "wing" correctement -- on reconstruit le motif à partir de cette forme
# fléchie plutôt que de la forme de citation buguée.
#
# Correction ciblée et vérifiée, pas un scanner générique sur les 4961
# entrées : un heuristique de plausibilité par mot-clé (comparaison naïve
# ancre/mots de la clé) sur-signale une dizaine de faux positifs
# (contractions comme "can't" -> "n't"/"not" invisibles à un split naïf),
# et un scanner fondé sur un appel spaCy par ancre coûterait un aller-retour
# par mot sur ~4961 entrées à CHAQUE démarrage du pipeline pour un seul cas
# réel trouvé.
CORRUPT_ANCHOR_REPAIRS = {
    "wing it": "winging it",
}


def repair_corrupt_anchor_lemmas(matcher) -> None:
    """Appelé une seule fois par processus, dans `mwe.get_matcher()`, juste
    après `Idiomatcher.from_pretrained()` et avant l'ajout des idiomes
    custom du projet (jamais corrompus par construction : leurs motifs sont
    construits par ce dépôt, pas par idiomatch/Wiktionary)."""

    for lemma, replacement in CORRUPT_ANCHOR_REPAIRS.items():
        if not any(idiom.lemma == lemma for idiom in matcher.idioms):
            continue  # idiomatch a changé sa base : rien à réparer
        matcher.remove(lemma)
        tokens = list(matcher.nlp(replacement))
        matcher.add(lemma, [_idiomatch_default_pattern(tokens, matcher.n)])


def _is_slot_spec(spec: dict) -> bool:
    return spec.get("TAG") == "PRP$" or spec.get("POS") == "PRON"


def _has_open_slot(alternatives: list[list[dict]]) -> bool:
    return any(any(_is_slot_spec(s) for s in alt) for alt in alternatives)


def _canonical_signature(alt: list[dict]) -> list[str]:
    return [json.dumps(s, sort_keys=True) for s in alt if not mwe_alignment._is_filler(s)]


def _passive_alternative_ok(alt: list[dict], assignment, span_tokens: list) -> bool:
    """R1 (2/3) — l'alternative réordonnée (produite par
    `openslot_passive`) n'est admise que sur un vrai passif morphologique
    À AU MOINS DEUX ANCRES LEXICALES. La condition du nombre d'ancres est
    nécessaire : sans elle, `it is not known` (une seule ancre, `know`)
    redevient acceptée — le slot y est l'unique argument du verbe,
    structurellement indiscernable d'un passif littéral générique. Avec
    elle, `shut one's mouth` (`shut` + `mouth`, 2 ancres) passe : `mouth`,
    ancre nominale, reste `nsubjpass` de `shut`, ancre verbale, tous deux
    présents et liés dans la phrase."""

    anchors = [(si, ti) for si, ti in assignment if not _is_slot_spec(alt[si])]
    if len(anchors) < 2:
        return False
    return any(
        span_tokens[ti].tag_ == "VBN" and any(c.dep_ == "auxpass" for c in span_tokens[ti].children)
        for _, ti in anchors
    )


def _slot_alignment_ok(alt: list[dict], reordered: bool, assignment, span_tokens: list) -> bool:
    if reordered and not _passive_alternative_ok(alt, assignment, span_tokens):
        return False

    slots = [(si, ti) for si, ti in assignment if _is_slot_spec(alt[si])]
    anchors = [(si, ti) for si, ti in assignment if not _is_slot_spec(alt[si])]
    for si, ti in slots:
        tok = span_tokens[ti]
        # R2 — le token du slot doit être en relation de dépendance directe
        # (tête->enfant ou enfant->tête) avec au moins une ancre lexicale du
        # même alignement. Rejette "know , I" (où "I" est nsubj de "go",
        # hors de la construction candidate).
        if not any(
            tok.head.i == span_tokens[aj].i or span_tokens[aj].head.i == tok.i
            for _, aj in anchors
        ):
            return False
        # R3 — un slot ne peut pas être le sujet d'une ancre qui le
        # PRÉCÈDE dans le motif. Le sujet d'une ancre qui le SUIT reste
        # licite : c'est exactement "let someone go" ("it" est nsubj de
        # "go", qui suit le slot dans le motif).
        if tok.dep_ in SUBJECT_DEPS and any(
            tok.head.i == span_tokens[aj].i for asi, aj in anchors if asi < si
        ):
            return False
    return True


def _slot_gate(idiom: str, span_tokens: list, nlp, n: int) -> str | None:
    alternatives = mwe_alignment.patterns_for_idiom(idiom, nlp, n)
    if not _has_open_slot(alternatives):
        return None

    reference = _canonical_signature(alternatives[0])
    for alt in alternatives:
        reordered = _canonical_signature(alt) != reference
        for assignment in mwe_alignment._iter_assignments(alt, span_tokens):
            if _slot_alignment_ok(alt, reordered, assignment, span_tokens):
                return None
    return "slot_saturation"


def _is_grammatical_idiom(idiom: str, nlp) -> bool:
    """Un idiome appartient à la classe si tous les tokens de sa forme de
    citation ont un POS grammatical, ou sont un verbe léger (`do`/`be`/
    `have`) — sauf si la forme de citation est [verbe léger][particule]
    (`do in`, `do up`, `have up`...) : cette forme est structurellement
    indiscernable d'un vrai phrasal verb séparable (`did him in`), donc
    reste hors de cette porte et retombe dans le périmètre non traité ici
    (sur-fusion par remplissage lexical, Correction S2-1)."""

    tokens = list(nlp(idiom))
    if not all(
        t.pos_ in GRAMMATICAL_POS or (t.pos_ in ("VERB", "AUX") and t.lemma_.lower() in LIGHT_VERB_LEMMAS)
        for t in tokens
    ):
        return False
    if (
        len(tokens) == 2
        and tokens[0].pos_ in ("VERB", "AUX")
        and tokens[0].lemma_.lower() in LIGHT_VERB_LEMMAS
        and tokens[1].lemma_.lower() in PARTICLE_CLOSED_CLASS
    ):
        return False
    return True


def _grammatical_gate(idiom: str, span_tokens: list, nlp) -> str | None:
    citation = list(nlp(idiom))
    if len(span_tokens) != len(citation):
        return "grammatical_gap"  # remplissage consommé : la surface n'est pas contiguë
    for cited_tok, occ_tok in zip(citation, span_tokens):
        # Strictement VERB en citation (pas AUX) : un idiome dont le verbe
        # léger est DÉJÀ un auxiliaire dans sa forme de citation ("how are
        # you", "being that") a cette réalisation comme emploi normal —
        # rien à rejeter. Seul un basculement VERB (citation) -> AUX
        # (occurrence) signale l'emploi grammatical parasite (35 des 47
        # "I do" : "I've done", "Did you see it").
        if (
            cited_tok.pos_ == "VERB"
            and cited_tok.lemma_.lower() in LIGHT_VERB_LEMMAS
            and occ_tok.pos_ == "AUX"
        ):
            return "grammatical_auxiliary"
    return None


def classify(idiom: str, span_tokens: list, nlp, n: int) -> str | None:
    """`None` si le candidat passe, sinon la famille de rejet
    (`"slot_saturation"`, `"grammatical_gap"`, `"grammatical_auxiliary"`).

    Les deux portes ne sont PAS mutuellement exclusives par construction :
    `someone`/`something` sont des slots pour idiomatch (`POS: PRON`) mais
    aussi des `PRON` ordinaires pour `_is_grammatical_idiom` (POS grammatical
    au sens large) — `or something`/`up to something` satisfont les deux
    définitions. L'ordre d'évaluation ci-dessous tranche : `_slot_gate` est
    appelé en premier et retourne avant que `_is_grammatical_idiom` ne soit
    évalué, donc ces idiomes sont toujours décidés par la porte A (vérifié :
    `or something` sort à 0/16, jamais compté deux fois)."""

    slot_reason = _slot_gate(idiom, span_tokens, nlp, n)
    if slot_reason:
        return slot_reason
    if _is_grammatical_idiom(idiom, nlp):
        return _grammatical_gate(idiom, span_tokens, nlp)
    return None
