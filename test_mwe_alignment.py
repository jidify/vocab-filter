"""Lot 1 — dérivation exacte des membres d'un candidat idiomatch
(pipeline/mwe_alignment.py). L'avancement du plan déclarait ce fichier créé
avec ce lot ; il ne l'était pas (écart trouvé et comblé au Lot 3, voir la
section Avancement). Les trois cas ci-dessous sont ceux nommés dans la
docstring de mwe_alignment.py et dans le plan (Lot 1) : ancre POS, mot
intercalé, cas ambiguous_alignment construit à la main."""

from __future__ import annotations

import unittest

from pipeline import mwe, mwe_alignment


class RealPatternAlignmentTests(unittest.TestCase):
    """Rejeu contre le VRAI matcher de production (pipeline.mwe.get_matcher,
    idiomatch n=2 + idiomes custom) — mêmes patterns que ceux utilisés par
    mwe.py en production, pas une doublure synthétique."""

    @classmethod
    def setUpClass(cls):
        cls.matcher = mwe.get_matcher()

    def test_pos_anchor_is_a_real_member_not_noise(self):
        # "let someone go" -> ANCHOR{LEMMA=let} ~slop ANCHOR{POS=PRON}
        # ~slop ANCHOR{LEMMA=go}. Dans "let him go", "him" valide l'ancre
        # POS=PRON : c'est un membre réel de l'idiome, pas du remplissage
        # à écarter (l'heuristique initiale, fausse, aurait ignoré tout
        # pronom — voir le plan, Partie 2 point D).
        doc = self.matcher.nlp("let him go")
        tokens = list(doc)
        result = mwe_alignment.align_members("let someone go", tokens, self.matcher.nlp, self.matcher.n)
        self.assertFalse(result.ambiguous)
        self.assertEqual(result.member_indices, frozenset({0, 1, 2}))

    def test_intercalated_word_is_excluded_from_members(self):
        # "turn off" -> ANCHOR{LEMMA=turn} ~slop{0,2} ANCHOR{LEMMA=off}.
        # Dans "turn the light off", "the light" tombe dans le trou slop :
        # ce sont eux, le mot intercalé du défaut A ("turned the lantern
        # off" avalait "lantern" via l'enveloppe) — jamais des membres.
        doc = self.matcher.nlp("turn the light off")
        tokens = list(doc)
        result = mwe_alignment.align_members("turn off", tokens, self.matcher.nlp, self.matcher.n)
        self.assertFalse(result.ambiguous)
        self.assertEqual(result.member_indices, frozenset({0, 3}))  # "turn", "off" — pas "the light"


class _FakeToken:
    """Duck-type minimal pour _spec_matches_token (lemma_/tag_/pos_/text
    seulement) — évite de dépendre de spaCy pour un cas construit à la
    main, purement structurel."""

    def __init__(self, text: str, lemma: str | None = None, pos: str = "X", tag: str = "X"):
        self.text = text
        self.lemma_ = lemma if lemma is not None else text
        self.pos_ = pos
        self.tag_ = tag


class HandBuiltAmbiguousAlignmentTests(unittest.TestCase):
    """Cas construit à la main (plan, Lot 1) : un motif à deux alternatives
    structurelles également valides sur le même span doit être exclu de la
    réservation (`ambiguous=True`), jamais résolu par un choix arbitraire.

    Motif : ANCHOR{LEMMA=a, OP=?} ~ WILDCARD{OP=?}, span = un seul token de
    texte "a". Deux lectures valident TOUTES DEUX le motif en entier :
    - l'ancre consomme "a" (membre), le wildcard ne consomme rien ;
    - l'ancre ne consomme rien, le wildcard consomme "a" (remplissage).
    Ces deux lectures produisent deux ENSEMBLES DE MEMBRES différents
    ({0} contre ∅) — c'est exactement la définition d'`ambiguous_alignment`
    (align_members ne s'appuie que sur le nombre d'ensembles DISTINCTS,
    pas sur le nombre de dérivations)."""

    def test_two_distinct_member_sets_mark_the_span_ambiguous(self):
        from idiomatch.configs import WILDCARD

        spec_list = [
            {"LEMMA": {"REGEX": "^a$"}, "OP": "?"},
            {"TEXT": {"REGEX": WILDCARD}, "OP": "?"},
        ]
        tokens = [_FakeToken("a")]
        solutions = mwe_alignment._all_alignments(spec_list, tokens)
        self.assertEqual(solutions, {frozenset(), frozenset({0})})

    def test_align_members_abstains_on_the_same_pattern(self):
        # Même motif, mais via align_members (l'API publique utilisée par
        # mwe.py) : passe par le cache de patterns custom, jamais par
        # patterns_for_idiom (pas un idiome réel) — on injecte directement
        # le motif dans le cache pour rester au niveau "hand-built".
        from idiomatch.configs import WILDCARD

        idiom = "__test_ambiguous_hand_built__"
        spec_list = [
            {"LEMMA": {"REGEX": "^a$"}, "OP": "?"},
            {"TEXT": {"REGEX": WILDCARD}, "OP": "?"},
        ]
        mwe_alignment._CUSTOM_PATTERN_CACHE[(idiom, 2)] = [spec_list]
        try:
            result = mwe_alignment.align_members(idiom, [_FakeToken("a")], nlp=None, n=2)
        finally:
            del mwe_alignment._CUSTOM_PATTERN_CACHE[(idiom, 2)]
        self.assertTrue(result.ambiguous)
        self.assertIsNone(result.member_indices)


if __name__ == "__main__":
    unittest.main()
