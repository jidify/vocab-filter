"""Portes S2 (`pipeline/mwe_gates.py`) — arrêter le bruit idiomatch avant
S3, voir `fix_pipeline/s2_fix/bug_idiomatch_slot_overmatch.md`. Contre le
vrai matcher de production (`mwe.get_matcher()`), sur le modèle de
`test_mwe_alignment.py` — mêmes patterns qu'en production, pas une doublure
synthétique."""

from __future__ import annotations

import unittest

from pipeline import mwe, mwe_gates


def _candidates(matcher, text: str, idiom: str | None = None) -> list[dict]:
    """Rejoue `find_candidates` sur un segment isolé, filtré sur `idiom`
    quand fourni. Utilise directement `matcher()`/`mwe_gates.classify`
    plutôt que `mwe.find_candidates` (qui attend des objets Segment) pour
    rester un test unitaire de la porte, pas de tout S2."""

    doc = matcher.nlp(text)
    out = []
    for m in matcher(doc):
        if idiom is not None and m["idiom"] != idiom:
            continue
        _, start, end = m["meta"]
        span = doc[start:end]
        rejected_by = mwe_gates.classify(m["idiom"], list(span), matcher.nlp, matcher.n)
        out.append({"idiom": m["idiom"], "surface": span.text, "rejected_by": rejected_by})
    return out


class SlotGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = mwe.get_matcher()

    def _kept(self, text: str, idiom: str) -> bool:
        hits = _candidates(self.matcher, text, idiom)
        return any(h["rejected_by"] is None for h in hits)

    def test_intransitive_know_is_rejected(self):
        for text in ["I know.", "You know, I can go.", "She doesn't know.",
                     "I know, I know."]:
            with self.subTest(text=text):
                self.assertFalse(self._kept(text, "know someone"))

    def test_generic_passive_know_is_rejected(self):
        # "it is not known" -- une seule ancre lexicale (know) : le slot en
        # est l'unique argument, indiscernable d'un passif littéral
        # générique. Cas discriminant de R1 (voir mwe_gates._passive_alternative_ok).
        self.assertFalse(self._kept("It is not known.", "know someone"))

    def test_transitive_know_is_kept(self):
        self.assertTrue(self._kept("Do you know her?", "know someone"))

    def test_let_someone_go_family_is_kept(self):
        for text in ["Let it go.", "let him go.", "let me go."]:
            with self.subTest(text=text):
                self.assertTrue(self._kept(text, "let someone go"))

    def test_roll_ones_eyes_is_kept(self):
        self.assertTrue(self._kept("roll my eyes.", "roll one's eyes"))

    def test_on_someones_mind_is_kept(self):
        self.assertTrue(self._kept("It's on her mind.", "on someone's mind"))

    def test_went_out_of_their_way_is_kept(self):
        self.assertTrue(self._kept("They went out of their way.", "go out of one's way"))

    def test_real_passive_with_two_anchors_is_kept(self):
        # "shut one's mouth" (shut + mouth, 2 ancres) : le passif réel doit
        # être admis -- contrairement à "it is not known" (1 ancre).
        for text in ["My mouth is shut.", "how is your mouth shut?"]:
            with self.subTest(text=text):
                self.assertTrue(self._kept(text, "shut one's mouth"))


class GrammaticalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = mwe.get_matcher()

    def _kept(self, text: str, idiom: str) -> bool:
        hits = _candidates(self.matcher, text, idiom)
        return any(h["rejected_by"] is None for h in hits)

    def test_auxiliary_do_is_rejected(self):
        for text, idioms in [
            ("I've done a lot.", {"I do"}),
            ("I know you did.", {"I do"}),
            ("Did you see it?", {"do it"}),
            ("I don't have a dog.", {"do a"}),
        ]:
            with self.subTest(text=text):
                hits = [h for h in _candidates(self.matcher, text) if h["idiom"] in idioms]
                self.assertTrue(hits, f"aucun candidat {idioms} sur {text!r}")
                self.assertTrue(all(h["rejected_by"] is not None for h in hits))

    def test_contiguous_light_verb_expressions_are_kept(self):
        self.assertTrue(self._kept("How are you?", "how are you"))
        self.assertTrue(self._kept("You guys are late.", "you guys"))
        self.assertTrue(self._kept("or something", "or something"))
        self.assertTrue(self._kept("It's up there.", "up there"))
        self.assertTrue(self._kept("kind of", "kind of"))
        self.assertTrue(self._kept("at all", "at all"))

    def test_phrasal_verb_shaped_entry_is_not_captured_by_this_gate(self):
        # "do in" a la forme [verbe léger][particule] : structurellement
        # indiscernable d'un vrai phrasal verb séparable ("did him in").
        # Reste hors de cette porte, quel que soit le rejected_by observé.
        hits = _candidates(self.matcher, "They did him in.", "do in")
        if hits:
            self.assertNotIn("grammatical_gap", [h["rejected_by"] for h in hits])
            self.assertNotIn("grammatical_auxiliary", [h["rejected_by"] for h in hits])


class CorruptAnchorRepairTests(unittest.TestCase):
    """Porte D — appliquée au chargement du matcher (mwe.get_matcher), pas
    via classify() : le motif buggé ne doit jamais être proposé comme
    candidat."""

    @classmethod
    def setUpClass(cls):
        cls.matcher = mwe.get_matcher()

    def test_false_we_it_matches_are_gone(self):
        for text in ["We all love it.", "We get it.", "We think it."]:
            with self.subTest(text=text):
                hits = [h for h in _candidates(self.matcher, text) if h["idiom"] == "wing it"]
                self.assertEqual(hits, [])

    def test_real_idiom_is_now_detectable(self):
        for text in ["I'm just winging it.", "He winged it."]:
            with self.subTest(text=text):
                hits = [h for h in _candidates(self.matcher, text) if h["idiom"] == "wing it"]
                self.assertTrue(hits)


class BookLevelAccountingTests(unittest.TestCase):
    """Invariant comptable : validés + rejetés == occurrences brutes, à la
    granularité occurrence, sur le livre réel."""

    def test_accepted_plus_rejected_equals_raw(self):
        from pipeline.corpus import load_segments
        from pipeline.mwe import find_candidates

        segments = load_segments()
        raw = list(find_candidates(segments))
        accepted = [c for c in raw if c["rejected_by"] is None]
        rejected = [c for c in raw if c["rejected_by"] is not None]
        self.assertEqual(len(accepted) + len(rejected), len(raw))
        self.assertTrue(all(
            r["rejected_by"] in ("slot_saturation", "grammatical_gap", "grammatical_auxiliary")
            for r in rejected
        ))

    def test_known_gates_named_in_the_action_plan_survive(self):
        # Correction S2-1/S1-3/S3-2/S7-4 : burn out et les phrasal verbs
        # que seul idiomatch voit dans ce livre ne doivent jamais être
        # rejetés par les portes S2 (hors périmètre par construction).
        from pipeline.corpus import load_segments
        from pipeline.mwe import find_candidates

        segments = load_segments()
        raw = list(find_candidates(segments))
        untouched = {"burn out", "let it go", "figure out", "grow up",
                     "talk about", "clean up", "crack open", "smart ass"}
        seen = {c["idiom"] for c in raw}
        for idiom in untouched & seen:
            with self.subTest(idiom=idiom):
                self.assertTrue(
                    any(c["rejected_by"] is None for c in raw if c["idiom"] == idiom),
                    f"{idiom!r} entièrement rejeté par les portes S2",
                )


if __name__ == "__main__":
    unittest.main()
