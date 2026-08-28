"""Q0-3 Phase 6 (fix_pipeline/detection_benchmark/phase6_decision.md) :
pipeline/rules_plus.py, port en production du prototype de benchmark
(fix_pipeline/detection_benchmark/rules_plus.py). Reproduit ici en test de
non-régression les deux affinements de rappel déjà mesurés dans
fix_pipeline/detection_benchmark/phase3_rules_plus_report.md."""

from __future__ import annotations

import unittest

from pipeline import analyze, rules_plus


class ScanPhrasalVerbCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = analyze.get_nlp()
        cls.lexicon = rules_plus.merged_phrasal_verb_lexicon()

    def test_coordination_noise_does_not_abort_the_search(self):
        """en_core_web_sm étiquette "pan" VERB/conj vers "puts" (au lieu de
        NOUN/conj vers "blanket") dans ce segment — sans le garde-fou
        _is_coordination_noise, la recherche de "down" avorterait à tort."""
        text = "Erik puts the blanket and pan down; he searches for the lantern."
        doc = self.nlp(text)
        candidates = rules_plus.scan_phrasal_verb_candidates(doc, 1, "dialogue", self.lexicon)
        by_surface = {c["surface"]: c for c in candidates}
        self.assertIn("puts the blanket and pan down", by_surface)
        cand = by_surface["puts the blanket and pan down"]
        self.assertEqual(cand["idiom"], "put down")
        self.assertEqual(cand["category"], "phrasal_verb_separable")
        self.assertFalse(cand["ambiguous_alignment"])
        self.assertFalse(cand["directional_context_dependent"])

    def test_genuine_second_clause_still_aborts_the_search(self):
        """Un vrai second verbe fléchi, dans une proposition subordonnée
        (dep_="advcl", PAS "conj" vers le verbe scanné) doit toujours
        interrompre la recherche — le garde-fou de coordination n'assouplit
        que le cas précis de tagging erroné (voir
        test_coordination_noise_does_not_abort_the_search), jamais une
        vraie frontière de proposition. Lexique jouet : force le test du
        garde-fou indépendamment de la couverture PARSEME/WordNet réelle."""
        text = "He tries hard before she comes down."
        doc = self.nlp(text)
        lexicon = {"try": (("down",),)}
        candidates = rules_plus.scan_phrasal_verb_candidates(doc, 1, "dialogue", lexicon)
        self.assertEqual(candidates, [])


class ScanCustomIdiomCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = analyze.get_nlp()
        from pipeline.mwe import CUSTOM_IDIOMS
        cls.sequences = rules_plus.custom_idiom_sequences(CUSTOM_IDIOMS)

    def test_bridges_a_three_word_interposed_object_beyond_idiomatch_slop_2(self):
        """idiomatch tourne à slop=2 (pipeline/mwe.py::get_matcher, "n=2") :
        il ne peut jamais relier "crack open" à travers un objet interposé
        de 3 mots ("the bathroom door"). Le cas canonique cité dans le
        plan lui-même."""
        text = "Brigid cracks the bathroom door open, hands Aimee the toilet paper."
        doc = self.nlp(text)
        candidates = rules_plus.scan_custom_idiom_candidates(doc, 1, "dialogue", self.sequences)
        by_surface = {c["surface"]: c for c in candidates}
        self.assertIn("cracks the bathroom door open", by_surface)
        self.assertEqual(by_surface["cracks the bathroom door open"]["idiom"], "crack open")

    def test_no_match_when_verb_absent(self):
        text = "She opened the window quietly."
        doc = self.nlp(text)
        candidates = rules_plus.scan_custom_idiom_candidates(doc, 1, "dialogue", self.sequences)
        self.assertEqual(candidates, [])


class ScanWordnetNominalCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = analyze.get_nlp()

    def test_matches_a_known_wordnet_compound(self):
        text = "She works at a nursing home downtown."
        doc = self.nlp(text)
        lexicon = rules_plus.wordnet_nominal_lexicon()
        candidates = rules_plus.scan_wordnet_nominal_candidates(doc, 1, "dialogue", lexicon)
        surfaces = {c["surface"] for c in candidates}
        self.assertIn("nursing home", surfaces)

    def test_never_crosses_a_dialogue_dash(self):
        """Motif "?—" -- le span ne doit jamais enjamber ce genre de
        ponctuation, même si un n-gramme de lemmes matcherait sinon."""
        self.assertTrue(rules_plus.crosses_hard_boundary("stomps around?—we", 0, 18))
        self.assertFalse(rules_plus.crosses_hard_boundary("stomps around", 0, 14))


class HyphenAndPossessiveBoundaryRulesTests(unittest.TestCase):
    def test_hyphen_chain_alone(self):
        candidates = rules_plus.hyphen_chain_candidates(1, "smart-ass remark")
        self.assertEqual([c["surface"] for c in candidates], ["smart-ass"])

    def test_hyphen_extend_walks_left_across_the_hyphen(self):
        text = "ground-floor apartment"
        rows = [{"segment_idx": 1, "surface": "floor apartment", "start_char": 7, "end_char": 23}]
        out = rules_plus.hyphen_extend_existing(rows, {1: text})
        self.assertEqual(out[0]["surface"], "ground-floor apartment")
        self.assertEqual(out[0]["start_char"], 0)

    def test_possessive_trim_stops_before_the_apostrophe_s(self):
        text = "New York City’s Chinatown"
        rows = [{"segment_idx": 1, "surface": "New York City’s", "start_char": 0, "end_char": 15}]
        out = rules_plus.possessive_trim_existing(rows, {1: text})
        self.assertEqual(out[0]["surface"], "New York City")
        self.assertEqual(out[0]["end_char"], 13)


if __name__ == "__main__":
    unittest.main()
