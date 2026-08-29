from __future__ import annotations

import unittest

from pipeline.sense_fr_frontier import SenseTranslation, build_entry


def _translation(**overrides) -> SenseTranslation:
    base = dict(
        sense_id="k", fr=["mot"], translation_type="equivalence_directe",
        sense_fit="ok", sense_fit_note="", source="reecrit", confidence="high",
    )
    base.update(overrides)
    return SenseTranslation(**base)


def _mwe_target(**overrides) -> dict:
    base = dict(
        key="mwe:give out:phrasal_verb", kind="mwe", pos="v",
        lemmas_en=["give out"], occurrences=1,
        definition_en="To break down, get out of order, fail.",
    )
    base.update(overrides)
    return base


class BuildEntryCoherenceGateTests(unittest.TestCase):
    def test_mismatch_never_locks(self):
        target = _mwe_target()
        translation = _translation(sense_id=target["key"], fr=["tomber en panne"], sense_fit="mismatch")
        entry = build_entry(target, translation)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["agreement"], "sense_id_suspect")

    def test_doubtful_never_locks(self):
        target = _mwe_target()
        translation = _translation(sense_id=target["key"], fr=["tomber en panne"], sense_fit="doubtful")
        entry = build_entry(target, translation)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["agreement"], "sense_id_douteux")

    def test_non_literal_translation_type_never_locks(self):
        target = _mwe_target()
        translation = _translation(
            sense_id=target["key"], fr=["tomber en panne"], translation_type="reformulation",
        )
        entry = build_entry(target, translation)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["agreement"], "frontier_reformulation")

    def test_ok_and_literal_without_resources_locks_as_auto_llm(self):
        target = _mwe_target()  # aucune omw-fr/WoNeF possible pour une MWE
        translation = _translation(sense_id=target["key"], fr=["tomber en panne"])
        entry = build_entry(target, translation)
        self.assertEqual(entry["status"], "auto_llm")
        self.assertEqual(entry["pos"], "v")  # identité complète transmise


if __name__ == "__main__":
    unittest.main()
