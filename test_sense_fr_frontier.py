from __future__ import annotations

import unittest

from pipeline.sense_fr_frontier import SenseTranslation, _format_item, _target_payload, build_entry


def _translation(**overrides) -> SenseTranslation:
    base = dict(
        sense_id="k", fr=["mot"], translation_type="equivalence_directe",
        sense_fit="ok", sense_fit_note="",
        definition_fr_fit="ok", definition_fr_fit_note="",
        source="reecrit", confidence="high",
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

    def test_definition_fr_contradiction_never_locks_even_with_sense_fit_ok(self):
        """Cas réel mesuré AVANT ce champ : "bring up" défini "amener d'une
        position basse à une position haute" mais traduit "évoquer",
        sense_fit="ok" (l'usage collait bien à UN sens de bring up, mais pas
        à CETTE définition ni à cette traduction) — verrouillé auto_llm."""
        target = _mwe_target(
            key="mwe:bring up:phrasal_verb",
            definition_en="To bring from a lower to a higher position.",
        )
        translation = _translation(
            sense_id=target["key"], fr=["évoquer"], sense_fit="ok",
            definition_fr_fit="contradiction",
        )
        entry = build_entry(target, translation)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["agreement"], "definition_fr_contradiction")
        self.assertEqual(entry["definition_fr_fit"], "contradiction")

    def test_definition_needs_review_blocks_even_when_model_says_ok(self):
        """Axe déterministe (plan §6 : "définition validée"), calculé en
        amont par S3-3/S4 — bloque indépendamment de ce que le modèle
        déclare sur sense_fit/definition_fr_fit."""
        target = _mwe_target(definition_needs_review=True)
        translation = _translation(sense_id=target["key"], fr=["tomber en panne"])
        entry = build_entry(target, translation)
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["agreement"], "definition_non_validee")

    def test_coherent_translation_locks_and_persists_new_fields(self):
        target = _mwe_target()
        translation = _translation(sense_id=target["key"], fr=["tomber en panne"])
        entry = build_entry(target, translation)
        self.assertEqual(entry["definition_fr_fit"], "ok")
        self.assertEqual(entry["definition_needs_review"], False)


class FormatItemEvidenceTests(unittest.TestCase):
    def test_french_sentence_included_when_present(self):
        target = _mwe_target()
        occ = {"context": "He gave out sandwiches.", "target_surface": "gave out",
               "french": "Il a distribué des sandwiches."}
        text = _format_item(target, [occ], [])
        self.assertIn("Il a distribué des sandwiches.", text)

    def test_french_sentence_omitted_when_absent(self):
        target = _mwe_target()
        occ = {"context": "He gave out sandwiches.", "target_surface": "gave out"}
        text = _format_item(target, [occ], [])
        self.assertNotIn("traduction officielle de cette phrase", text)

    def test_unreliable_definition_flagged_in_prompt(self):
        target = _mwe_target(definition_needs_review=True)
        text = _format_item(target, [], [])
        self.assertIn("n'a pas été validée en amont", text)

    def test_reliable_definition_not_flagged(self):
        target = _mwe_target(definition_needs_review=False)
        text = _format_item(target, [], [])
        self.assertNotIn("n'a pas été validée en amont", text)


class TargetPayloadCacheKeyTests(unittest.TestCase):
    def test_french_sentence_changes_payload(self):
        target = _mwe_target()
        occ_without = {"context": "c", "target_surface": "give out"}
        occ_with = {"context": "c", "target_surface": "give out", "french": "traduction"}
        self.assertNotEqual(
            _target_payload(target, [occ_without], []),
            _target_payload(target, [occ_with], []),
        )

    def test_definition_needs_review_changes_payload(self):
        target = _mwe_target(definition_needs_review=False)
        target_flagged = _mwe_target(definition_needs_review=True)
        self.assertNotEqual(
            _target_payload(target, [], []),
            _target_payload(target_flagged, [], []),
        )


if __name__ == "__main__":
    unittest.main()
