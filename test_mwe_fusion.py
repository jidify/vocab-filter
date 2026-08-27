"""Lot 3 — fusion idiomatch/VPC, magasins MWE à deux niveaux, gel de
l'inventaire (voir le plan, Partie 4, Lot 3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import config, inventory, mwe, mwe_stores
from pipeline.analyze import _offset_char_spans


class OffsetCharSpansTests(unittest.TestCase):
    """analyze.py convertit les spans d'un PhrasalVerbDetection sérialisé,
    relatifs au début de PHRASE (pipeline/vpc/adapter.py), en absolu dans
    le SEGMENT — le référentiel utilisé partout ailleurs (occurrences.jsonl,
    mwe_candidates.jsonl)."""

    def test_offsets_all_span_fields_by_sentence_start(self):
        record = {
            "verb_char_span": [3, 8],
            "particle_char_spans": [[9, 11]],
            "token_char_spans": [[3, 8], [9, 11]],
            "normalized_char_spans": None,
            "original_char_spans": [[3, 8], [9, 11]],
        }
        _offset_char_spans(record, offset=95)
        self.assertEqual(record["verb_char_span"], [98, 103])
        self.assertEqual(record["particle_char_spans"], [[104, 106]])
        self.assertEqual(record["token_char_spans"], [[98, 103], [104, 106]])
        self.assertIsNone(record["normalized_char_spans"])
        self.assertEqual(record["original_char_spans"], [[98, 103], [104, 106]])


def _idiomatch_candidate(occurrence_id, idiom, segment_idx=1, start_char=0, end_char=10):
    return {
        "occurrence_id": occurrence_id,
        "segment_idx": segment_idx,
        "kind": "dialogue",
        "idiom": idiom,
        "surface": idiom,
        "start_token": 0,
        "end_token": 2,
        "start_char": start_char,
        "end_char": end_char,
        "n_tokens_span": 2,
        "n_tokens_lemma": 2,
        "member_char_spans": [[start_char, start_char + 4], [end_char - 3, end_char]],
        "ambiguous_alignment": False,
        "source": "idiomatch",
        "directional_context_dependent": False,
    }


def _vpc_candidate(occurrence_id, idiom, segment_idx=1, start_char=0, end_char=10,
                    directional=False):
    return {
        "occurrence_id": occurrence_id,
        "segment_idx": segment_idx,
        "kind": "vpc",
        "idiom": idiom,
        "surface": idiom,
        "start_token": None,
        "end_token": None,
        "start_char": start_char,
        "end_char": end_char,
        "n_tokens_span": 2,
        "n_tokens_lemma": 2,
        "member_char_spans": [[start_char, start_char + 4], [end_char - 3, end_char]],
        "ambiguous_alignment": False,
        "source": "vpc",
        "directional_context_dependent": directional,
        "vpc_decision": "matched_reference",
        "vpc_decision_reason": "test",
    }


class MergeCandidateSourcesTests(unittest.TestCase):
    def test_disjoint_occurrences_are_all_kept(self):
        idiomatch = [_idiomatch_candidate("m:1:0:10", "turn off")]
        vpc = [_vpc_candidate("m:1:20:30", "wake up")]
        merged = mwe.merge_candidate_sources(idiomatch, vpc)
        self.assertEqual({c["occurrence_id"] for c in merged}, {"m:1:0:10", "m:1:20:30"})

    def test_same_occurrence_id_idiomatch_wins(self):
        idiomatch = [_idiomatch_candidate("m:1:0:10", "turn off")]
        vpc = [_vpc_candidate("m:1:0:10", "turn off")]
        merged = mwe.merge_candidate_sources(idiomatch, vpc)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "idiomatch")

    def test_directional_flag_survives_even_when_idiomatch_wins(self):
        idiomatch = [_idiomatch_candidate("m:1:0:10", "walk up")]
        vpc = [_vpc_candidate("m:1:0:10", "walk up", directional=True)]
        merged = mwe.merge_candidate_sources(idiomatch, vpc)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "idiomatch")
        self.assertTrue(merged[0]["directional_context_dependent"])

    def test_vpc_only_occurrence_keeps_its_own_directional_flag(self):
        merged = mwe.merge_candidate_sources([], [_vpc_candidate("m:1:0:10", "wake up", directional=True)])
        self.assertTrue(merged[0]["directional_context_dependent"])


class MweStoresProtectionTests(unittest.TestCase):
    def test_is_protected_only_for_validated_status(self):
        self.assertFalse(mwe_stores.is_protected(None))
        self.assertFalse(mwe_stores.is_protected({"status": "auto"}))
        self.assertTrue(mwe_stores.is_protected({"status": "validated"}))

    def test_load_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store.jsonl"
            with mock.patch.object(config, "ensure_data_dir", lambda: None):
                store = {"turn off": mwe_stores.build_entry("turn off", {
                    "label": "phrasal_verb", "confidence": 0.9, "reason": "x",
                })}
                mwe_stores._write(path, store)
                reloaded = mwe_stores._load(path)
            self.assertEqual(reloaded["turn off"]["label"], "phrasal_verb")
            self.assertEqual(reloaded["turn off"]["status"], "auto")

    def test_missing_store_file_loads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mwe_stores._load(Path(tmp) / "absent.jsonl"), {})


class SelectMweSpansOccurrenceOverrideTests(unittest.TestCase):
    """mwe_judge.select_mwe_spans : une décision d'occurrence
    (`occ["occurrence_decision"]`) prime sur la décision de type — voir le
    plan, point C."""

    def _entry(self, occ):
        return {
            "idiom": "walk up",
            "label": "phrasal_verb",
            "confidence": 0.9,
            "occurrences": [occ],
        }

    def _occ(self, occurrence_id="m:1:0:7", segment_idx=1, start_char=0, end_char=7, **extra):
        base = {
            "occurrence_id": occurrence_id,
            "segment_idx": segment_idx,
            "start_char": start_char,
            "end_char": end_char,
            "surface": "walked up",
            "n_tokens_span": 2,
            "member_char_spans": [[0, 6], [7, 9]],
            "ambiguous_alignment": False,
        }
        base.update(extra)
        return base

    def test_type_level_decision_used_when_no_override(self):
        from pipeline import mwe_judge
        resolved = mwe_judge.select_mwe_spans([self._entry(self._occ())])
        self.assertEqual(len(resolved[1]), 1)

    def test_occurrence_override_blocks_reservation_despite_lexicalized_type(self):
        from pipeline import mwe_judge
        occ = self._occ(occurrence_decision={"label": "littéral", "confidence": 0.9, "reason": "x"})
        resolved = mwe_judge.select_mwe_spans([self._entry(occ)])
        self.assertEqual(resolved.get(1, []), [])

    def test_occurrence_override_allows_reservation_despite_uncertain_type(self):
        from pipeline import mwe_judge
        entry = self._entry(self._occ(
            occurrence_decision={"label": "phrasal_verb", "confidence": 0.9, "reason": "x"}
        ))
        entry["label"] = "incertain"
        entry["confidence"] = 0.0
        resolved = mwe_judge.select_mwe_spans([entry])
        self.assertEqual(len(resolved[1]), 1)
        self.assertEqual(resolved[1][0]["label"], "phrasal_verb")


class LlmFailureNotCachedTests(unittest.TestCase):
    """Une panne LLM (ollama injoignable, réponse illisible) n'est pas une
    décision — ne doit jamais être écrite dans le magasin permanent (sinon
    un hoquet réseau fige "incertain" pour toujours, contraire au principe
    d'abstention). Bug observé en pratique lors de la vérification de ce
    lot : cf. l'avancement du plan."""

    def test_genuine_llm_error_is_a_failure(self):
        from pipeline import mwe_judge
        decision = {"label": "incertain", "confidence": 0.0,
                    "reason": "LLM indisponible: ollama injoignable (...)"}
        self.assertTrue(mwe_judge.is_llm_failure(decision))

    def test_unparseable_response_is_a_failure(self):
        from pipeline import mwe_judge
        decision = {"label": "incertain", "confidence": 0.0,
                    "reason": "réponse LLM invalide: {}"}
        self.assertTrue(mwe_judge.is_llm_failure(decision))

    def test_genuine_model_verdict_of_incertain_is_not_a_failure(self):
        from pipeline import mwe_judge
        decision = {"label": "incertain", "confidence": 0.3,
                    "reason": "Le contexte ne permet pas de trancher."}
        self.assertFalse(mwe_judge.is_llm_failure(decision))

    def test_genuine_lexicalized_verdict_is_not_a_failure(self):
        from pipeline import mwe_judge
        decision = {"label": "phrasal_verb", "confidence": 0.9, "reason": "x"}
        self.assertFalse(mwe_judge.is_llm_failure(decision))


class InventoryHashTests(unittest.TestCase):
    def test_hash_is_order_independent(self):
        rows_a = [
            {"occurrence_id": "w:1:0", "unit_key": "walk:v"},
            {"occurrence_id": "m:1:0:7", "unit_key": "mwe:walk up:phrasal_verb"},
        ]
        rows_b = list(reversed(rows_a))
        self.assertEqual(inventory.compute_hash(rows_a), inventory.compute_hash(rows_b))

    def test_hash_changes_when_inventory_changes(self):
        rows = [{"occurrence_id": "w:1:0", "unit_key": "walk:v"}]
        other = [{"occurrence_id": "w:1:0", "unit_key": "walk:n"}]
        self.assertNotEqual(inventory.compute_hash(rows), inventory.compute_hash(other))

    def test_current_hash_raises_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(config, "INVENTORY_HASH_PATH", Path(tmp) / "absent.sha256"):
                with self.assertRaises(SystemExit):
                    inventory.current_hash("test")

    def test_verify_consumer_raises_on_stale_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inv_hash_path = tmp_path / "inventory.sha256"
            inv_hash_path.write_text("aaa\n", encoding="utf-8")
            consumer_path = tmp_path / "senses.inventory.sha256"
            consumer_path.write_text("bbb\n", encoding="utf-8")
            with mock.patch.object(config, "INVENTORY_HASH_PATH", inv_hash_path):
                with self.assertRaises(SystemExit):
                    inventory.verify_consumer(consumer_path, "test")

    def test_verify_consumer_passes_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inv_hash_path = tmp_path / "inventory.sha256"
            inv_hash_path.write_text("aaa\n", encoding="utf-8")
            consumer_path = tmp_path / "senses.inventory.sha256"
            consumer_path.write_text("aaa\n", encoding="utf-8")
            with mock.patch.object(config, "INVENTORY_HASH_PATH", inv_hash_path):
                digest = inventory.verify_consumer(consumer_path, "test")
            self.assertEqual(digest, "aaa")


if __name__ == "__main__":
    unittest.main()
