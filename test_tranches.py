"""Lot 6 — reprise par tranches (plan Partie 3, Partie 4 Lot 6).

`analyze_occurrence`/`load_segments` sont court-circuités ici (aucun spaCy,
aucun GlossBERT) : ces tests ne portent QUE sur la logique de fusion par
`occurrence_id` et le filtrage par tranche de `pipeline/senses.py::run()`,
sur `run_pipeline.py::parse_tranches` et sur
`pipeline/zones.py::segment_idxs_for_tranches` — pas sur la désambiguïsation
elle-même (déjà couverte ailleurs) ni sur un vrai layout de zones (déjà
couvert par test_zones.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import config, inventory, sense_fr_frontier, senses, zones
from run_pipeline import parse_tranches


class ParseTranchesTests(unittest.TestCase):
    def test_all_means_no_filter(self):
        self.assertIsNone(parse_tranches("all"))
        self.assertIsNone(parse_tranches("ALL"))

    def test_single_range(self):
        self.assertEqual(parse_tranches("1-3"), {1, 2, 3})

    def test_comma_list(self):
        self.assertEqual(parse_tranches("1,4,7"), {1, 4, 7})

    def test_combined_ranges_and_singles(self):
        self.assertEqual(parse_tranches("1-3,7,10-12"), {1, 2, 3, 7, 10, 11, 12})

    def test_reversed_range_is_normalized(self):
        self.assertEqual(parse_tranches("5-3"), {3, 4, 5})

    def test_garbage_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_tranches("abc")


class SegmentIdxsForTranchesTests(unittest.TestCase):
    def test_selects_only_requested_zone_ordinals(self):
        layout = {
            "segment_zone_map": {
                "0": "zone-01", "1": "zone-01", "2": "zone-02", "3": "zone-03",
            }
        }
        self.assertEqual(zones.segment_idxs_for_tranches(layout, {1}), {0, 1})
        self.assertEqual(zones.segment_idxs_for_tranches(layout, {2, 3}), {2, 3})
        self.assertEqual(zones.segment_idxs_for_tranches(layout, {99}), set())


def _inventory_row(occurrence_id, unit_key, segment_idx, zone_id):
    return {
        "occurrence_id": occurrence_id, "unit_key": unit_key,
        "segment_idx": segment_idx, "start_char": 0, "end_char": 1,
        "zone_id": zone_id, "touched_zone_ids": [zone_id],
    }


class SensesTrancheMergeTests(unittest.TestCase):
    """Rejeu de `senses.run()` contre des artefacts minimaux — un seul type
    ("walk"/v) avec deux occurrences dans deux zones différentes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp_path = Path(self.tmp.name)

        self.rows = [
            _inventory_row("w:1:0", "walk:v", 1, "zone-01"),
            _inventory_row("w:2:0", "walk:v", 2, "zone-02"),
        ]
        self.digest = inventory.compute_hash(self.rows)

        selected_types_path = tmp_path / "selected_types.jsonl"
        selected_types_path.write_text(
            json.dumps({"lemma": "walk", "wn_pos": "v", "zipf": 4.0, "book_count": 2}) + "\n",
            encoding="utf-8",
        )
        inventory_path = tmp_path / "lexical_inventory.jsonl"
        with inventory_path.open("w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        inventory_hash_path = tmp_path / "inventory.sha256"
        inventory_hash_path.write_text(self.digest + "\n", encoding="utf-8")
        self.senses_path = tmp_path / "senses.jsonl"
        senses_inventory_hash_path = tmp_path / "senses.inventory.sha256"

        patches = [
            mock.patch.object(config, "SELECTED_TYPES_PATH", selected_types_path),
            mock.patch.object(config, "LEXICAL_INVENTORY_PATH", inventory_path),
            mock.patch.object(config, "INVENTORY_HASH_PATH", inventory_hash_path),
            mock.patch.object(config, "SENSES_PATH", self.senses_path),
            mock.patch.object(config, "SENSES_INVENTORY_HASH_PATH", senses_inventory_hash_path),
            mock.patch.object(config, "ensure_out_dir", lambda: None),
            mock.patch.object(senses, "load_segments", lambda: []),
        ]

        self.calls: list[tuple[str, str, int]] = []

        def fake_analyze_occurrence(word, pos, segments, seg_idx,
                                    allow_arbitration=True, target_surface=None,
                                    **_resolution_signals):
            self.calls.append((word, pos, seg_idx))
            return {
                "word": word, "pos": pos, "segment_idx": seg_idx,
                "best_sense": f"{word}.{pos}.01", "candidates": [], "needs_review": False,
            }

        patches.append(mock.patch.object(senses, "analyze_occurrence", fake_analyze_occurrence))

        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _read_senses(self) -> list[dict]:
        if not self.senses_path.exists():
            return []
        return [json.loads(l) for l in self.senses_path.read_text(encoding="utf-8").splitlines() if l]

    def test_first_tranche_only_computes_requested_segment(self):
        senses.run(segment_idxs={1})
        records = self._read_senses()
        self.assertEqual({r["occurrence_id"] for r in records}, {"w:1:0"})
        self.assertEqual(self.calls, [("walk", "v", 1)])

    def test_second_tranche_reuses_first_without_recomputing(self):
        senses.run(segment_idxs={1})
        self.calls.clear()
        senses.run(segment_idxs={2})
        records = self._read_senses()
        self.assertEqual({r["occurrence_id"] for r in records}, {"w:1:0", "w:2:0"})
        # w:1:0 est déjà à jour (même inventory_digest) -> jamais recalculé.
        self.assertEqual(self.calls, [("walk", "v", 2)])

    def test_stale_digest_is_recomputed_when_in_scope(self):
        stale = {
            "word": "walk", "pos": "v", "segment_idx": 1, "best_sense": "walk.v.01",
            "candidates": [], "needs_review": False,
            "occurrence_id": "w:1:0", "inventory_digest": "un-ancien-digest",
        }
        self.senses_path.write_text(json.dumps(stale) + "\n", encoding="utf-8")

        senses.run(segment_idxs={1})
        records = self._read_senses()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["inventory_digest"], self.digest)
        self.assertEqual(self.calls, [("walk", "v", 1)])

    def test_pre_lot6_record_without_occurrence_id_is_never_reused(self):
        pre_lot6 = {
            "word": "walk", "pos": "v", "segment_idx": 1, "best_sense": "walk.v.01",
            "candidates": [], "needs_review": False,
        }  # pas d'occurrence_id : format d'avant ce lot.
        self.senses_path.write_text(json.dumps(pre_lot6) + "\n", encoding="utf-8")

        senses.run(segment_idxs={1})
        self.assertEqual(self.calls, [("walk", "v", 1)])

    def test_no_filter_computes_everything(self):
        senses.run(segment_idxs=None)
        records = self._read_senses()
        self.assertEqual({r["occurrence_id"] for r in records}, {"w:1:0", "w:2:0"})


class SenseFrFrontierProtectedFilterTests(unittest.TestCase):
    """Lot 6 (Partie 3, point 31) : une cible déjà `validated`/`auto_joint`
    ne doit jamais atteindre `_translate_units` — donc jamais générer
    d'appel LLM réel, ni même être mise en lot. Vérifié sur le VRAI
    `sense_fr_frontier.run()`, avec seulement `collect_frontier_targets` /
    `sense_fr.load_store` / `senses.load_occurrences_by_sense` /
    `_translate_units` doublés : aucun réseau, aucun coût, aucune écriture
    disque (`dry_run=True`)."""

    def test_protected_target_never_reaches_translate_units(self):
        protected_target = {
            "key": "mwe:validated one:idiome", "kind": "mwe",
            "lemmas_en": ["validated one"], "occurrences": 3,
            "definition_en": "déjà validée à la main",
        }
        pending_target = {
            "key": "mwe:pending one:idiome", "kind": "mwe",
            "lemmas_en": ["pending one"], "occurrences": 2,
            "definition_en": "pas encore traduite",
        }
        store = {"mwe:validated one:idiome": {"status": "validated", "fr": "déjà", "fr_alt": []}}

        seen_items: list = []

        def fake_translate_units(items, model, *, batch_size, mode_batch):
            seen_items.extend(items)
            return {}, 0.0

        with mock.patch.object(
                sense_fr_frontier, "collect_frontier_targets",
                return_value=([protected_target, pending_target], [])), \
             mock.patch.object(sense_fr_frontier.sense_fr, "load_store", return_value=store), \
             mock.patch.object(senses, "load_occurrences_by_sense", return_value={}), \
             mock.patch.object(sense_fr_frontier, "_translate_units",
                                side_effect=fake_translate_units), \
             mock.patch.object(inventory, "verify_consumer", lambda *a, **k: "digest"):
            sense_fr_frontier.run(dry_run=True)

        seen_keys = {t["key"] for t, _occs, _cands in seen_items}
        self.assertNotIn("mwe:validated one:idiome", seen_keys)
        self.assertIn("mwe:pending one:idiome", seen_keys)


class CoverageReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp_path = Path(self.tmp.name)

        inventory_path = tmp_path / "lexical_inventory.jsonl"
        rows = [
            _inventory_row("w:1:0", "walk:v", 1, "zone-01"),
            _inventory_row("w:2:0", "walk:v", 2, "zone-02"),
        ]
        with inventory_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        self.senses_path = tmp_path / "senses.jsonl"
        self.senses_path.write_text(
            json.dumps({"occurrence_id": "w:1:0", "inventory_digest": "x"}) + "\n",
            encoding="utf-8",
        )

        patches = [
            mock.patch.object(config, "LEXICAL_INVENTORY_PATH", inventory_path),
            mock.patch.object(config, "SENSES_PATH", self.senses_path),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_partial_coverage_reports_missing_zone(self):
        report = senses.coverage_report()
        self.assertEqual(report, {"total": 2, "covered": 1, "missing_zone_ids": ["zone-02"]})

    def test_full_coverage_reports_no_missing_zone(self):
        self.senses_path.write_text(
            "\n".join(json.dumps({"occurrence_id": oid, "inventory_digest": "x"})
                      for oid in ("w:1:0", "w:2:0")) + "\n",
            encoding="utf-8",
        )
        report = senses.coverage_report()
        self.assertEqual(report, {"total": 2, "covered": 2, "missing_zone_ids": []})


if __name__ == "__main__":
    unittest.main()
