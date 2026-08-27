from __future__ import annotations

import unittest

from pipeline.purge_unit import (
    filter_csv_rows,
    filter_occurrence_lines,
    filter_types_entries,
    fully_removed_types,
    split_senses_by_key,
)


class SplitSensesByKeyTests(unittest.TestCase):
    def test_removes_only_matching_best_sense(self):
        entries = [
            ({"word": "york", "pos": "n", "best_sense": "york.n.01"}, "L1"),
            ({"word": "york", "pos": "n", "best_sense": "york.n.01"}, "L2"),
            ({"word": "queens", "pos": "n", "best_sense": "queens.n.01"}, "L3"),
        ]
        kept, removed_wp, remaining_wp = split_senses_by_key(entries, "york.n.01")
        self.assertEqual(kept, ["L3"])
        self.assertEqual(removed_wp, {("york", "n")})
        self.assertEqual(remaining_wp, {("queens", "n")})

    def test_keeps_unreadable_lines_raw_never_counted(self):
        # Une ligne illisible (occ=None, voir _read_jsonl_tolerant) est
        # toujours conservée, jamais comptée dans les ensembles retirés
        # ni survivants — on ne devine jamais ce qu'elle visait.
        entries = [
            (None, "{corrupted"),
            ({"word": "york", "pos": "n", "best_sense": "york.n.01"}, "L2"),
        ]
        kept, removed_wp, remaining_wp = split_senses_by_key(entries, "york.n.01")
        self.assertEqual(kept, ["{corrupted"])
        self.assertEqual(removed_wp, {("york", "n")})
        self.assertEqual(remaining_wp, set())

    def test_a_surviving_unresolved_occurrence_keeps_the_word_pos_alive(self):
        # Cas réel : "york" a 13 occurrences dans senses.jsonl, 12 vers
        # york.n.01 et 1 "aucun_sens_adapte" — cette dernière doit rester
        # dans remaining_word_pos pour empêcher la suppression du type.
        entries = [
            ({"word": "york", "pos": "n", "best_sense": "york.n.01"}, "L1"),
            ({"word": "york", "pos": "n", "best_sense": "aucun_sens_adapte"}, "L2"),
        ]
        kept, removed_wp, remaining_wp = split_senses_by_key(entries, "york.n.01")
        self.assertEqual(kept, ["L2"])
        self.assertEqual(removed_wp, {("york", "n")})
        self.assertEqual(remaining_wp, {("york", "n")})
        self.assertEqual(fully_removed_types(removed_wp, remaining_wp), set())


class FullyRemovedTypesTests(unittest.TestCase):
    def test_only_types_with_zero_survivors_are_fully_removed(self):
        removed_wp = {("york", "n"), ("beat", "n")}
        remaining_wp = {("beat", "n")}  # "beat" a un autre sens survivant
        self.assertEqual(fully_removed_types(removed_wp, remaining_wp), {("york", "n")})


class FilterTypesAndOccurrencesTests(unittest.TestCase):
    def test_filter_types_entries_drops_only_fully_removed_pairs(self):
        types = [
            {"lemma": "york", "wn_pos": "n"},
            {"lemma": "beat", "wn_pos": "n"},
        ]
        kept = filter_types_entries(types, {("york", "n")})
        self.assertEqual(kept, [{"lemma": "beat", "wn_pos": "n"}])

    def test_filter_occurrence_lines_keeps_unreadable_and_unaffected(self):
        entries = [
            ({"lemma": "york", "wn_pos": "n"}, "L1"),
            ({"lemma": "beat", "wn_pos": "n"}, "L2"),
            (None, "{corrupted"),
        ]
        kept = filter_occurrence_lines(entries, {("york", "n")})
        self.assertEqual(kept, ["L2", "{corrupted"])


class FilterCsvRowsTests(unittest.TestCase):
    def test_never_filters_on_substring_only_on_key_column(self):
        rows = [
            {"key": "york.n.01", "contexte_en": "irrelevant"},
            {"key": "queens.n.01", "contexte_en": "I'm not opening anything until New York"},
            {"key": "broadway.n.01", "contexte_en": "New York City"},
        ]
        kept = filter_csv_rows(rows, "york.n.01")
        self.assertEqual([r["key"] for r in kept], ["queens.n.01", "broadway.n.01"])


if __name__ == "__main__":
    unittest.main()
