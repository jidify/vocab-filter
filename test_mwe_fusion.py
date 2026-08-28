"""Lot 3 — fusion idiomatch/VPC, magasins MWE à deux niveaux, gel de
l'inventaire (voir le plan, Partie 4, Lot 3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import config, inventory, mwe, mwe_stores, select
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

    def test_rules_plus_fills_gaps_but_never_wins_a_collision(self):
        """Q0-3 Phase 6 : rules_plus n'a jamais de pouvoir de rejet NI de
        priorité — il ne comble que ce qu'idiomatch et VPC ont raté."""
        idiomatch = [_idiomatch_candidate("m:1:0:10", "turn off")]
        vpc = [_vpc_candidate("m:1:20:30", "wake up")]
        rules_plus_only = _idiomatch_candidate("m:1:40:50", "figure out")
        rules_plus_only["source"] = "rules_plus_phrasal_verb_scan"
        collision_with_vpc = _idiomatch_candidate("m:1:20:30", "wake up")
        collision_with_vpc["source"] = "rules_plus_phrasal_verb_scan"
        collision_with_vpc["surface"] = "should never win"

        merged = mwe.merge_candidate_sources(
            idiomatch, vpc, [rules_plus_only, collision_with_vpc]
        )
        by_id = {c["occurrence_id"]: c for c in merged}
        self.assertEqual(
            {c["occurrence_id"] for c in merged}, {"m:1:0:10", "m:1:20:30", "m:1:40:50"}
        )
        self.assertEqual(by_id["m:1:40:50"]["source"], "rules_plus_phrasal_verb_scan")
        self.assertEqual(by_id["m:1:20:30"]["source"], "vpc")


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

    def test_type_level_summary_never_substitutes_for_occurrence_decision(self):
        from pipeline import mwe_judge
        resolved = mwe_judge.select_mwe_spans([self._entry(self._occ())])
        self.assertEqual(resolved.get(1, []), [])

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


class OccurrenceFirstDecisionTests(unittest.TestCase):
    def _result(self, label, paraphrase, confidence=.9, evidence=None):
        return {
            "label": label,
            "canonical_form": "let go",
            "pos": "VERB",
            "contextual_paraphrase": paraphrase,
            "confidence": confidence,
            "evidence": evidence or ["contextual substitution changes the meaning"],
            "reason": "fixture",
        }

    def test_same_candidate_type_can_receive_distinct_occurrence_verdicts(self):
        from pipeline import mwe_judge
        replies = [
            self._result("idiome", "stop worrying about it"),
            self._result("phrasal_verb", "release him"),
            self._result("littéral", "depart now"),
        ]
        occurrences = [
            {"occurrence_id": f"m:{i}:0:8", "segment_idx": i, "surface": surface,
             "source": "idiomatch"}
            for i, surface in enumerate(("let it go", "let him go", "let's go"), 1)
        ]
        with mock.patch.object(mwe_judge.llm, "call_json", side_effect=replies):
            decisions = [mwe_judge.judge_occurrence("let go", occ, {}) for occ in occurrences]
        self.assertEqual([d["label"] for d in decisions], ["idiome", "phrasal_verb", "littéral"])
        self.assertEqual(len({d["contextual_paraphrase"] for d in decisions}), 3)

    def test_llm_confidence_without_observable_evidence_is_capped(self):
        from pipeline import mwe_judge
        occurrence = {"occurrence_id": "m:1:0:15", "segment_idx": 1,
                      "surface": "could care less", "source": "rules_plus"}
        reply = self._result("idiome", "could not care less", confidence=1.0, evidence=[])
        # Explicitly retain an empty list (the helper's default would add evidence).
        reply["evidence"] = []
        with mock.patch.object(mwe_judge.llm, "call_json", return_value=reply):
            decision = mwe_judge.judge_occurrence("could care less", occurrence, {})
        self.assertEqual(decision["model_confidence"], 1.0)
        self.assertLess(decision["confidence"], mwe_judge.MIN_CONFIDENCE)
        self.assertNotIn("observable_evidence_present", decision["confidence_features"])

    def test_cache_key_keeps_competing_hypotheses_distinct(self):
        from pipeline import mwe_judge
        self.assertNotEqual(
            mwe_judge.occurrence_store_key("let go", "m:1:0:9"),
            mwe_judge.occurrence_store_key("let it go", "m:1:0:9"),
        )

    def test_cache_key_changes_with_protocol_model_and_context(self):
        from pipeline import mwe_judge
        base = mwe_judge.occurrence_store_key(
            "could care less", "m:1", model="local", context_signature="context-a"
        )
        self.assertNotEqual(base, mwe_judge.occurrence_store_key(
            "could care less", "m:1", model="frontier", context_signature="context-a"
        ))
        self.assertNotEqual(base, mwe_judge.occurrence_store_key(
            "could care less", "m:1", model="local", backend="catgpt",
            context_signature="context-a"
        ))
        self.assertNotEqual(base, mwe_judge.occurrence_store_key(
            "could care less", "m:1", model="local", context_signature="context-b"
        ))
        with mock.patch.object(mwe_judge, "S3_PROMPT_VERSION", "s3-judge-prompt-next"):
            changed = mwe_judge.occurrence_store_key(
                "could care less", "m:1", model="local", context_signature="context-a"
            )
        self.assertNotEqual(base, changed)
        self.assertNotEqual(base, "m:1|could care less")

    def test_sense_clustering_separates_polysemy_and_merges_synonyms(self):
        from pipeline import mwe_judge
        def occurrence(occurrence_id, paraphrase):
            return {"occurrence_id": occurrence_id, "occurrence_decision": {
                "label": "phrasal_verb", "canonical_form": "burn out", "pos": "VERB",
                "contextual_paraphrase": paraphrase, "wordnet_sense_id": None,
            }}
        records = [{"occurrences": [
            occurrence("m:1", "become completely exhausted"),
            occurrence("m:2", "become completely exhausted"),
            occurrence("m:3", "stop producing light"),
        ]}]
        mwe_judge.assign_sense_ids(records)
        decisions = [o["occurrence_decision"] for o in records[0]["occurrences"]]
        self.assertEqual(decisions[0]["sense_id"], decisions[1]["sense_id"])
        self.assertNotEqual(decisions[0]["sense_id"], decisions[2]["sense_id"])
        self.assertTrue(all(d["pos"] == "VERB" for d in decisions))
        self.assertTrue(all(d["sense_id"].startswith("mwe-custom-v1:") for d in decisions))
        self.assertFalse({"idiome", "phrasal_verb", "semi_fige"} &
                         {d["sense_id"] for d in decisions})

    def test_custom_sense_id_is_stable_and_versioned(self):
        from pipeline import mwe_judge
        first = mwe_judge.custom_sense_id("come back to earth", "VERB", "return to reality")
        second = mwe_judge.custom_sense_id("Come Back To Earth", "verb", "To return to the reality")
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("mwe-custom-v1:"))


class ContextualMweDefinitionTests(unittest.TestCase):
    def _occurrences(self):
        return [{"occurrence_id": "m:10", "segment_idx": 10,
                 "occurrence_decision": {"contextual_paraphrase": "stop operating"}}]

    def test_exact_competing_sense_is_selected_after_all_are_presented(self):
        from pipeline import mwe_judge
        candidates = [
            {"candidate_id": "give-out-announce", "definition": "announce publicly", "source": "idiomatch"},
            {"candidate_id": "fail.v.04", "definition": "stop operating or functioning", "source": "wordnet"},
        ]
        reply = {"candidate_id": "fail.v.04", "custom_definition": "",
                 "occurrence_checks": [{"occurrence_id": "m:10", "contradicts": False}]}
        with mock.patch.object(mwe_judge, "definition_candidates", return_value=candidates), \
             mock.patch.object(mwe_judge.llm, "call_json", return_value=reply) as call:
            selected = mwe_judge.choose_cluster_definition("give out", "VERB", self._occurrences(), {})
        prompt = call.call_args.args[0]
        self.assertIn("announce publicly", prompt)
        self.assertIn("stop operating or functioning", prompt)
        self.assertEqual(selected["definition_en"], "stop operating or functioning")
        self.assertFalse(selected["definition_needs_review"])

    def test_custom_definition_is_allowed_when_no_candidate_is_exact(self):
        from pipeline import mwe_judge
        reply = {"candidate_id": None,
                 "custom_definition": "To contact someone briefly for an update.",
                 "occurrence_checks": [{"occurrence_id": "m:10", "contradicts": False}]}
        with mock.patch.object(mwe_judge, "definition_candidates", return_value=[
            {"candidate_id": "hotel", "definition": "announce arrival at a hotel", "source": "wordnet"}
        ]), mock.patch.object(mwe_judge.llm, "call_json", return_value=reply):
            selected = mwe_judge.choose_cluster_definition("check in", "VERB", self._occurrences(), {})
        self.assertEqual(selected["definition_source"], "custom")
        self.assertEqual(selected["definition_en"], reply["custom_definition"])

    def test_missing_or_contradictory_occurrence_check_forces_review(self):
        from pipeline import mwe_judge
        reply = {"candidate_id": "wrong", "custom_definition": "announce publicly",
                 "occurrence_checks": [{"occurrence_id": "m:10", "contradicts": True}]}
        with mock.patch.object(mwe_judge, "definition_candidates", return_value=[]), \
             mock.patch.object(mwe_judge.llm, "call_json", return_value=reply):
            selected = mwe_judge.choose_cluster_definition("give out", "VERB", self._occurrences(), {})
        self.assertTrue(selected["definition_needs_review"])
        self.assertEqual(selected["definition_en"], "stop operating")

    def test_book_cases_choose_contextual_sense_over_competing_first_sense(self):
        from pipeline import mwe_judge
        expected = {
            "break up": "To end a romantic relationship.",
            "bring up": "To mention or introduce a subject for discussion.",
            "check in": "To contact someone briefly to ask how things are going or to exchange an update.",
            "get a grip": "To regain self-control and composure.",
            "give out": "To stop working or fail after use, especially a machine or appliance.",
            "keep up": "To prevent someone from sleeping.",
            "look after": "To take care of or attend to someone or something.",
            "turn off": "To switch off a light, device, or electrical appliance.",
            "work out": "To solve, arrange, or find a way through a problem or situation.",
        }
        for canonical, definition in expected.items():
            with self.subTest(canonical=canonical):
                candidates = [
                    {"candidate_id": "wrong-first", "definition": "an incompatible competing sense",
                     "source": "idiomatch"},
                    {"candidate_id": "book-sense", "definition": definition, "source": "fixture"},
                ]
                reply = {"candidate_id": "book-sense", "custom_definition": "",
                         "occurrence_checks": [{"occurrence_id": "m:10", "contradicts": False}]}
                with mock.patch.object(mwe_judge, "definition_candidates", return_value=candidates), \
                     mock.patch.object(mwe_judge.llm, "call_json", return_value=reply):
                    selected = mwe_judge.choose_cluster_definition(
                        canonical, "VERB", self._occurrences(), {}
                    )
                self.assertEqual(selected["definition_en"], definition)


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
    def test_semantic_unit_key_contains_canon_pos_and_sense(self):
        self.assertEqual(
            inventory.make_unit_key("Burn Out", "VERB", "mwe-custom-v1:human", kind="mwe"),
            "mwe:burn out:verb:mwe-custom-v1:human",
        )

    def test_polysemous_mwe_keys_are_distinct(self):
        human = inventory.make_unit_key("burn out", "VERB", "human", kind="mwe")
        bulb = inventory.make_unit_key("burn out", "VERB", "bulb", kind="mwe")
        self.assertNotEqual(human, bulb)

    def test_different_canons_never_collapse_into_come_to(self):
        keys = {
            inventory.make_unit_key(canon, "VERB", "sense", kind="mwe")
            for canon in ("come to", "come back to earth", "come home to", "come talk to")
        }
        self.assertEqual(len(keys), 4)

    def test_selected_mwe_surfaces_are_aggregated_per_semantic_tuple(self):
        def span(occurrence_id, surface, sense_id, paraphrase):
            return {
                "occurrence_id": occurrence_id, "idiom": "burn out",
                "canonical_form": "burn out", "pos": "VERB", "sense_id": sense_id,
                "sense_id_source": "fixture", "label": "phrasal_verb",
                "confidence": .9, "surface": surface, "start_char": 0,
                "end_char": len(surface), "definition_en": paraphrase,
                "definition_needs_review": False,
            }
        units = select.build_mwe_units({
            1: [span("m:1", "burned out", "human", "become exhausted")],
            2: [span("m:2", "burnt out", "human", "become exhausted")],
            3: [span("m:3", "burns out", "bulb", "stop producing light")],
        })
        self.assertEqual(len(units), 2)
        by_sense = {u["sense_id"]: u for u in units}
        self.assertEqual(by_sense["human"]["surface_forms"], ["burned out", "burnt out"])
        self.assertEqual(by_sense["bulb"]["surface_forms"], ["burns out"])
        self.assertNotEqual(by_sense["human"]["unit_key"], by_sense["bulb"]["unit_key"])

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
