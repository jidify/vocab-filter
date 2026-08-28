"""Q0-2 offline characterization corpus.

Expected answers intentionally live in this test module, never in production
stores.  Known failures are marked ``expectedFailure`` until their owning
S1--S7 correction is implemented; an unexpected success therefore asks for the
test to be promoted to an ordinary regression test.
"""

from __future__ import annotations

import csv
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from fix_pipeline.evaluate_fix_quality import evaluate, normalize, read_csv

OUT = Path("pipeline_out")


def jsonl(name):
    with (OUT / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_rows(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def vocab_rows(canon=None):
    rows = csv_rows("vocab.csv")
    return rows if canon is None else [r for r in rows if normalize(r["canonical_form"]) == normalize(canon)]


def check_mwe_merged_boundaries():
    selected = {normalize(r["canonical_form"]): r for r in jsonl("selected_mwe.jsonl")}
    bad = [s for s in selected["come to"]["surface_forms"] if len(normalize(s).split()) != 2]
    assert not bad, f"S2 boundary failure: lexical tokens swallowed by 'come to': {bad}"


def check_mwe_occurrence_senses():
    decision = next(r for r in jsonl("mwe_decisions.jsonl") if normalize(r["idiom"]) == "let someone go")
    missing = [o["surface"] for o in decision["occurrences"] if "occurrence_decision" not in o]
    assert not missing, f"S3 occurrence judgment missing for heterogeneous 'let someone go': {missing}"


def check_mwe_recall():
    expected = {"let it go", "come back to earth", "get worked up", "at ease", "burn out", "put to rest", "steer clear of", "could care less", "tighten one's belt"}
    present = {normalize(r["canonical_form"]) for r in vocab_rows() if normalize(r["unit_type"]) == "mwe"}
    missing = sorted(expected - present)
    assert not missing, f"S2/S3 MWE recall failure: eligible canonical units absent: {missing}"


def check_mwe_polysemy():
    rows = vocab_rows("burn out")
    ids = {normalize(r["sense_id"]) for r in rows}
    translations = {normalize(r["meaning_fr_official"]) for r in rows}
    assert len(rows) >= 2 and len(ids) >= 2 and {"s'épuiser", "griller"} <= translations, (
        f"S3 sense clustering failure: burn out rows={len(rows)}, ids={sorted(ids)}, FR={sorted(translations)}"
    )


def check_pos_lemma_alternatives():
    expected = {
        "frost": ("n", "frosting.n.01"),
        "creep": ("s", "creeping.s.01"),
        "facility": ("n", "facilities.n.02"),
        "stress": ("v", "de_stress.v.01"),
        "bitch": ("n", "bitch.n.01"),
    }
    wrong = {}
    for canon, target in expected.items():
        actual = {(normalize(r["pos"]), normalize(r["sense_id"])) for r in vocab_rows(canon)}
        if target not in actual:
            wrong[canon] = {"expected": target, "actual": sorted(actual)}
    assert not wrong, f"S1/S5 POS-lemma alternatives lost: {wrong}"


def check_wordnet_context_senses():
    expected = {"facility": "facilities.n.02", "plow": "plow.v.03", "haggard": "haggard.s.02", "spa": "health_spa.n.01"}
    wrong = {word: [r["sense_id"] for r in vocab_rows(word)] for word, sense in expected.items() if sense not in {r["sense_id"] for r in vocab_rows(word)}}
    assert not wrong, f"S5 WordNet context selection failure: {wrong}"


def check_compound_entity_fragments():
    compounds = {"york": "new york", "virgin": "virgin mary", "ranch": "ranch dip", "observation": "observation deck", "nursing": "nursing home", "crystal": "crystal ball"}
    leaked = []
    for fragment, compound in compounds.items():
        rows = vocab_rows(fragment)
        if rows and all(compound in normalize(r["contexte_en"]) for r in rows):
            leaked.append((fragment, compound))
    assert not leaked, f"S1/S5 compound/entity fragments exported as standalone words: {leaked}"


def check_no_sense_disappearance():
    unresolved = [r for r in jsonl("senses.jsonl") if normalize(r.get("best_sense")) == "aucun_sens_adapte"]
    final_canons = {normalize(r["canonical_form"]) for r in vocab_rows()}
    review_canons = {normalize(r["canonical_form"]) for r in csv_rows("review_queue.csv")}
    vanished = sorted({normalize(r["word"]) for r in unresolved} - final_canons - review_canons)
    assert not vanished, f"S5/S7 silent disappearance after aucun_sens_adapte: {vanished}"


def check_transparent_selection():
    transparent = {"affection", "intelligible"}
    retained = sorted(transparent & {normalize(r["canonical_form"]) for r in vocab_rows()})
    assert not retained, f"S7 transparent cognates retained without pedagogical justification: {retained}"


def check_pending_accounting():
    review = {(normalize(r["canonical_form"]), normalize(r["sense_id"])) for r in csv_rows("review_queue.csv")}
    lost = [(r["canonical_form"], r["sense_id"]) for r in vocab_rows() if not normalize(r["meaning_fr_official"]) and (normalize(r["canonical_form"]), normalize(r["sense_id"])) not in review]
    assert not lost, f"S6/S7 pending translations exported empty and absent from review ({len(lost)}): {lost[:8]}"


def check_end_to_end_gate():
    result = evaluate(read_csv(OUT / "vocab.csv"), read_csv(OUT / "vocab_corrige.csv"))
    thresholds = {"unit_precision": .97, "unit_recall": .97, "pos_accuracy": .99, "sense_identity_accuracy": .97, "definition_accuracy": .98, "official_fr_coverage": 1.0, "official_fr_soft_accuracy": .98, "sense_definition_fr_coherence": 1.0}
    failed = {key: result["metrics"][key]["value"] for key, minimum in thresholds.items() if (result["metrics"][key]["value"] or 0) < minimum}
    assert not failed, f"Q0-1 final thresholds not met: {failed}"


STRATIFIED_CHECKS = {
    "mwe_fusionnees": check_mwe_merged_boundaries,
    "mwe_occurrences_heterogenes": check_mwe_occurrence_senses,
    "mwe_manquees": check_mwe_recall,
    "mwe_polysemiques": check_mwe_polysemy,
    "pos_lemme": check_pos_lemma_alternatives,
    "sens_wordnet": check_wordnet_context_senses,
    "composes_entites": check_compound_entity_fragments,
    "aucun_sens_adapte": check_no_sense_disappearance,
    "transparence": check_transparent_selection,
    "pending": check_pending_accounting,
    "end_to_end": check_end_to_end_gate,
}

KNOWN_CASE_COVERAGE = {
    "come to": "mwe_fusionnees", "let someone go": "mwe_occurrences_heterogenes",
    "burn out": "mwe_polysemiques", "latch": "aucun_sens_adapte",
    "affection": "transparence", "intelligible": "transparence",
    "facility": "sens_wordnet", "frosting": "pos_lemme", "York": "composes_entites",
    "traductions officielles manquantes": "pending",
}


class Q02CorpusContractTests(unittest.TestCase):
    def test_every_q01_named_anomaly_has_a_stratum(self):
        self.assertEqual(len(KNOWN_CASE_COVERAGE), 10)
        self.assertTrue(set(KNOWN_CASE_COVERAGE.values()) <= set(STRATIFIED_CHECKS))

    def test_llm_fixture_is_frozen_and_offline(self):
        # mwe_judge only needs nltk for optional WordNet lookup.  Supply a
        # deliberately inert import fixture so this ordinary test remains
        # runnable even in the minimal, dependency-free Python environment.
        nltk = types.ModuleType("nltk")
        corpus = types.ModuleType("nltk.corpus")
        corpus.wordnet = types.SimpleNamespace(synsets=lambda _word: ())
        nltk.corpus = corpus
        with mock.patch.dict(sys.modules, {"nltk": nltk, "nltk.corpus": corpus}):
            from pipeline import mwe_judge
        frozen = {"label": "phrasal_verb", "confidence": 0.91, "reason": "fixture figée"}
        occurrences = [{"segment_idx": 1, "surface": "worked up"}]
        with mock.patch.object(mwe_judge.llm, "call_json", side_effect=lambda *a, **k: dict(frozen)) as call:
            first = mwe_judge.judge_type("work up", occurrences, {}, wordnet_candidates=[])
            second = mwe_judge.judge_type("work up", occurrences, {}, wordnet_candidates=[])
        self.assertEqual(first, second)
        self.assertEqual(call.call_count, 2)


def _add_expected_failure(name, check):
    @unittest.expectedFailure
    def test(self):
        check()
    test.__name__ = f"test_known_failure_{name}"
    test.__doc__ = f"Caractérise la strate {name}; doit échouer avant sa correction propriétaire."
    setattr(Q02CorpusContractTests, test.__name__, test)


for _name, _check in STRATIFIED_CHECKS.items():
    _add_expected_failure(_name, _check)


if __name__ == "__main__":
    unittest.main()
