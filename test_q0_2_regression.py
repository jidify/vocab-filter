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
CORPUS_PATH = Path("fix_pipeline/q0_2_stratified_corpus.json")


def jsonl(name):
    with (OUT / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_rows(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def vocab_rows(canon=None):
    rows = csv_rows("vocab.csv")
    return rows if canon is None else [r for r in rows if normalize(r["canonical_form"]) == normalize(canon)]


def corpus_cases():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["cases"]


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


def check_valid_out_of_scope_idiom_is_conserved():
    rows = vocab_rows("on one's feet")
    valid = [r for r in rows if normalize(r["unit_type"]) == "mwe" and normalize(r["meaning_fr_official"])]
    assert valid, "S7 out-of-scope policy failure: genuine idiom 'on one's feet' was not conserved as a complete translated MWE"


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
    "ameliorations_hors_perimetre": check_valid_out_of_scope_idiom_is_conserved,
    "end_to_end": check_end_to_end_gate,
}

EXPECTED_CURRENT_OUTCOMES = {
    "mwe_fusionnees": ("known_failure", "S2 boundary failure"),
    "mwe_occurrences_heterogenes": ("known_failure", "S3 occurrence judgment missing"),
    "mwe_manquees": ("known_failure", "S2/S3 MWE recall failure"),
    "mwe_polysemiques": ("known_failure", "S3 sense clustering failure"),
    "pos_lemme": ("known_failure", "S1/S5 POS-lemma alternatives lost"),
    "sens_wordnet": ("known_failure", "S5 WordNet context selection failure"),
    "composes_entites": ("known_failure", "S1/S5 compound/entity fragments"),
    "aucun_sens_adapte": ("known_failure", "S5/S7 silent disappearance"),
    "transparence": ("known_failure", "S7 transparent cognates retained"),
    "pending": ("known_failure", "S6/S7 pending translations"),
    "ameliorations_hors_perimetre": ("passes", "invariant satisfied"),
    "end_to_end": ("known_failure", "Q0-1 final thresholds not met"),
}

KNOWN_CASE_COVERAGE = {
    "come to": "mwe_fusionnees", "let someone go": "mwe_occurrences_heterogenes",
    "burn out": "mwe_polysemiques", "latch": "aucun_sens_adapte",
    "affection": "transparence", "intelligible": "transparence",
    "facility": "sens_wordnet", "frosting": "pos_lemme", "York": "composes_entites",
    "traductions officielles manquantes": "pending",
    "on one's feet (ajout valide)": "ameliorations_hors_perimetre",
}


class Q02CorpusContractTests(unittest.TestCase):
    def test_every_q01_named_anomaly_has_a_stratum(self):
        self.assertEqual(len(KNOWN_CASE_COVERAGE), 11)
        self.assertTrue(set(KNOWN_CASE_COVERAGE.values()) <= set(STRATIFIED_CHECKS))

    def test_corpus_is_minimal_observation_only_and_covers_every_stratum(self):
        cases = corpus_cases()
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        forbidden = {"expected", "benchmark", "target_sense", "target_translation", "answer"}
        self.assertFalse([(c["id"], forbidden & set(c)) for c in cases if forbidden & set(c)])
        self.assertTrue(set(STRATIFIED_CHECKS) - {"end_to_end"} <= {c["stratum"] for c in cases})

    def test_every_current_defect_fails_for_its_expected_stage_reason(self):
        observed = {}
        for name, check in STRATIFIED_CHECKS.items():
            try:
                check()
            except AssertionError as exc:
                observed[name] = ("known_failure", str(exc))
            else:
                observed[name] = ("passes", "invariant satisfied")
        for name, (status, reason_prefix) in EXPECTED_CURRENT_OUTCOMES.items():
            self.assertEqual(observed[name][0], status, name)
            self.assertTrue(observed[name][1].startswith(reason_prefix), (name, observed[name][1]))

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
        frozen = {"label": "idiome", "confidence": 0.91, "reason": "fixture figée"}
        occurrences = [{"segment_idx": 1, "surface": "on her feet"}]
        with mock.patch.object(mwe_judge.llm, "call_json", side_effect=lambda *a, **k: dict(frozen)) as call, mock.patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            first = mwe_judge.judge_type("on one's feet", occurrences, {}, wordnet_candidates=[])
            second = mwe_judge.judge_type("on one's feet", occurrences, {}, wordnet_candidates=[])
        self.assertEqual(first, second)
        self.assertEqual(call.call_count, 2)


def _add_characterization(name, check):
    is_failure = EXPECTED_CURRENT_OUTCOMES[name][0] == "known_failure"
    decorator = unittest.expectedFailure if is_failure else (lambda f: f)
    @decorator
    def test(self):
        check()
    test.__name__ = f"test_{'known_failure' if is_failure else 'preserved_invariant'}_{name}"
    test.__doc__ = f"Caractérise la strate {name} avec son état Q0-2 attendu."
    setattr(Q02CorpusContractTests, test.__name__, test)


for _name, _check in STRATIFIED_CHECKS.items():
    _add_characterization(_name, _check)


if __name__ == "__main__":
    unittest.main()
