"""Tests de pipeline/llm_client.py::run_units — l'orchestrateur qui appelle
en LOT et stocke en UNITAIRE (pipeline/llm_store.py). Couvre le cœur du plan
de décorrélation lot/stockage : le test d'acceptation central est
`test_changing_batch_size_triggers_zero_new_calls`.

Offline (litellm mocké), magasin isolé (LLM_RESULTS_DB_PATH -> tmpdir),
comme test_llm_client.py isole pipeline_out/cache/ et test_llm_store.py
isole data/llm_results.sqlite3."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from pipeline import config, llm_client, llm_store


class _Decision(BaseModel):
    label: str


class _BatchDecisions(BaseModel):
    decisions: list[_Decision]


class _DecisionWithId(BaseModel):
    unit_id: str
    label: str


class _BatchDecisionsWithId(BaseModel):
    decisions: list[_DecisionWithId]


def _choice(payload_json: str):
    return type("Choice", (), {"message": type("Message", (), {"content": payload_json})()})()


class _Response:
    def __init__(self, model: BaseModel):
        self.choices = [_choice(model.model_dump_json())]


def _units(ids: list[str]) -> list[llm_client.Unit]:
    return [llm_client.Unit(unit_id=i, payload={"surface": i}) for i in ids]


def _render_unit(unit: llm_client.Unit) -> tuple[str, str]:
    return "system", f"unit:{unit.unit_id}"


def _render_batch(units: list[llm_client.Unit]) -> tuple[str, str]:
    return "system", "batch:" + ",".join(u.unit_id for u in units)


def _parse_unit(parsed: _DecisionWithId, unit: llm_client.Unit) -> dict:
    return {"label": parsed.label}


def _parse_batch(parsed: _BatchDecisionsWithId, units: list[llm_client.Unit]) -> dict[str, dict]:
    by_id = llm_client.dedupe_batch_items(parsed.decisions, id_key="unit_id")
    return {uid: {"label": d.label} for uid, d in by_id.items()}


class RunUnitsIsolatedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch.object(
            config, "LLM_RESULTS_DB_PATH", Path(self._tmp.name) / "llm_results.sqlite3",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, ids, *, model="m", batch_size=50, mode_batch=True, on_failure=None):
        return llm_client.run_units(
            _units(ids), task_id="T", model=model, protocol="p1",
            render_unit=_render_unit, render_batch=_render_batch,
            parse_unit=_parse_unit, parse_batch=_parse_batch,
            response_model_unit=_DecisionWithId, response_model_batch=_BatchDecisionsWithId,
            batch_size=batch_size, mode_batch=mode_batch, on_failure=on_failure,
        )


class BatchSizeIndependenceTests(RunUnitsIsolatedTests):
    """Le test d'acceptation central du plan : une unité déjà stockée reste
    un hit quel que soit le batch_size demandé ensuite — plus aucun appel."""

    def test_changing_batch_size_triggers_zero_new_calls(self):
        payload = _BatchDecisionsWithId(decisions=[
            _DecisionWithId(unit_id=str(i), label="ok") for i in range(20)
        ])
        with patch.object(llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [_Response(payload) for _ in kw["messages"]]) as mocked:
            results = self._run([str(i) for i in range(20)], batch_size=5)
        mocked.assert_called_once()
        self.assertEqual(len(mocked.call_args.kwargs["messages"]), 4)  # 20 / batch_size=5
        self.assertEqual(len(results), 20)

        # même 20 unités, batch_size différent (5 -> 7) : zéro appel réseau
        with patch.object(llm_client.litellm, "batch_completion") as mocked2:
            results2 = self._run([str(i) for i in range(20)], batch_size=7)
        mocked2.assert_not_called()
        self.assertEqual(results2, results)

        # unitaire cette fois (mode_batch=False) : toujours zéro appel
        with patch.object(llm_client.litellm, "batch_completion") as mocked3:
            results3 = self._run([str(i) for i in range(20)], batch_size=1, mode_batch=False)
        mocked3.assert_not_called()
        self.assertEqual(results3, results)

    def test_different_model_repays_all_units(self):
        payload = _BatchDecisionsWithId(decisions=[_DecisionWithId(unit_id="0", label="ok")])
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(payload)]) as mocked:
            self._run(["0"], model="model-a", batch_size=5)
        mocked.assert_called_once()

        payload2 = _BatchDecisionsWithId(decisions=[_DecisionWithId(unit_id="0", label="ok-b")])
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(payload2)]) as mocked2:
            results = self._run(["0"], model="model-b", batch_size=5)
        mocked2.assert_called_once()  # modèle différent -> rejoué, pas un hit
        self.assertEqual(results["0"]["label"], "ok-b")


class ChunkingAndPartialFailureTests(RunUnitsIsolatedTests):
    def test_batches_are_chunked_by_batch_size(self):
        payload = _BatchDecisionsWithId(decisions=[
            _DecisionWithId(unit_id=str(i), label="ok") for i in range(5)
        ])
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(payload), _Response(payload)]) as mocked:
            self._run([str(i) for i in range(10)], batch_size=5)
        # 10 unités / batch_size=5 -> 2 tranches envoyées en un seul appel
        # HTTP parallèle (litellm.batch_completion), pas 2 appels séparés.
        self.assertEqual(len(mocked.call_args.kwargs["messages"]), 2)

    def test_unit_mode_sends_one_message_per_unit(self):
        with patch.object(llm_client.litellm, "batch_completion",
                          side_effect=lambda **kw: [
                              _Response(_DecisionWithId(unit_id=str(i), label="ok"))
                              for i in range(len(kw["messages"]))
                          ]) as mocked:
            results = self._run([str(i) for i in range(3)], batch_size=1, mode_batch=False)
        self.assertEqual(len(mocked.call_args.kwargs["messages"]), 3)
        self.assertEqual(len(results), 3)

    def test_partial_batch_failure_still_stores_successful_units(self):
        """Une unité absente de la réponse de lot part en échec ; les autres
        unités du MÊME lot sont quand même stockées — jamais de
        tout-ou-rien par lot (contrairement à l'ancien cache disque)."""
        payload = _BatchDecisionsWithId(decisions=[
            _DecisionWithId(unit_id="0", label="ok"),
            _DecisionWithId(unit_id="2", label="ok"),
            # "1" manquant : reste en échec
        ])
        failures = []
        with patch.object(llm_client.litellm, "batch_completion", return_value=[_Response(payload)]):
            results = self._run(["0", "1", "2"], batch_size=5,
                                on_failure=lambda unit, reason: failures.append(unit.unit_id))
        self.assertEqual(set(results), {"0", "2"})
        self.assertEqual(failures, ["1"])

        # "1" seul, relancé plus tard : pas de hit fantôme, repart au modèle
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(_BatchDecisionsWithId(
                              decisions=[_DecisionWithId(unit_id="1", label="ok")]))]) as mocked:
            results2 = self._run(["1"], batch_size=5)
        mocked.assert_called_once()
        self.assertEqual(results2["1"]["label"], "ok")

    def test_chunk_level_exception_fails_only_that_chunks_units(self):
        # Mode unitaire (batch_size=1) : chaque unité est sa propre tranche,
        # les 3 sont envoyées dans le MÊME appel HTTP parallèle — l'échec de
        # l'une (item 0) ne doit rien coûter aux deux autres.
        responses = [
            RuntimeError("dead"),
            _Response(_DecisionWithId(unit_id="1", label="ok")),
            _Response(_DecisionWithId(unit_id="2", label="ok")),
        ]
        with patch.object(llm_client.litellm, "batch_completion", return_value=responses):
            failures = []
            results = self._run(["0", "1", "2"], batch_size=1, mode_batch=False,
                                on_failure=lambda unit, reason: failures.append(unit.unit_id))
        self.assertEqual(set(results), {"1", "2"})
        self.assertEqual(failures, ["0"])

    def test_duplicate_unit_id_in_batch_response_is_a_failure_not_a_random_pick(self):
        payload = _BatchDecisionsWithId(decisions=[
            _DecisionWithId(unit_id="0", label="a"),
            _DecisionWithId(unit_id="0", label="b"),
        ])
        with patch.object(llm_client.litellm, "batch_completion", return_value=[_Response(payload)]):
            failures = []
            results = self._run(["0"], batch_size=5,
                                on_failure=lambda unit, reason: failures.append(unit.unit_id))
        self.assertEqual(results, {})
        self.assertEqual(failures, ["0"])

    def test_llm_errors_are_never_stored(self):
        """Symétrique à la règle historique du magasin métier : une panne
        LLM n'est jamais mise en cache (mwe_judge.py, 'pas mises en cache
        non plus') — on ne doit jamais retrouver de hit fantôme ensuite."""
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[RuntimeError("boom")]):
            results = self._run(["0"], batch_size=5)
        self.assertEqual(results, {})
        rows = llm_store.stats(task_id="T")
        self.assertEqual(rows, [])

    def test_systemic_failure_never_raises_and_preserves_prior_hits(self):
        """Une panne SYSTÉMIQUE (litellm.batch_completion lève, pas un item
        qui échoue individuellement) ne doit ni faire planter run_units, ni
        perdre les hits déjà trouvés dans le magasin avant l'appel réseau —
        seules les unités encore en attente partent vers on_failure."""
        payload = _BatchDecisionsWithId(decisions=[_DecisionWithId(unit_id="hit", label="ok")])
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(payload)]):
            first = self._run(["hit"], batch_size=5)
        self.assertEqual(set(first), {"hit"})

        failures = []
        with patch.object(llm_client.litellm, "batch_completion",
                          side_effect=RuntimeError("provider down")):
            results = self._run(["hit", "new"], batch_size=5,
                                on_failure=lambda unit, reason: failures.append(unit.unit_id))
        self.assertEqual(set(results), {"hit"})  # le hit survit à la panne
        self.assertEqual(failures, ["new"])


class ParseCallbackRobustnessTests(RunUnitsIsolatedTests):
    def test_parse_batch_exception_fails_its_chunk_without_crashing(self):
        def _parse_batch_boom(parsed, units):
            raise ValueError("malformed response")

        payload = _BatchDecisionsWithId(decisions=[_DecisionWithId(unit_id="0", label="ok")])
        with patch.object(llm_client.litellm, "batch_completion", return_value=[_Response(payload)]):
            failures = []
            results = llm_client.run_units(
                _units(["0"]), task_id="T", model="m", protocol="p1",
                render_unit=_render_unit, render_batch=_render_batch,
                parse_unit=_parse_unit, parse_batch=_parse_batch_boom,
                response_model_unit=_DecisionWithId, response_model_batch=_BatchDecisionsWithId,
                batch_size=5, mode_batch=True,
                on_failure=lambda unit, reason: failures.append(unit.unit_id),
            )
        self.assertEqual(results, {})
        self.assertEqual(failures, ["0"])


class NoUnitsTests(RunUnitsIsolatedTests):
    def test_empty_units_list_returns_empty_without_any_call(self):
        with patch.object(llm_client.litellm, "batch_completion") as mocked:
            results = self._run([])
        mocked.assert_not_called()
        self.assertEqual(results, {})


class ReturnCostTests(RunUnitsIsolatedTests):
    def test_return_cost_reports_real_cost_on_miss_and_zero_on_hit(self):
        payload = _BatchDecisionsWithId(decisions=[_DecisionWithId(unit_id="0", label="ok")])
        with patch.object(llm_client.litellm, "batch_completion",
                          return_value=[_Response(payload)]), \
             patch.object(llm_client.litellm, "completion_cost", return_value=0.0042):
            results, cost = llm_client.run_units(
                _units(["0"]), task_id="T", model="m", protocol="p1",
                render_unit=_render_unit, render_batch=_render_batch,
                parse_unit=_parse_unit, parse_batch=_parse_batch,
                response_model_unit=_DecisionWithId, response_model_batch=_BatchDecisionsWithId,
                batch_size=5, mode_batch=True, return_cost=True,
            )
        self.assertEqual(results["0"]["label"], "ok")
        self.assertAlmostEqual(cost, 0.0042)

        with patch.object(llm_client.litellm, "batch_completion") as mocked:
            results2, cost2 = llm_client.run_units(
                _units(["0"]), task_id="T", model="m", protocol="p1",
                render_unit=_render_unit, render_batch=_render_batch,
                parse_unit=_parse_unit, parse_batch=_parse_batch,
                response_model_unit=_DecisionWithId, response_model_batch=_BatchDecisionsWithId,
                batch_size=5, mode_batch=True, return_cost=True,
            )
        mocked.assert_not_called()
        self.assertEqual(cost2, 0.0)
        self.assertEqual(results2["0"]["label"], "ok")

    def test_return_cost_zero_on_empty_units(self):
        results, cost = llm_client.run_units(
            [], task_id="T", model="m", protocol="p1",
            render_unit=_render_unit, render_batch=_render_batch,
            parse_unit=_parse_unit, parse_batch=_parse_batch,
            response_model_unit=_DecisionWithId, response_model_batch=_BatchDecisionsWithId,
            batch_size=5, mode_batch=True, return_cost=True,
        )
        self.assertEqual((results, cost), ({}, 0.0))


if __name__ == "__main__":
    unittest.main()
