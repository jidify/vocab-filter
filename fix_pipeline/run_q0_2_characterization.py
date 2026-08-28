"""Run Q0-2 checks directly and publish the precise current failure reasons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_q0_2_regression import KNOWN_CASE_COVERAGE, STRATIFIED_CHECKS


def main() -> int:
    results = []
    for stratum, check in STRATIFIED_CHECKS.items():
        try:
            check()
        except AssertionError as exc:
            results.append({"stratum": stratum, "status": "known_failure", "reason": str(exc)})
        except Exception as exc:
            results.append({"stratum": stratum, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"stratum": stratum, "status": "passes", "reason": "invariant satisfied"})
    payload = {"schema_version": 1, "mode": "offline_artifact_characterization", "known_case_coverage": KNOWN_CASE_COVERAGE, "results": results}
    out_json = Path("pipeline_out/q0_2_regression_results.json")
    out_md = Path("pipeline_out/q0_2_regression_report.md")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Rapport de caractérisation Q0-2", "", "Mode hors réseau : artefacts existants et réponses LLM figées dans les tests.", "", "| Strate | État actuel | Raison localisée |", "|---|---|---|"]
    for row in results:
        reason = row["reason"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{row['stratum']}` | {row['status']} | {reason} |")
    lines += ["", "## Couverture des anomalies Q0-1", ""]
    lines += [f"- `{case}` → `{stratum}`" for case, stratum in KNOWN_CASE_COVERAGE.items()]
    lines += ["", "Les `known_failure` sont intentionnels avant S1–S7. Une correction doit d'abord transformer le test propriétaire en succès, puis retirer son décorateur `expectedFailure`. L'évaluation LLM réelle demeure dans `test_q0_2_real_eval.py` et est opt-in.", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return 0 if all(r["status"] == "known_failure" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
