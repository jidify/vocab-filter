"""Evaluation hors production de la politique S5-2 sur les annotations Q0-2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import senses


def ratio(a, b):
    return a / b if b else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--senses", type=Path, default=ROOT / "pipeline_out/senses.jsonl")
    parser.add_argument("--gold", type=Path, default=ROOT / "fix_pipeline/s5_2_calibration_gold.json")
    parser.add_argument("--out", type=Path, default=ROOT / "fix_pipeline/s5_2_calibration")
    args = parser.parse_args()
    gold = {(r["word"], r["segment_idx"]): r for r in json.loads(args.gold.read_text(encoding="utf-8"))["cases"]}
    observed = {}
    with args.senses.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row.get("word"), row.get("segment_idx"))
            if key in gold:
                observed[key] = row
    missing = sorted(set(gold) - set(observed))
    if missing:
        raise SystemExit(f"Cas Q0-2 absents de senses.jsonl: {missing}")

    details = []
    for key, annotation in gold.items():
        row = observed[key]
        results = row.get("candidates") or []
        gloss_best = max(results, key=lambda r: r.get("gloss_score", 0.0))["synset"] if results else None
        fr_rows = [r for r in results if r.get("fr_score", 0.0)]
        fr_best = max(fr_rows, key=lambda r: r["fr_score"])["synset"] if fr_rows else None
        policy = senses.calibrated_resolution_policy(
            results, located=bool(row.get("target_surface")), pos_compatible=True,
            has_bilingual=bool(row.get("french")),
            structural_conflict=bool(row.get("multi_token_candidates")),
            model_disagreement=bool(gloss_best and fr_best and gloss_best != fr_best),
        )
        predicted = row.get("best_sense")
        correct = predicted == annotation["expected"]
        details.append({"word": key[0], "segment_idx": key[1], "expected": annotation["expected"],
                        "prediction": predicted, "correct": correct, "legacy_review": row.get("margin", 1) < 0.15,
                        "policy_review": policy["needs_arbitration"], "confidence": policy["confidence"],
                        "entropy": policy["entropy"], "reasons": policy["reasons"]})

    def matrix(review_field):
        return {
            "correct_accept": sum(d["correct"] and not d[review_field] for d in details),
            "correct_review": sum(d["correct"] and d[review_field] for d in details),
            "incorrect_accept": sum(not d["correct"] and not d[review_field] for d in details),
            "incorrect_review": sum(not d["correct"] and d[review_field] for d in details),
        }

    bins = []
    for low, high in ((0, .4), (.4, .55), (.55, .72), (.72, 1.000001)):
        rows = [d for d in details if low <= d["confidence"] < high]
        bins.append({"range": f"[{low:.2f},{min(high, 1):.2f}]", "count": len(rows),
                     "mean_confidence": ratio(sum(d["confidence"] for d in rows), len(rows)),
                     "accuracy": ratio(sum(d["correct"] for d in rows), len(rows))})
    payload = {"schema_version": 1, "n": len(details),
               "raw_top1_accuracy": ratio(sum(d["correct"] for d in details), len(details)),
               "legacy_margin_confusion": matrix("legacy_review"),
               "calibrated_policy_confusion": matrix("policy_review"),
               "calibration_bins": bins, "cases": details}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m0, m1 = payload["legacy_margin_confusion"], payload["calibrated_policy_confusion"]
    lines = ["# Calibration S5-2", "", f"Corpus Q0-2 ciblé : **{len(details)} occurrences**.", "",
             "## Matrice de confusion (acceptation automatique / révision)", "",
             "| Politique | Correct accepté | Correct révisé | Incorrect accepté | Incorrect révisé |", "|---|---:|---:|---:|---:|",
             f"| Marge historique `< 0,15` | {m0['correct_accept']} | {m0['correct_review']} | {m0['incorrect_accept']} | {m0['incorrect_review']} |",
             f"| Politique calibrée | {m1['correct_accept']} | {m1['correct_review']} | {m1['incorrect_accept']} | {m1['incorrect_review']} |", "",
             "## Calibration par tranche", "", "| Confiance | N | Confiance moyenne | Exactitude top-1 brute |", "|---|---:|---:|---:|"]
    lines += [f"| {b['range']} | {b['count']} | {b['mean_confidence']:.3f} | {b['accuracy']:.3f} |" for b in bins]
    lines += ["", "La calibration mesure ici la fiabilité du top-1 avant arbitrage. Les cas routés en révision ne sont pas présentés comme des erreurs corrigées : ils attendent l'arbitre fermé WordNet/custom.", "", "## Cas nommés", "", "| Mot@segment | Attendu | Top-1 | Décision | Confiance |", "|---|---|---|---|---:|"]
    lines += [f"| {d['word']}@{d['segment_idx']} | `{d['expected']}` | `{d['prediction']}` | {'révision' if d['policy_review'] else 'accepté'} | {d['confidence']:.3f} |" for d in details]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
