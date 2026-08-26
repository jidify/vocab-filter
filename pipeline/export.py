"""S7 — Sorties : CSV relisible, JSONL exhaustif, file de révision,
rapport comparant les variantes."""

from __future__ import annotations

import csv
import json

from pipeline import config
from pipeline.score import build_records, aggregate_and_score, build_mwe_units

CSV_FIELDS = [
    "canonical_form", "surface_forms", "unit_type", "pos", "sense_id",
    "meaning_fr", "meaning_fr_official", "meaning_fr_alt", "contexte_en",
    "fr_status", "definition_en",
    "occurrences", "book_count", "dispersion",
    "zipf_need", "aoa_component", "fr_opacity", "sense_surprise", "confidence",
    "score_comprehension", "score_reuse", "score_default", "needs_review",
]


def write_csv(units: list[dict], path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for u in units:
            row = dict(u)
            row["surface_forms"] = "/".join(u["surface_forms"])
            for field in ("zipf_need", "aoa_component", "fr_opacity", "sense_surprise",
                          "confidence", "score_comprehension", "score_reuse", "score_default"):
                row[field] = round(row[field], 3)
            writer.writerow(row)


def write_jsonl(units: list[dict], path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")


def write_review_queue(units: list[dict], path) -> None:
    reviewable = [u for u in units if u["needs_review"]]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for u in reviewable:
            row = dict(u)
            row["surface_forms"] = "/".join(u["surface_forms"])
            writer.writerow(row)


def write_report(units: list[dict], path) -> None:
    by_comprehension = sorted(units, key=lambda u: -u["score_comprehension"])[:100]
    by_reuse = sorted(units, key=lambda u: -u["score_reuse"])[:100]
    by_default = sorted(units, key=lambda u: -u["score_default"])[:100]

    # Mots où les 2 tris divergent le plus (item 7 du plan : arbitrage
    # AoA sur les ~60 cas les plus discriminants).
    rank_comp = {u["canonical_form"]: i for i, u in enumerate(by_comprehension)}
    rank_reuse = {u["canonical_form"]: i for i, u in enumerate(by_reuse)}
    common = set(rank_comp) & set(rank_reuse)
    discriminating = sorted(
        common, key=lambda w: -abs(rank_comp[w] - rank_reuse[w])
    )[:60]

    lines = ["# Rapport — sélection de vocabulaire (The Humans)", ""]
    lines.append(f"{len(units)} unités (forme, POS, sens) exportées.")
    lines.append(f"{sum(1 for u in units if u['needs_review'])} en file de révision.")
    lines.append("")
    lines.append("## Top 100 — score_default (mélange comprehension/reuse)")
    lines.append("")
    lines.append("| # | forme | POS | sens | FR | zipf_need | aoa | fr_opacity | score |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, u in enumerate(by_default, start=1):
        lines.append(
            f"| {i} | {u['canonical_form']} | {u['pos']} | {u['sense_id']} | "
            f"{u['meaning_fr'] or '?'} | {u['zipf_need']:.2f} | {u['aoa_component']:.2f} | "
            f"{u['fr_opacity']:.2f} | {u['score_default']:.3f} |"
        )

    lines.append("")
    lines.append("## Mots discriminants pour l'arbitrage AoA (item 7 du plan)")
    lines.append("")
    lines.append("À annoter connu/inconnu (~10 min) pour trancher le signe de l'AoA "
                  "(voir score.py::AOA_SIGN).")
    lines.append("")
    for w in discriminating:
        u = next(u for u in units if u["canonical_form"] == w)
        lines.append(f"- **{w}** ({u['pos']}, sens={u['sense_id']}) — "
                      f"rang compréhension={rank_comp[w]+1}, rang réutilisabilité={rank_reuse[w]+1}")

    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> int:
    config.ensure_out_dir()
    records = build_records()
    units = aggregate_and_score(records)
    units.extend(build_mwe_units())
    units.sort(key=lambda u: -u["score_default"])

    write_csv(units, config.VOCAB_CSV_PATH)
    write_jsonl(units, config.VOCAB_JSONL_PATH)
    write_review_queue(units, config.REVIEW_QUEUE_PATH)
    write_report(units, config.REPORT_PATH)

    print(f"{len(units)} unités exportées.")
    print(f"  {config.VOCAB_CSV_PATH}")
    print(f"  {config.VOCAB_JSONL_PATH}")
    print(f"  {config.REVIEW_QUEUE_PATH}")
    print(f"  {config.REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
