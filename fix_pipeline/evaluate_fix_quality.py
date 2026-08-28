"""Read-only Q0-1 evaluator for a generated vocabulary and its benchmark.

This module deliberately lives outside :mod:`pipeline`: production code must not
depend on the hand-corrected benchmark.  It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIELDS = {
    "canonical_form",
    "surface_forms",
    "unit_type",
    "pos",
    "sense_id",
    "meaning_fr_official",
    "meaning_fr_alt",
    "definition_en",
    "needs_review",
}


def normalize(value: str | None) -> str:
    """Normalize Unicode, case, apostrophes and whitespace for comparisons."""
    value = unicodedata.normalize("NFKC", value or "")
    value = value.translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "\u00a0": " "}))
    return " ".join(value.casefold().split())


def _truthy(value: str | None) -> bool:
    return normalize(value) in {"1", "true", "yes", "oui"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV without ever opening it in a writable mode."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = FIELDS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def unit_key(row: dict[str, str]) -> tuple[str, str]:
    return normalize(row["canonical_form"]), normalize(row["unit_type"])


def _surface_set(row: dict[str, str]) -> frozenset[str]:
    return frozenset(normalize(x) for x in row["surface_forms"].split("/") if normalize(x))


def _pair_score(actual: dict[str, str], expected: dict[str, str]) -> tuple[int, ...]:
    """Rank pairings inside an homonym group, without collapsing its rows."""
    sa, se = _surface_set(actual), _surface_set(expected)
    overlap = len(sa & se)
    return (
        int(bool(sa) and sa == se),
        overlap,
        int(normalize(actual["sense_id"]) == normalize(expected["sense_id"])),
        int(normalize(actual["definition_en"]) == normalize(expected["definition_en"])),
        int(normalize(actual["pos"]) == normalize(expected["pos"])),
        int(normalize(actual["meaning_fr_official"]) == normalize(expected["meaning_fr_official"])),
    )


@dataclass(frozen=True)
class Pair:
    actual_index: int
    expected_index: int


def match_rows(actual: list[dict[str, str]], expected: list[dict[str, str]]) -> tuple[list[Pair], list[int], list[int]]:
    """Deterministic maximum-quality matching within canonical/type groups.

    Multiple senses sharing one canonical form remain separate.  The exhaustive
    assignment is practical here because homonym groups are small, and avoids an
    optional dependency or a greedy, order-sensitive match.
    """
    ag: dict[tuple[str, str], list[int]] = defaultdict(list)
    eg: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(actual):
        ag[unit_key(row)].append(i)
    for i, row in enumerate(expected):
        eg[unit_key(row)].append(i)

    pairs: list[Pair] = []
    used_a: set[int] = set()
    used_e: set[int] = set()
    for key in sorted(ag.keys() & eg.keys()):
        aa, ee = ag[key], eg[key]
        n = min(len(aa), len(ee))
        best: tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int], ...]] | None = None
        for chosen_a in itertools.combinations(aa, n):
            for chosen_e in itertools.permutations(ee, n):
                assignment = tuple(zip(chosen_a, chosen_e))
                totals = tuple(sum(_pair_score(actual[a], expected[e])[i] for a, e in assignment) for i in range(6))
                # Negative indices make the final tie break favor original order.
                candidate = totals, tuple(-x for pair in assignment for x in pair), assignment
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        assert best is not None
        for a, e in best[2]:
            pairs.append(Pair(a, e))
            used_a.add(a)
            used_e.add(e)
    pairs.sort(key=lambda p: (unit_key(actual[p.actual_index]), p.actual_index, p.expected_index))
    return pairs, sorted(set(range(len(actual))) - used_a), sorted(set(range(len(expected))) - used_e)


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator, "value": round(numerator / denominator, 6) if denominator else None}


def _soft_fr_equal(a: str, b: str) -> bool:
    """Conservative deterministic variant match; no semantic model is implied."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Alternatives are conventionally separated by slash or semicolon.
    va = {normalize(x) for x in re.split(r"[/;]", a) if normalize(x)}
    vb = {normalize(x) for x in re.split(r"[/;]", b) if normalize(x)}
    return bool(va & vb)


OUT_OF_SCOPE_STATUSES = {"acceptable_variant", "validated_improvement", "needs_review", "false_positive"}


def _audit_key(row: dict[str, str]) -> tuple[str, str, str]:
    return normalize(row.get("canonical_form")), normalize(row.get("unit_type")), normalize(row.get("sense_id"))


def read_audit(path: Path | None) -> dict[tuple[str, str, str], dict]:
    """Read optional human adjudications for produced units outside the benchmark."""
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", data) if isinstance(data, dict) else data
    result = {}
    for entry in entries:
        status = entry.get("status", "")
        if status not in OUT_OF_SCOPE_STATUSES:
            raise ValueError(f"{path}: invalid out-of-scope status {status!r}")
        key = _audit_key(entry)
        if not key[0] or not key[1] or key in result:
            raise ValueError(f"{path}: invalid or duplicate audit identity {key!r}")
        result[key] = dict(entry)
    return result


def evaluate(actual: list[dict[str, str]], expected: list[dict[str, str]], audit: dict[tuple[str, str, str], dict] | None = None) -> dict:
    audit = audit or {}
    pairs, only_a, only_e = match_rows(actual, expected)
    paired = [(actual[p.actual_index], expected[p.expected_index]) for p in pairs]
    mwe_pairs = [(a, e) for a, e in paired if normalize(e["unit_type"]) == "mwe"]
    exact_pos = sum(normalize(a["pos"]) == normalize(e["pos"]) for a, e in paired)
    exact_sense = sum(normalize(a["sense_id"]) == normalize(e["sense_id"]) for a, e in paired)
    exact_def = sum(normalize(a["definition_en"]) == normalize(e["definition_en"]) for a, e in paired)
    exact_surfaces = sum(_surface_set(a) == _surface_set(e) for a, e in paired)
    exact_decision = sum(_truthy(a["needs_review"]) == _truthy(e["needs_review"]) for a, e in paired)
    fr_present = sum(bool(normalize(a["meaning_fr_official"])) for a in actual)
    fr_soft = sum(_soft_fr_equal(a["meaning_fr_official"], e["meaning_fr_official"]) for a, e in paired if normalize(e["meaning_fr_official"]))
    fr_expected = sum(bool(normalize(e["meaning_fr_official"])) for _, e in paired)
    coherent = sum(
        normalize(a["sense_id"]) == normalize(e["sense_id"])
        and normalize(a["definition_en"]) == normalize(e["definition_en"])
        and _soft_fr_equal(a["meaning_fr_official"], e["meaning_fr_official"])
        for a, e in paired if not _truthy(a["needs_review"])
    )
    non_reviewed = sum(not _truthy(a["needs_review"]) for a, _ in paired)
    current_mwe = sum(normalize(r["unit_type"]) == "mwe" for r in actual)
    expected_mwe = sum(normalize(r["unit_type"]) == "mwe" for r in expected)
    matched_mwe = len(mwe_pairs)
    missing_fr_fixed = [a for a, e in paired if not normalize(a["meaning_fr_official"]) and normalize(e["meaning_fr_official"])]

    def summary(row: dict[str, str]) -> dict[str, str]:
        return {k: row.get(k, "") for k in ("canonical_form", "surface_forms", "unit_type", "pos", "sense_id", "meaning_fr_official", "definition_en", "needs_review")}

    out_of_scope = []
    for i in only_a:
        item = summary(actual[i])
        annotation = audit.get(_audit_key(actual[i]))
        item.update({"classification": annotation["status"] if annotation else "needs_review", "audit_reason": (annotation or {}).get("reason", "Absence du benchmark : adjudication requise.")})
        out_of_scope.append(item)
    class_counts = {status: sum(x["classification"] == status for x in out_of_scope) for status in OUT_OF_SCOPE_STATUSES}
    precision_denominator = len(pairs) + class_counts["false_positive"]
    audited_additions = class_counts["validated_improvement"] + class_counts["false_positive"]

    differences = []
    for a, e in paired:
        fields = [f for f in ("surface_forms", "pos", "sense_id", "definition_en", "meaning_fr_official", "needs_review") if normalize(a[f]) != normalize(e[f])]
        if fields:
            differences.append({"canonical_form": e["canonical_form"], "unit_type": e["unit_type"], "fields": fields, "actual": summary(a), "expected": summary(e)})
    matched_exact_count = len(pairs) - len(differences)

    metrics = {
        "unit_precision": _ratio(len(pairs), precision_denominator),
        "unit_recall": _ratio(len(pairs), len(expected)),
        "mwe_precision": _ratio(matched_mwe, matched_mwe + sum(x["classification"] == "false_positive" and normalize(x["unit_type"]) == "mwe" for x in out_of_scope)),
        "mwe_recall": _ratio(matched_mwe, expected_mwe),
        "surface_accuracy": _ratio(exact_surfaces, len(paired)),
        "pos_accuracy": _ratio(exact_pos, len(paired)),
        "sense_identity_accuracy": _ratio(exact_sense, len(paired)),
        "definition_accuracy": _ratio(exact_def, len(paired)),
        "official_fr_coverage": _ratio(fr_present, len(actual)),
        "official_fr_soft_accuracy": _ratio(fr_soft, fr_expected),
        "sense_definition_fr_coherence": _ratio(coherent, non_reviewed),
        # With a positive-only benchmark, inventory precision is the measurable
        # pedagogical selection precision; exclusions have no rows to pair.
        "pedagogical_selection_precision": _ratio(len(pairs), precision_denominator),
        "review_decision_accuracy": _ratio(exact_decision, len(paired)),
        "review_queue_size": sum(_truthy(r["needs_review"]) for r in actual),
        "review_rate": _ratio(sum(_truthy(r["needs_review"]) for r in actual), len(actual)),
        "audited_out_of_scope_precision": _ratio(class_counts["validated_improvement"], audited_additions),
    }
    counts = {
        "actual_rows": len(actual), "benchmark_rows": len(expected), "matched_rows": len(pairs),
        "actual_only_rows": len(only_a), "benchmark_only_rows": len(only_e),
        "actual_only_mwe": sum(normalize(actual[i]["unit_type"]) == "mwe" for i in only_a),
        "actual_only_words": sum(normalize(actual[i]["unit_type"]) == "word" for i in only_a),
        "benchmark_only_mwe": sum(normalize(expected[i]["unit_type"]) == "mwe" for i in only_e),
        "benchmark_only_words": sum(normalize(expected[i]["unit_type"]) == "word" for i in only_e),
        "mwe_missing_pos": sum(not normalize(a["pos"]) and bool(normalize(e["pos"])) for a, e in mwe_pairs),
        "mwe_sense_mismatches": sum(normalize(a["sense_id"]) != normalize(e["sense_id"]) for a, e in mwe_pairs),
        "mwe_definition_mismatches": sum(normalize(a["definition_en"]) != normalize(e["definition_en"]) for a, e in mwe_pairs),
        # Baseline d'acceptation consignée avant Q0-1. Elle est affichée à
        # côté des 43 lignes reproductibles, jamais utilisée comme métrique.
        "documented_mwe_definition_mismatches_baseline": 38,
        "word_pos_or_sense_mismatches": sum((normalize(a["pos"]), normalize(a["sense_id"])) != (normalize(e["pos"]), normalize(e["sense_id"])) for a, e in paired if normalize(e["unit_type"]) == "word"),
        "missing_official_fr_filled_by_benchmark": len(missing_fr_fixed),
        "actual_empty_official_fr": len(actual) - fr_present,
        "canonical_equals_official_fr": sum(normalize(r["canonical_form"]) == normalize(r["meaning_fr_official"]) and bool(normalize(r["canonical_form"])) for r in actual),
        "acceptable_variants": class_counts["acceptable_variant"],
        "validated_out_of_scope_improvements": class_counts["validated_improvement"],
        "out_of_scope_needs_review": class_counts["needs_review"],
        "true_false_positives": class_counts["false_positive"],
        "correspondences": matched_exact_count,
        "matched_revisions": len(differences),
    }
    exact_named = []
    for a, e in paired:
        canon = normalize(e["canonical_form"])
        if any(name in canon for name in NAMED_CASES) and not any(
            normalize(a[f]) != normalize(e[f])
            for f in ("surface_forms", "pos", "sense_id", "definition_en", "meaning_fr_official", "needs_review")
        ):
            exact_named.append(summary(e))
    return {
        "schema_version": 1,
        "matching": "normalized (canonical_form, unit_type) multiset; deterministic homonym assignment by surfaces then semantic fields",
        "metrics": metrics,
        "counts": counts,
        "actual_only": out_of_scope,
        "benchmark_only": [summary(expected[i]) for i in only_e],
        "out_of_scope_policy": "An absence from the benchmark defaults to needs_review and never to false_positive.",
        "classifications": {
            "correspondences": matched_exact_count,
            "acceptable_variants": class_counts["acceptable_variant"],
            "validated_improvements": class_counts["validated_improvement"],
            "revisions": len(differences) + class_counts["needs_review"] + len(only_e),
            "true_false_positives": class_counts["false_positive"],
        },
        "named_gates": {
            "latch": {
                "status": "expected_recovery",
                "present_in_actual": any(normalize(r["canonical_form"]) == "latch" for r in actual),
                "policy": "Expected out-of-scope recovery; never a false positive solely because it is absent from the benchmark.",
            }
        },
        "differences": differences,
        "missing_official_fr_filled": [
            {**summary(a), "benchmark_meaning_fr_official": e["meaning_fr_official"], "benchmark_sense_id": e["sense_id"]}
            for a, e in paired
            if not normalize(a["meaning_fr_official"]) and normalize(e["meaning_fr_official"])
        ],
        "exact_named": exact_named,
    }


NAMED_CASES = ("come to", "let someone go", "burn out", "latch", "affection", "intelligible", "facility", "frosting", "york")


def render_report(result: dict, actual_path: Path, expected_path: Path) -> str:
    c, m = result["counts"], result["metrics"]
    lines = [
        "# Rapport qualité Q0-1", "",
        f"Comparaison en lecture seule de `{actual_path.as_posix()}` à `{expected_path.as_posix()}`.",
        "L'appariement est un appariement de multiensembles par `(canonical_form, unit_type)` normalisé ; les homonymes sont conservés et affectés déterministement selon leurs surfaces puis leurs attributs sémantiques.", "",
        "## Métriques par dimension", "",
        "| Dimension | Résultat | Compte |", "|---|---:|---:|",
    ]
    labels = {
        "unit_precision": "Précision des unités", "unit_recall": "Rappel des unités",
        "mwe_precision": "Précision MWE", "mwe_recall": "Rappel MWE", "surface_accuracy": "Exactitude spans/surfaces", "pos_accuracy": "Exactitude POS",
        "sense_identity_accuracy": "Exactitude identité de sens", "definition_accuracy": "Exactitude définition",
        "official_fr_coverage": "Couverture traduction officielle FR", "official_fr_soft_accuracy": "Exactitude FR souple",
        "sense_definition_fr_coherence": "Cohérence sens–définition–FR (non révisés)",
        "pedagogical_selection_precision": "Précision sélection pédagogique", "review_decision_accuracy": "Exactitude décision garder/réviser", "review_rate": "Taux de révision",
    }
    for key, label in labels.items():
        item = m[key]; value = "n/a" if item["value"] is None else f"{100 * item['value']:.2f} %"
        lines.append(f"| {label} | {value} | {item['numerator']}/{item['denominator']} |")
    lines += [f"| Taille de la file de révision | {m['review_queue_size']} | — |", "", "## Baseline recalculée", "",
              f"- Résultat courant : {c['actual_rows']} lignes ; benchmark : {c['benchmark_rows']} lignes ; appariées : {c['matched_rows']}.",
              f"- Seulement dans le résultat courant : {c['actual_only_rows']} lignes ({c['actual_only_mwe']} MWE, {c['actual_only_words']} mots).",
              f"- Seulement dans le benchmark : {c['benchmark_only_rows']} lignes ({c['benchmark_only_mwe']} MWE, {c['benchmark_only_words']} mots).",
              f"- MWE appariées sans POS alors que le benchmark en fournit un : {c['mwe_missing_pos']}.",
              f"- Identités de sens MWE divergentes : {c['mwe_sense_mismatches']} ; définitions MWE divergentes : {c['mwe_definition_mismatches']}.",
              f"- Mots appariés avec POS ou sens divergent : {c['word_pos_or_sense_mismatches']}.",
              f"- Traductions officielles vides remplies dans le benchmark : {c['missing_official_fr_filled_by_benchmark']} (vides au total dans le résultat : {c['actual_empty_official_fr']}).", "",
              f"- Formes canoniques identiques à leur traduction officielle : {c['canonical_equals_official_fr']}.", "",
              "## Écarts hors benchmark", "",
              "Une absence du benchmark n'est jamais une erreur automatique. Sans adjudication indépendante explicite, elle est classée en révision.", "",
              f"- Variantes acceptables : {c['acceptable_variants']}.",
              f"- Améliorations hors périmètre validées : {c['validated_out_of_scope_improvements']}.",
              f"- Révisions nécessaires : {c['out_of_scope_needs_review']}.",
              f"- Vrais faux positifs audités : {c['true_false_positives']}.",
              f"- Précision auditée des ajouts : {('n/a' if m['audited_out_of_scope_precision']['value'] is None else format(100 * m['audited_out_of_scope_precision']['value'], '.2f') + ' %')}.", "",
              "### Explication des écarts avec les nombres documentés", "",
              "Le plan annonce 43 *unités* seulement dans le résultat (34 MWE, 9 mots). Ce chiffre est retrouvé si l'on transforme les clés en ensemble, mais cette opération écrase les homonymes. Le multiensemble trouve 50 *lignes* (34 MWE, 16 mots), soit 7 lignes homonymes supplémentaires.",
              "Ces 7 appariements expliquent exactement tous les autres écarts : 100 POS MWE manquants au lieu de 98 (+2), 131 sens MWE divergents au lieu de 126 (+5), 43 définitions MWE au lieu de 38 (+5), 16 mots POS/sens au lieu de 14 (+2), et 106 traductions remplies au lieu de 99 (+7). Les nombres documentés correspondent donc au comparateur grossier indexé par clé unique que Q0-1 devait précisément remplacer.", "",
              "## Cas nommés", ""]
    all_entries = result["actual_only"] + result["benchmark_only"] + result["differences"]
    for name in NAMED_CASES:
        hits = [x for x in all_entries if any(
            name in normalize(value)
            for value in (
                x.get("canonical_form", ""), x.get("surface_forms", ""),
                x.get("actual", {}).get("canonical_form", ""), x.get("actual", {}).get("surface_forms", ""),
                x.get("expected", {}).get("canonical_form", ""), x.get("expected", {}).get("surface_forms", ""),
            )
        )]
        if hits:
            kinds = []
            for hit in hits:
                if "fields" in hit:
                    kinds.append("champs divergents : " + ", ".join(hit["fields"]))
                elif hit in result["actual_only"]:
                    kinds.append("seulement dans le résultat courant — " + hit["classification"])
                else:
                    kinds.append("seulement dans le benchmark")
            lines.append(f"- **{name}** — {' ; '.join(kinds)}.")
        else:
            exact = [p for p in result["exact_named"] if name in normalize(p["canonical_form"])]
            if exact:
                lines.append(f"- **{name}** — présent à l'identique dans les deux fichiers ; le benchmark fourni ne matérialise donc pas l'exclusion annoncée par le plan.")
            else:
                suffix = " ; récupération/amélioration attendue, jamais faux positif" if name == "latch" else ""
                lines.append(f"- **{name}** — absent des deux inventaires sous ce canon exact (disparition silencieuse visible dans les deux artefacts){suffix}.")
    lines += ["", "## Traductions officielles manquantes", "",
              f"{c['missing_official_fr_filled_by_benchmark']} lignes appariées ont une traduction vide dans le résultat et renseignée dans le benchmark.", ""]
    for row in result["missing_official_fr_filled"]:
        lines.append(f"- `{row['canonical_form']}` ({row['benchmark_sense_id'] or row['sense_id'] or 'sans sense_id'}) → {row['benchmark_meaning_fr_official']}")
    lines += ["", "## Limites explicites", "",
              "L'exactitude souple FR reconnaît uniquement l'égalité normalisée et les variantes explicitement séparées par `/` ou `;`; elle ne prétend pas juger automatiquement tous les synonymes. La cohérence sémantique est donc un proxy strict fondé sur l'accord avec le triplet annoté du benchmark. La précision pédagogique est mesurable par les lignes conservées, mais le rappel des exclusions demanderait un artefact d'exclusions annoté.", ""]
    return "\n".join(lines)


def run(actual_path: Path, expected_path: Path, json_path: Path, report_path: Path, audit_path: Path | None = None) -> dict:
    result = evaluate(read_csv(actual_path), read_csv(expected_path), read_audit(audit_path))
    # Outputs are separate from both inputs; the benchmark is never opened for writing.
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(result, actual_path, expected_path), encoding="utf-8")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=Path("pipeline_out/vocab.csv"))
    parser.add_argument("--benchmark", type=Path, default=Path("pipeline_out/vocab_corrige.csv"))
    parser.add_argument("--json", type=Path, default=Path("pipeline_out/fix_quality_metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("pipeline_out/fix_quality_report.md"))
    parser.add_argument("--audit", type=Path, default=Path("fix_pipeline/q0_1_out_of_scope_audit.json"))
    args = parser.parse_args(argv)
    if args.benchmark.resolve() in {args.json.resolve(), args.report.resolve()}:
        parser.error("output paths must not overwrite the benchmark")
    run(args.actual, args.benchmark, args.json, args.report, args.audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
