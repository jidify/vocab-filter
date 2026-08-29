"""Mesure du dispositif d'arbitrage des traductions françaises.

Ce rapport ne mesure pas une précision humaine absolue : les signaux
``translation_type`` et ``sense_fit`` proviennent du même appel que ``fr``.
Il présente surtout le taux d'accord des sources indépendantes DBnary et
Apertium.

Lit ``pipeline_out/sense_fr_adjudication.csv`` (produit par
``pipeline.sense_fr_adjudicate``) et ``data/sense_fr.jsonl``.

Usage :
    uv run python -m tools.evaluation.eval_sense_fr
"""

from __future__ import annotations

import collections
import csv

from pipeline import config, sense_fr


ADJUDICATION_CSV_PATH = config.OUT_DIR / "sense_fr_adjudication.csv"


def load_audit() -> list[dict]:
    if not ADJUDICATION_CSV_PATH.exists():
        return []
    with ADJUDICATION_CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def report_signal_agreement(rows: list[dict]) -> None:
    print("=== Taux d'accord par signal (candidats pending/auto_llm arbitrés) ===")
    for signal, label in (
        ("resource_match", "omw-fr/WoNeF (auto., voir sense_fr_frontier)"),
        ("dbnary_match", "DBnary (Wiktionnaire, humain, indépendant)"),
        ("apertium_match", "Apertium (dict. humain, indépendant, sens non distingué)"),
    ):
        vals = [r[signal] for r in rows if r.get(signal) not in (None, "")]
        n = len(vals)
        n_true = sum(1 for v in vals if v == "True")
        rate = n_true / n if n else 0.0
        print(f"  {signal:16s} {n_true:>4d}/{n:<4d} ({rate:>4.0%})  — {label}")


def report_by_translation_type() -> None:
    """Rapporte tous les types, y compris hors résidu d'arbitrage."""
    store = sense_fr.load_store()
    print("\n=== Ventilation par translation_type (passe primaire contextuelle) ===")
    counts = collections.Counter(
        entry.get("translation_type") for entry in store.values()
        if entry.get("translation_type")
    )
    total = sum(counts.values())
    if not total:
        print("  (aucune donnée — pipeline/sense_fr_frontier.py n'a pas encore été exécuté avec une clé API)")
        return
    for translation_type, count in counts.most_common():
        marker = " <- seule catégorie verrouillable automatiquement" if translation_type == "equivalence_directe" else ""
        print(f"  {translation_type:20s} {count:>4d}/{total:<4d} ({count / total:.0%}){marker}")


def report_decision_breakdown(rows: list[dict]) -> None:
    print("\n=== Décisions de la dernière passe d'arbitrage (sense_fr_adjudicate) ===")
    counts = collections.Counter(r["decision"] or "(aucune — reste pending/auto_llm)" for r in rows)
    for decision, count in counts.most_common():
        print(f"  {decision:40s} {count}")


def report_store_summary() -> None:
    store = sense_fr.load_store()
    counts = collections.Counter(entry["status"] for entry in store.values())
    print("\n=== Magasin (data/sense_fr.jsonl) ===")
    for status, count in counts.most_common():
        print(f"  {status:20s} {count}")
    exportable_statuses = {"validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged"}
    n_exportable = sum(
        1 for entry in store.values()
        if entry["status"] in exportable_statuses and entry.get("fr")
    )
    print(
        f"  -> {n_exportable}/{len(store)} traduction(s) officielle(s) exportable(s) "
        "(statut verrouillé + fr non vide, voir resolve_official_fr dans score.py)."
    )


def run() -> int:
    rows = load_audit()
    if not rows:
        print(f"Aucun audit trouvé ({ADJUDICATION_CSV_PATH}) — lancer d'abord pipeline/sense_fr_adjudicate.py.")
    else:
        report_signal_agreement(rows)
        report_decision_breakdown(rows)
    report_by_translation_type()
    report_store_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
