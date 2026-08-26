"""Mesure du dispositif d'arbitrage — étape 7 du plan "Valider / corriger
suggested_fr et suggested_fr_alt".

PAS une mesure de précision : `translation_type`/`sense_fit` viennent du
MÊME appel que `fr` (pipeline/sense_fr_frontier.py, passe primaire et
contextuelle depuis la fusion de l'ancienne pipeline/sense_fr_context.py)
— ce ne sont donc pas des sources indépendantes au sens du plan §5.5, tout
au plus un filtre de cohérence interne déjà appliqué EN AMONT de
sense_fr_adjudicate.py (voir sa docstring). Les seuls signaux réellement
indépendants du dispositif sont DBnary et Apertium (écrits par des
humains, voir pipeline/lex_bilingual.py) — c'est leur taux d'accord avec
le reste qui se rapproche le plus d'une mesure externe.

Lit pipeline_out/sense_fr_adjudication.csv (produit par
pipeline/sense_fr_adjudicate.py — le réexécuter d'abord si absent ou
périmé) et data/sense_fr.jsonl.

Usage :
    uv run python -m pipeline.eval_sense_fr
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
    """Contrairement aux autres rapports de ce module, lit directement le
    magasin plutôt que l'audit : `translation_type` est écrit par
    sense_fr_frontier.py pour CHAQUE sens (pas seulement le résidu
    pending/auto_llm arbitré par sense_fr_adjudicate.py)."""
    store = sense_fr.load_store()
    print("\n=== Ventilation par translation_type (passe primaire contextuelle) ===")
    counts = collections.Counter(e.get("translation_type") for e in store.values() if e.get("translation_type"))
    total = sum(counts.values())
    if not total:
        print("  (aucune donnée — pipeline/sense_fr_frontier.py n'a pas encore été exécuté "
              "avec une clé API)")
        return
    for t, c in counts.most_common():
        marker = " <- seule catégorie verrouillable automatiquement" if t == "equivalence_directe" else ""
        print(f"  {t:20s} {c:>4d}/{total:<4d} ({c / total:.0%}){marker}")


def report_decision_breakdown(rows: list[dict]) -> None:
    print("\n=== Décisions de la dernière passe d'arbitrage (sense_fr_adjudicate) ===")
    counts = collections.Counter(r["decision"] or "(aucune — reste pending/auto_llm)" for r in rows)
    for d, c in counts.most_common():
        print(f"  {d:40s} {c}")


def report_store_summary() -> None:
    store = sense_fr.load_store()
    counts = collections.Counter(e["status"] for e in store.values())
    print("\n=== Magasin (data/sense_fr.jsonl) ===")
    for status, c in counts.most_common():
        print(f"  {status:20s} {c}")
    exportable_statuses = {"validated", "auto_strong", "auto_llm", "auto_corroborated", "auto_judged"}
    n_exportable = sum(1 for e in store.values() if e["status"] in exportable_statuses and e.get("fr"))
    n_total = len(store)
    print(f"  -> {n_exportable}/{n_total} traduction(s) officielle(s) exportable(s) "
          f"(statut verrouillé + fr non vide, voir resolve_official_fr dans score.py).")


def run() -> int:
    rows = load_audit()
    if not rows:
        print(f"Aucun audit trouvé ({ADJUDICATION_CSV_PATH}) — lancer d'abord "
              f"pipeline/sense_fr_adjudicate.py.")
    else:
        report_signal_agreement(rows)
        report_decision_breakdown(rows)
    report_by_translation_type()
    report_store_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
