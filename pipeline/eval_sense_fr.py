"""Mesure du dispositif d'arbitrage — étape 7 du plan "Valider / corriger
suggested_fr et suggested_fr_alt".

PAS une mesure de précision : le signal le plus disponible (traduction
contextuelle à sens imposé, pipeline/sense_fr_context.py) vient de la
MÊME famille de modèle que la traduction "de dictionnaire" qu'elle sert à
arbitrer — les deux ne sont pas des sources indépendantes au sens du
plan §5.5. Ce que ce rapport mesure est un TAUX D'ACCORD entre deux
lectures (contextuelle vs dictionnaire), et entre chaque signal et la
décision finale — utile pour régler des seuils (jaccard, seuil LaBSE,
nombre de signaux requis) et comparer deux configurations, JAMAIS à
annoncer comme une précision absolue. Les seuls signaux réellement
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
        ("context_match", "passe contextuelle — MÊME famille de modèle, pas indépendant"),
        ("dbnary_match", "DBnary (Wiktionnaire, humain, indépendant)"),
        ("apertium_match", "Apertium (dict. humain, indépendant, sens non distingué)"),
    ):
        vals = [r[signal] for r in rows if r.get(signal) not in (None, "")]
        n = len(vals)
        n_true = sum(1 for v in vals if v == "True")
        rate = n_true / n if n else 0.0
        print(f"  {signal:16s} {n_true:>4d}/{n:<4d} ({rate:>4.0%})  — {label}")


def report_by_translation_type(rows: list[dict]) -> None:
    print("\n=== Ventilation par translation_type (passe contextuelle, §6) ===")
    counts = collections.Counter(r["context_translation_type"] for r in rows if r.get("context_translation_type"))
    total = sum(counts.values())
    if not total:
        print("  (aucune donnée — pipeline/sense_fr_context.py n'a pas encore été exécuté "
              "avec une clé API)")
        return
    for t, c in counts.most_common():
        marker = " <- seule catégorie qui corrobore" if t == "equivalence_directe" else ""
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
        report_by_translation_type(rows)
        report_decision_breakdown(rows)
    report_store_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
