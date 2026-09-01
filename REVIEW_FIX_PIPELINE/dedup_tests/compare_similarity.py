"""Banc de comparaison des méthodes de similarité pour le dédoublonnage
MWE (voir REVIEW_FIX_PIPELINE/RAPPORT/rapport_dedoublonnage.md) : Jaccard
sur gloses normalisées (méthode S3 actuelle,
pipeline/mwe_judge.py::paraphrases_compatible) contre deux modèles
d'embeddings de phrases (sentence-transformers).

Compare UNIQUEMENT sur `definition_en` (jamais sur un champ FR — le
dédoublonnage doit rester utilisable avant S6/traduction, voir le plan).

Usage :
    uv run python REVIEW_FIX_PIPELINE/dedup_tests/compare_similarity.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path("C:/DOCS/_perso/vocab-filter")
sys.path.insert(0, str(ROOT))

SELECTED_MWE_PATH = ROOT / "pipeline_out" / "selected_mwe.jsonl"

from pipeline.mwe_judge import normalize_sense_gloss  # noqa: E402

# --------------------------------------------------
# Modèles d'embeddings comparés. LaBSE est un modèle de bitext mining
# multilingue (déjà en cache localement, historiquement utilisé ailleurs
# dans ce dépôt pour l'alignement FR/EN) ; all-MiniLM-L6-v2 est un modèle
# STS anglais dédié, plus léger. Les deux comparés côte à côte : rien ne
# garantit a priori que le modèle multilingue soit le meilleur choix ici
# puisque `definition_en` est toujours en anglais.
# --------------------------------------------------

EMBEDDING_MODELS = [
    "sentence-transformers/LaBSE",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# Seuils de similarité balayés (Jaccard et cosinus sont sur la même
# échelle [0,1], comparables directement).
THRESHOLDS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

# Groupes (canonical_form, pos) suivis en détail : les plus gros groupes
# dupliqués mesurés dans pipeline_out/selected_mwe.jsonl (voir le plan),
# plus un cas de non-fusion attendue (deux sens réellement distincts).
WATCH_GROUPS = [
    ("be going to", "VERB"),
    ("have got to", "VERB"),
    ("get it", "VERB"),      # doit se séparer : "comprendre" vs "prendre en charge"
    ("come on", "OTHER"),
    ("no way", "OTHER"),
    ("care package", "NOUN"),
    ("piece of work", "NOUN"),
    ("all right", "OTHER"),
]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cluster_jaccard(rows: list[dict], threshold: float) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for r in rows:
        n = normalize_sense_gloss(r["definition_en"] or "")
        target = next(
            (c for c in clusters if jaccard(n, c[0]["_norm"]) >= threshold), None
        )
        r["_norm"] = n
        if target is None:
            clusters.append([r])
        else:
            target.append(r)
    return clusters


def cluster_embeddings(rows: list[dict], embeddings, threshold: float) -> list[list[dict]]:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    if len(rows) == 1:
        return [rows]
    X = np.array([embeddings[id(r)] for r in rows])
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold,
        metric="cosine", linkage="average",
    ).fit_predict(X)
    clusters: dict[int, list[dict]] = defaultdict(list)
    for r, lab in zip(rows, labels):
        clusters[lab].append(r)
    return list(clusters.values())


def main() -> None:
    with SELECTED_MWE_PATH.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["canonical_form"], r["pos"])].append(r)

    print(f"{len(rows)} unités MWE, {len(groups)} groupes (canon, POS).")
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"{len(dup_groups)} groupes dupliqués ({sum(len(v) for v in dup_groups.values())} lignes).\n")

    # --------------------------------------------------
    # Tableau global : nombre d'unités restantes après fusion, par
    # méthode et par seuil (baseline sans fusion = len(rows)).
    # --------------------------------------------------

    print("=== Unités MWE restantes après fusion, par méthode et seuil ===\n")
    header = f"{'seuil':>6} | {'Jaccard':>8}"
    embeddings_by_model = {}
    for model_name in EMBEDDING_MODELS:
        from sentence_transformers import SentenceTransformer

        print(f"Chargement {model_name} ...", flush=True)
        model = SentenceTransformer(model_name)
        defs = [r["definition_en"] or "" for r in rows]
        vecs = model.encode(defs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        embeddings_by_model[model_name] = {id(r): v for r, v in zip(rows, vecs)}
        header += f" | {model_name.split('/')[-1]:>20}"
    print()
    print(header)
    print("-" * len(header))

    for t in THRESHOLDS:
        jac_total = sum(len(cluster_jaccard(v, t)) for v in groups.values())
        line = f"{t:>6.2f} | {jac_total:>8}"
        for model_name in EMBEDDING_MODELS:
            emb = embeddings_by_model[model_name]
            emb_total = sum(len(cluster_embeddings(v, emb, t)) for v in groups.values())
            line += f" | {emb_total:>20}"
        print(line)

    print(f"\n(baseline sans fusion : {len(rows)} unités)\n")

    # --------------------------------------------------
    # Détail par groupe témoin, méthode la plus prometteuse (embeddings,
    # seuil 0.60 mesuré comme point de fonctionnement dans le plan).
    # --------------------------------------------------

    watch_threshold = 0.60
    for model_name in EMBEDDING_MODELS:
        print(f"=== Cas témoins @ seuil {watch_threshold:.2f} — {model_name} ===\n")
        emb = embeddings_by_model[model_name]
        print(f"{'groupe':30} {'n':>3} {'clusters':>9}")
        for key in WATCH_GROUPS:
            v = groups.get(key)
            if not v:
                continue
            n_clusters = len(cluster_embeddings(v, emb, watch_threshold))
            print(f"{str(key):30} {len(v):>3} {n_clusters:>9}")
        print()

    # --------------------------------------------------
    # Vérification du cas critique "get it" : le modèle retenu doit
    # séparer "comprendre" de "prendre en charge / aller chercher".
    # --------------------------------------------------

    for model_name in EMBEDDING_MODELS:
        emb = embeddings_by_model[model_name]
        v = groups.get(("get it", "VERB"))
        if not v:
            continue
        clusters = cluster_embeddings(v, emb, watch_threshold)
        print(f"--- détail 'get it'/VERB — {model_name} ({len(clusters)} clusters) ---")
        for c in clusters:
            print(f"  [{len(c)}] " + " | ".join((r["definition_en"] or "")[:60] for r in c[:3]))
        print()


if __name__ == "__main__":
    main()
