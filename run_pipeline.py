"""Orchestrateur du pipeline de sélection de vocabulaire.

Usage :
    uv run python run_pipeline.py                 # toutes les étapes
    uv run python run_pipeline.py --from select    # reprendre à partir de S4
    uv run python run_pipeline.py --only mwe_judge # une seule étape

Étapes, dans l'ordre d'exécution (S0+validation -> S1 -> S2/S3 -> S4 -> S5 ->
S6b -> S6/S7). Lot 3 : select (S4) tourne désormais UNE SEULE FOIS, après
mwe_judge (S3) — le premier passage select avant mwe/mwe_judge était mort
(voir le plan, point 6) puisqu'il ne pouvait réserver aucun span MWE tant
que S2/S3 n'avaient pas tourné :

S5 resout maintenant conjointement lemme/POS/sens dans `senses`, a partir
des analyses sourcees de S1. S6b tourne ensuite en deux temps : sense_fr_frontier (passe primaire et
contextuelle, modèle frontière via LiteLLM — voir sa docstring) puis
sense_fr_adjudicate (arbitrage hors ligne, Stage A seul par défaut ;
Stage B/C restent manuels, --with-backtranslation/--with-judge). Le chemin
ollama local historique (pipeline/sense_fr.py::classify_synset_key) reste
disponible via `uv run python -m pipeline.sense_fr --retry-pending` mais
n'est plus dans l'enchaînement par défaut. `sense_fr_reassign` suit
l'adjudication comme filet structurel standard. Les étapes frontière sont
SAUTÉES proprement (pas d'échec du run) si aucune clé API LiteLLM n'est
disponible (ANTHROPIC_API_KEY/OPENAI_API_KEY, .env ou environnement) —
export retombe alors sur ce qui est déjà dans data/sense_fr.jsonl.
"""

from __future__ import annotations

import argparse
import sys

from pipeline import atomic, config, zones

STAGES = [
    ("corpus", "pipeline.corpus"),
    ("analyze", "pipeline.analyze"),
    ("mwe", "pipeline.mwe"),
    ("mwe_judge", "pipeline.mwe_judge"),
    # Lot 3 (plan, point 6) : le premier passage select (avant mwe/mwe_judge)
    # était mort — select.py ignore les spans MWE tant que mwe_judge n'a pas
    # tourné, donc il ne pouvait rien réserver correctement à ce stade. Un
    # seul passage désormais, après mwe_judge, qui écrit aussi
    # lexical_inventory.jsonl + inventory.sha256 (point E).
    ("select", "pipeline.select"),
    ("senses", "pipeline.senses"),
    ("sense_fr_frontier", "pipeline.sense_fr_frontier"),   # S6b-1 : passe primaire contextuelle
    ("sense_fr_adjudicate", "pipeline.sense_fr_adjudicate"),  # S6b-2 : arbitrage hors ligne (Stage A)
    ("sense_fr_reassign", "pipeline.sense_fr_reassign"),  # filet structurel; S5 conjoint vit dans senses
    ("export", "pipeline.export"),
]

# Sautées proprement (pas d'échec du run) si aucune clé API LiteLLM n'est
# disponible — utile pour rejouer S0-S5/S7 sans dépendre d'un fournisseur
# externe. sense_fr_adjudicate tourne par défaut en Stage A seul (aucun
# appel LLM), donc n'échoue en pratique jamais ici ; inclus quand même par
# prudence si les valeurs par défaut changent un jour.
FRONTIER_STAGES = {"sense_fr_frontier", "sense_fr_adjudicate", "sense_fr_reassign"}

# Lot 6 (plan, Partie 3) : --tranches borne la chaine avale via `senses` —
# corpus/analyze/mwe/mwe_judge/select tournent TOUJOURS sur le livre entier,
# jamais par tranche (l'inventaire doit être figé une seule fois, voir la
# Partie 3 et le principe directeur en tête du plan). Mécaniquement, seul
# `senses.py` a besoin d'un paramètre explicite (`segment_idxs`) : c'est la
# seule étape qui itère occurrence par occurrence. `sense_fr_frontier`/
# `sense_fr_adjudicate`/`export` travaillent par sense_id, pas par
# occurrence — leur périmètre est déjà celui de ce que `senses.jsonl`
# contient au moment du run (Partie 3 : elles "recalculent depuis l'union de
# ce que senses.jsonl et sense_fr.jsonl contiennent"), donc aucun paramètre
# supplémentaire n'est nécessaire pour elles : limiter --tranches à
# senses.py suffit à limiter le travail de TOUTE la chaîne avale.
TRANCHE_SEGMENT_PARAM_STAGES = {"senses"}


def parse_tranches(spec: str) -> set[int] | None:
    """'all' (défaut) -> None (pas de filtre). '1-3' -> {1,2,3}.
    '1,4,7' -> {1,4,7}. Combinable : '1-3,7,10-12'. Les nombres sont des
    ordinaux de zone 1-indexés (`zone-01`, `zone-02`... — voir
    pipeline/zones.py, Lot 5), pas des segment_idx."""

    if spec.strip().lower() == "all":
        return None
    ordinals: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            ordinals.update(range(start, end + 1))
        else:
            ordinals.add(int(part))
    return ordinals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", default=None,
                         help="nom de l'étape à partir de laquelle reprendre")
    parser.add_argument("--only", dest="only_stage", default=None,
                         help="ne lancer qu'une seule étape")
    parser.add_argument(
        "--tranches", dest="tranches", default="all",
        help="Limite senses/sense_fr_frontier/sense_fr_adjudicate/export aux "
             "tranches de zone_layout.json (Lot 5) indiquées : 'all' (défaut), "
             "'1-10', '1,4,7', combinable '1-3,7'. corpus/analyze/mwe/"
             "mwe_judge/select tournent toujours sur le livre entier.",
    )
    # --llm-backend/--llm-model ne couvrent QUE le backend global (ollama/catgpt),
    # utilisé en repli par S3-judge-occurrence, S3-definition-cluster, S5-arbitrate,
    # S6-translate-local et S6-backtranslate-local (registre pipeline/llm_tasks.py,
    # tant qu'aucun VOCAB_LLM_<TASK_ID> dédié n'est posé pour la tâche). Les quatre
    # tâches S6 routées par LiteLLM — S6-translate-frontier, S6-backtranslate,
    # S6-judge-dossier, S6-reassign (sense_fr_frontier.py/sense_fr_adjudicate.py/
    # sense_fr_reassign.py) — ne lisent JAMAIS ces deux options : leur modèle se
    # configure exclusivement via les variables VOCAB_LLM_S6_* (voir le tableau des
    # task_id dans README.md).
    parser.add_argument(
        "--llm-backend", choices=["ollama", "catgpt"], default=None,
        help="Backend global de repli (S3/S5/S6-*-local uniquement, voir README.md "
             "'Configuration multi-modèles par tâche') — sans effet sur S6-translate-"
             "frontier/S6-backtranslate/S6-judge-dossier/S6-reassign.",
    )
    parser.add_argument("--llm-base-url", default=None,
                        help="URL Ollama ou URL /v1 de CatGPT-Gateway (backend global de repli, "
                             "même périmètre que --llm-backend)")
    parser.add_argument(
        "--llm-model", default=None,
        help="Modèle du backend global de repli (S3/S5/S6-*-local uniquement, voir "
             "README.md) — ne configure PAS le modèle des 4 tâches S6 frontière/"
             "adjudication/reassign, chacune ayant son propre slot VOCAB_LLM_S6_*.",
    )
    parser.add_argument("--catgpt-api-token", default=None)
    parser.add_argument("--catgpt-timeout", type=float, default=None)
    args = parser.parse_args()

    config.configure_llm(backend=args.llm_backend, base_url=args.llm_base_url,
                         api_token=args.catgpt_api_token, model=args.llm_model,
                         timeout=args.catgpt_timeout)

    import importlib

    stages = STAGES
    if args.only_stage:
        stages = [s for s in STAGES if s[0] == args.only_stage]
        if not stages:
            print(f"Étape inconnue : {args.only_stage}")
            return 1
    elif args.from_stage:
        names = [s[0] for s in STAGES]
        if args.from_stage not in names:
            print(f"Étape inconnue : {args.from_stage}")
            return 1
        stages = STAGES[names.index(args.from_stage):]

    try:
        tranche_ordinals = parse_tranches(args.tranches)
    except ValueError as exc:
        print(f"--tranches invalide ({args.tranches!r}) : {exc}")
        return 1

    # Calculé seulement si une étape de `stages` en a réellement besoin —
    # sinon un `--only mwe_judge --tranches 1-10` échouerait pour rien sur
    # l'absence de zone_layout.json alors que mwe_judge ne le consomme pas.
    segment_idxs: set[int] | None = None
    if tranche_ordinals is not None and any(
        name in TRANCHE_SEGMENT_PARAM_STAGES for name, _ in stages
    ):
        layout = zones.load()
        if layout is None:
            print(f"--tranches demande un layout de zones ({config.ZONE_LAYOUT_PATH.name}) "
                  f"— lance `analyze` au moins une fois avant (Lot 5).")
            return 1
        segment_idxs = zones.segment_idxs_for_tranches(layout, tranche_ordinals)
        if not segment_idxs:
            print(f"--tranches {args.tranches!r} ne correspond à aucune zone du layout "
                  f"courant ({layout['zone_count']} zones) — rien à calculer.")

    # Lot 0 — verrou de run (pipeline/atomic.py::run_lock) : empêche un
    # second run de démarrer pendant qu'un premier écrit encore dans
    # pipeline_out/ (voir le diagnostic de corruption de senses.jsonl dans
    # atomic.py). Un verrou plus vieux que LOCK_STALE_SECONDS est considéré
    # abandonné plutôt que bloquant.
    try:
        with atomic.run_lock():
            for name, module_name in stages:
                print(f"\n=== {name} ({module_name}) ===")
                module = importlib.import_module(module_name)
                kwargs = {}
                if name in TRANCHE_SEGMENT_PARAM_STAGES and segment_idxs is not None:
                    kwargs["segment_idxs"] = segment_idxs
                try:
                    code = module.run(**kwargs)
                except Exception as exc:
                    if name in FRONTIER_STAGES:
                        import litellm
                        if isinstance(exc, litellm.exceptions.AuthenticationError):
                            print(f"  {name} sautée : aucune clé API LiteLLM disponible ({exc}).")
                            continue
                    raise
                if code:
                    print(f"Étape {name} a échoué (code {code}).")
                    return code
    except atomic.RunLockError as exc:
        print(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
