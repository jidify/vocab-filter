"""Orchestrateur du pipeline de sélection de vocabulaire.

Usage :
    uv run python run_pipeline.py                 # toutes les étapes
    uv run python run_pipeline.py --from select    # reprendre à partir de S4
    uv run python run_pipeline.py --only mwe_judge # une seule étape

Étapes, dans l'ordre d'exécution recommandé par le plan
(S0+validation -> S1 -> S4 -> S2/S3 -> S5 -> S6b -> S6/S7) :

S6b tourne en deux temps : sense_fr_frontier (passe primaire et
contextuelle, modèle frontière via LiteLLM — voir sa docstring) puis
sense_fr_adjudicate (arbitrage hors ligne, Stage A seul par défaut ;
Stage B/C restent manuels, --with-backtranslation/--with-judge). Le chemin
ollama local historique (pipeline/sense_fr.py::classify_synset_key) reste
disponible via `uv run python -m pipeline.sense_fr --retry-pending` mais
n'est plus dans l'enchaînement par défaut. Les deux étapes S6b sont
SAUTÉES proprement (pas d'échec du run) si aucune clé API LiteLLM n'est
disponible (ANTHROPIC_API_KEY/OPENAI_API_KEY, .env ou environnement) —
export retombe alors sur ce qui est déjà dans data/sense_fr.jsonl.
"""

from __future__ import annotations

import argparse
import sys

from pipeline import atomic

STAGES = [
    ("corpus", "pipeline.corpus"),
    ("analyze", "pipeline.analyze"),
    ("select", "pipeline.select"),
    ("mwe", "pipeline.mwe"),
    ("mwe_judge", "pipeline.mwe_judge"),
    ("select2", "pipeline.select"),  # relancé après mwe_judge pour appliquer les spans réservés
    ("senses", "pipeline.senses"),
    ("sense_fr_frontier", "pipeline.sense_fr_frontier"),   # S6b-1 : passe primaire contextuelle
    ("sense_fr_adjudicate", "pipeline.sense_fr_adjudicate"),  # S6b-2 : arbitrage hors ligne (Stage A)
    ("export", "pipeline.export"),
]

# Sautées proprement (pas d'échec du run) si aucune clé API LiteLLM n'est
# disponible — utile pour rejouer S0-S5/S7 sans dépendre d'un fournisseur
# externe. sense_fr_adjudicate tourne par défaut en Stage A seul (aucun
# appel LLM), donc n'échoue en pratique jamais ici ; inclus quand même par
# prudence si les valeurs par défaut changent un jour.
FRONTIER_STAGES = {"sense_fr_frontier", "sense_fr_adjudicate"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", default=None,
                         help="nom de l'étape à partir de laquelle reprendre")
    parser.add_argument("--only", dest="only_stage", default=None,
                         help="ne lancer qu'une seule étape")
    args = parser.parse_args()

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
                try:
                    code = module.run()
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
