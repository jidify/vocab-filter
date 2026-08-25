"""Orchestrateur du pipeline de sélection de vocabulaire.

Usage :
    uv run python run_pipeline.py                 # toutes les étapes
    uv run python run_pipeline.py --from select    # reprendre à partir de S4
    uv run python run_pipeline.py --only mwe_judge # une seule étape

Étapes, dans l'ordre d'exécution recommandé par le plan
(S0+validation -> S1 -> S4 -> S2/S3 -> S5 -> S6b -> S6/S7) :
"""

from __future__ import annotations

import argparse
import sys

STAGES = [
    ("corpus", "pipeline.corpus"),
    ("analyze", "pipeline.analyze"),
    ("select", "pipeline.select"),
    ("mwe", "pipeline.mwe"),
    ("mwe_judge", "pipeline.mwe_judge"),
    ("select2", "pipeline.select"),  # relancé après mwe_judge pour appliquer les spans réservés
    ("senses", "pipeline.senses"),
    ("sense_fr", "pipeline.sense_fr"),  # S6b : traduction FR de référence, par sense_id
    ("export", "pipeline.export"),
]


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

    for name, module_name in stages:
        print(f"\n=== {name} ({module_name}) ===")
        module = importlib.import_module(module_name)
        code = module.run()
        if code:
            print(f"Étape {name} a échoué (code {code}).")
            return code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
