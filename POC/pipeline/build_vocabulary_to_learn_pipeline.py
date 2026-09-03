"""Orchestrateur du pipeline POC de sélection de vocabulaire — enchaîne
programmatiquement les six étapes jusqu'ici lancées à la main dans
POC/traitement_word/, POC/traitement_mwe/, POC/traitement_merge/ et
POC/traitement_localisation/, chacune étant l'entrée de la suivante :

  1. word_extract   (traitement_word/claude/extract_word_contexts.py)
  2. word_translate  (traitement_word/claude/translate_word_context.py)
  3. mwe_extract    (traitement_mwe/claude/extract_mwe_contexts.py)
  4. mwe_translate   (traitement_mwe/claude/translate_mwe_context.py)
  5. merge          (traitement_merge/merge_word_and_mwe_analysis.py)
  6. localize       (traitement_localisation/localize_words_and_mwe.py)

Ne dépend d'AUCUN fichier hors de POC/ : les six scripts ci-dessus importent
`poc_pipeline` (copie autonome, vendorée depuis `pipeline/` — voir le plan
"Pipeline POC autonome"), jamais `pipeline` de production. `POC/` peut être
déplacé ou copié ailleurs et fonctionner tel quel.

Piège évité (documenté dans le plan) : lancés à la main sans argument, les
scripts 5 et 6 ont des défauts qui pointent soit vers des fixtures de test,
soit vers un nom de fichier qui écrase silencieusement le résultat d'un
autre livre. Cet orchestrateur passe TOUJOURS tous les chemins
explicitement — aucun défaut de script n'est utilisé.

Usage :
    uv run POC/pipeline/build_vocabulary_to_learn_pipeline.py \\
        --file "books_excerpts/The Humans - Stephen Karam - excerpt.txt" \\
        --output "POC/pipeline/out/humans_excerpt_vocab.csv"

    # Livre complet (front matter connu : copyright/sommaire/distribution
    # jusqu'à la ligne 182 avant les épigraphes) :
    uv run POC/pipeline/build_vocabulary_to_learn_pipeline.py \\
        --file "books/The Humans - Stephen Karam.txt" --skip-lines 182 \\
        --output "POC/pipeline/out/humans_full_vocab.csv"

    # Reprendre à partir d'une étape, ou n'en lancer qu'une :
    uv run POC/pipeline/build_vocabulary_to_learn_pipeline.py --file ... --output ... --from mwe_extract
    uv run POC/pipeline/build_vocabulary_to_learn_pipeline.py --file ... --output ... --only word_translate
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(POC_ROOT))
from poc_pipeline import config  # noqa: E402

WORD_DIR = POC_ROOT / "traitement_word" / "claude"
MWE_DIR = POC_ROOT / "traitement_mwe" / "claude"
MERGE_DIR = POC_ROOT / "traitement_merge"
LOCALIZE_DIR = POC_ROOT / "traitement_localisation"

STAGE_NAMES = [
    "word_extract", "word_translate", "mwe_extract", "mwe_translate",
    "merge", "localize",
]

# Étapes déterministes (pas de LLM) : sautées si leur sortie principale
# existe déjà, sauf --force. Les étapes LLM (word_translate, mwe_translate)
# gèrent leur propre reprise en relisant leur CSV de sortie — voir
# read_done_lemmas()/read_done_candidates() dans les scripts respectifs —
# et sont donc toujours relancées.
DETERMINISTIC_STAGES = {"word_extract", "mwe_extract", "merge", "localize"}


def slugify(stem: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-")
    return slug or "book"


@dataclass
class RunPaths:
    work_dir: Path
    output: Path

    def __post_init__(self) -> None:
        self.word_contexts = self.work_dir / "01_word_contexts.csv"
        self.word_analysis = self.work_dir / "02_word_analysis.csv"
        self.mwe_contexts = self.work_dir / "03_mwe_contexts.csv"
        self.mwe_analysis = self.work_dir / "04_mwe_analysis.csv"
        self.mwe_analysis_empty = self.work_dir / "04_mwe_analysis_empty.csv"
        self.merged = self.work_dir / "05_word_and_mwe_analysis.csv"
        self.audit = self.work_dir / "audit"
        self.mwe_exclusions = self.audit / "mwe_exclusions.csv"
        self.cognates_removed = self.audit / "cognates_removed.csv"
        self.mwe_gate_rejections = self.audit / "mwe_gate_rejections.csv"
        self.vpc_candidates_extract = self.audit / "vpc_candidates.jsonl"
        self.rules_plus_candidates_extract = self.audit / "rules_plus_candidates.jsonl"
        self.localisation_unmatched = self.audit / "localisation_unmatched.csv"
        self.zone_layout = self.audit / "zone_layout.json"
        self.vpc_candidates_localize = self.audit / "vpc_candidates_localize.jsonl"
        self.rules_plus_candidates_localize = self.audit / "rules_plus_candidates_localize.jsonl"


@dataclass
class Options:
    file: Path
    output: Path
    skip_lines: int
    work_dir: Path
    force: bool
    restart: bool
    no_cache: bool
    batch_max_phrases: int
    limit: int
    max_phrases: int
    zone_percent: float


def run_step(name: str, script: Path, args: list[str]) -> None:
    """Lance un script POC dans son propre sous-processus (jamais importé
    dans celui de l'orchestrateur — voir docstring du plan : les étapes
    word_extract/mwe_extract chargent spaCy avec des réglages de tokenizer
    différents, et deux scripts mutent `config.VPC_CANDIDATES_PATH` en
    mémoire, non thread-safe)."""

    cmd = [sys.executable, str(script), *args]
    print(f"\n=== {name} : {script.name} ===")
    print("  " + " ".join(f'"{a}"' if " " in a else a for a in cmd[1:]))
    result = subprocess.run(cmd, cwd=str(POC_ROOT))
    if result.returncode != 0:
        raise SystemExit(
            f"Étape {name} a échoué (code {result.returncode}) — voir le log ci-dessus."
        )


def check_catgpt_gateway() -> bool:
    """Sonde CATGPT_BASE_URL avant de lancer les étapes LLM, pour échouer
    avec un message clair plutôt qu'après plusieurs minutes de chargement
    spaCy suivies d'une pile d'exceptions LiteLLM."""

    url = config.CATGPT_BASE_URL + "/models"
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except urllib.error.HTTPError:
        # Répond, même en erreur HTTP (ex. 404 sur /models) -> la passerelle
        # est bien là.
        return True
    except Exception:
        return False


def run_word_extract(opts: Options, paths: RunPaths) -> None:
    if not opts.force and paths.word_contexts.exists():
        print(f"\n=== word_extract : sautée (sortie déjà présente : {paths.word_contexts}) ===")
        return
    run_step("word_extract", WORD_DIR / "extract_word_contexts.py", [
        "--book", str(opts.file),
        "--out", str(paths.word_contexts),
        "--mwe-exclusions-out", str(paths.mwe_exclusions),
        "--cognates-removed-out", str(paths.cognates_removed),
        "--max-phrases", str(opts.max_phrases),
        "--skip-lines", str(opts.skip_lines),
    ])


def run_word_translate(opts: Options, paths: RunPaths) -> None:
    args = [
        "--in", str(paths.word_contexts),
        "--out", str(paths.word_analysis),
        "--limit", str(opts.limit),
        "--batch-max-phrases", str(opts.batch_max_phrases),
    ]
    if opts.restart:
        args.append("--restart")
    if opts.no_cache:
        args.append("--no-cache")
    run_step("word_translate", WORD_DIR / "translate_word_context.py", args)


def run_mwe_extract(opts: Options, paths: RunPaths) -> None:
    if not opts.force and paths.mwe_contexts.exists():
        print(f"\n=== mwe_extract : sautée (sortie déjà présente : {paths.mwe_contexts}) ===")
        return
    run_step("mwe_extract", MWE_DIR / "extract_mwe_contexts.py", [
        "--book", str(opts.file),
        "--out", str(paths.mwe_contexts),
        "--gate-rejections-out", str(paths.mwe_gate_rejections),
        "--vpc-candidates-out", str(paths.vpc_candidates_extract),
        "--rules-plus-candidates-out", str(paths.rules_plus_candidates_extract),
        "--max-phrases", str(opts.max_phrases),
        "--skip-lines", str(opts.skip_lines),
    ])


def run_mwe_translate(opts: Options, paths: RunPaths) -> None:
    args = [
        "--in", str(paths.mwe_contexts),
        "--out", str(paths.mwe_analysis),
        "--empty-out", str(paths.mwe_analysis_empty),
        "--limit", str(opts.limit),
        "--batch-max-phrases", str(opts.batch_max_phrases),
    ]
    if opts.restart:
        args.append("--restart")
    if opts.no_cache:
        args.append("--no-cache")
    run_step("mwe_translate", MWE_DIR / "translate_mwe_context.py", args)


def run_merge(opts: Options, paths: RunPaths) -> None:
    if not opts.force and paths.merged.exists():
        print(f"\n=== merge : sautée (sortie déjà présente : {paths.merged}) ===")
        return
    run_step("merge", MERGE_DIR / "merge_word_and_mwe_analysis.py", [
        "--word-in", str(paths.word_analysis),
        "--mwe-in", str(paths.mwe_analysis),
        "--out", str(paths.merged),
    ])


def run_localize(opts: Options, paths: RunPaths) -> None:
    if not opts.force and paths.output.exists():
        print(f"\n=== localize : sautée (sortie déjà présente : {paths.output}) ===")
        return
    run_step("localize", LOCALIZE_DIR / "localize_words_and_mwe.py", [
        "--book", str(opts.file),
        "--in", str(paths.merged),
        "--out", str(paths.output),
        "--unmatched-out", str(paths.localisation_unmatched),
        "--layout-out", str(paths.zone_layout),
        "--vpc-candidates-out", str(paths.vpc_candidates_localize),
        "--rules-plus-candidates-out", str(paths.rules_plus_candidates_localize),
        "--zone-percent", str(opts.zone_percent),
        "--skip-lines", str(opts.skip_lines),
    ])


STAGES: list[tuple[str, callable]] = [
    ("word_extract", run_word_extract),
    ("word_translate", run_word_translate),
    ("mwe_extract", run_mwe_extract),
    ("mwe_translate", run_mwe_translate),
    ("merge", run_merge),
    ("localize", run_localize),
]

LLM_STAGES = {"word_translate", "mwe_translate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="Livre .txt en entrée")
    parser.add_argument("--output", required=True, help="CSV final localisé")
    parser.add_argument("--skip-lines", type=int, default=0,
                         help="Lignes de tête à ignorer (hors-œuvre) en plus de la détection "
                              "par motifs (0 = aucune, défaut ; 182 pour le livre complet "
                              "The Humans). Voir poc_pipeline/config.py::FRONT_MATTER_SKIP_LINES.")
    parser.add_argument("--work-dir", default=None,
                         help="Répertoire des intermédiaires et de l'audit "
                              "(défaut : POC/pipeline/runs/<slug-du-livre>)")
    parser.add_argument("--from", dest="from_stage", default=None, choices=STAGE_NAMES,
                         help="Reprendre à partir de cette étape")
    parser.add_argument("--only", dest="only_stage", default=None, choices=STAGE_NAMES,
                         help="Ne lancer que cette étape")
    parser.add_argument("--force", action="store_true",
                         help="Rejoue les étapes déterministes même si leur sortie existe déjà")
    parser.add_argument("--restart", action="store_true",
                         help="Étapes LLM : ignore et réécrit leurs CSV de sortie/reprise")
    parser.add_argument("--no-cache", action="store_true",
                         help="Étapes LLM : désactive le cache disque DSPy (~/.dspy_cache)")
    parser.add_argument("--batch-max-phrases", type=int, default=50,
                         help="Étapes LLM : phrases visées par lot avant appel groupé "
                              "(défaut 50 ; 0 = séquentiel, un appel par lemme/candidat)")
    parser.add_argument("--limit", type=int, default=0,
                         help="Étapes LLM : plafond de lemmes/candidats traités (0 = tous)")
    parser.add_argument("--max-phrases", type=int, default=0,
                         help="Plafond de phrases affichées par entrée dans les CSV de "
                              "contextes (0 = toutes, défaut)")
    parser.add_argument("--zone-percent", type=float, default=5.0,
                         help="Taille des tranches de localisation, en %% (défaut 5.0)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Résolus en absolu tout de suite : les scripts d'étape tournent dans un
    # sous-processus dont le cwd est POC_ROOT (voir run_step), donc un chemin
    # relatif au répertoire d'appel de l'utilisateur ne survivrait pas tel quel.
    book_path = Path(args.file).resolve()
    if not book_path.exists():
        print(f"Livre introuvable : {book_path}")
        return 1

    for label, directory in [
        ("extract_word_contexts.py", WORD_DIR / "extract_word_contexts.py"),
        ("translate_word_context.py", WORD_DIR / "translate_word_context.py"),
        ("extract_mwe_contexts.py", MWE_DIR / "extract_mwe_contexts.py"),
        ("translate_mwe_context.py", MWE_DIR / "translate_mwe_context.py"),
        ("merge_word_and_mwe_analysis.py", MERGE_DIR / "merge_word_and_mwe_analysis.py"),
        ("localize_words_and_mwe.py", LOCALIZE_DIR / "localize_words_and_mwe.py"),
    ]:
        if not directory.exists():
            print(f"Script POC introuvable : {directory}")
            return 1

    work_dir = Path(args.work_dir).resolve() if args.work_dir else (
        POC_ROOT / "pipeline" / "runs" / slugify(book_path.stem)
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "audit").mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opts = Options(
        file=book_path, output=output_path, skip_lines=args.skip_lines,
        work_dir=work_dir, force=args.force, restart=args.restart,
        no_cache=args.no_cache, batch_max_phrases=args.batch_max_phrases,
        limit=args.limit, max_phrases=args.max_phrases, zone_percent=args.zone_percent,
    )
    paths = RunPaths(work_dir=work_dir, output=output_path)

    if args.only_stage:
        stages = [(n, fn) for n, fn in STAGES if n == args.only_stage]
    elif args.from_stage:
        idx = STAGE_NAMES.index(args.from_stage)
        stages = STAGES[idx:]
    else:
        stages = STAGES

    if any(n in LLM_STAGES for n, _ in stages):
        if not check_catgpt_gateway():
            print(
                f"Passerelle CatGPT injoignable sur {config.CATGPT_BASE_URL} — "
                "les étapes word_translate/mwe_translate en ont besoin. "
                "Démarre la passerelle (CatGPT-Gateway, pilotée par navigateur) "
                "avant de relancer, ou passe --only pour n'exécuter que les "
                "étapes déterministes (word_extract, mwe_extract, merge, localize)."
            )
            return 1

    print(f"Livre           : {book_path}")
    print(f"Sortie finale   : {output_path}")
    print(f"Répertoire de run : {work_dir}")
    print(f"Lignes ignorées : {opts.skip_lines}")

    for name, fn in stages:
        fn(opts, paths)

    print(f"\nTerminé -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
