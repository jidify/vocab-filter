"""POC — pour chaque lemme de word_contexts.csv (sortie de
extract_word_contexts.py), regroupe ses phrases d'exemple en SENS DISTINCTS
et produit, pour chaque sens, une définition anglaise dense, ses traductions
françaises, un exemple verbatim et un drapeau faux-ami.

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production — même statut que extract_word_contexts.py, dont
il reprend la source (word_contexts.csv) et les conventions CSV.

Utilise DSPy (dspy.ChainOfThought) au-dessus de la gateway CatGPT locale, via
l'adaptateur LiteLLM déjà écrit pour la prod (pipeline/llm_litellm_catgpt.py)
— rien ici ne réimplémente l'appel HTTP, seulement l'enregistrement du
provider "catgpt" et la configuration du LM DSPy.

Entrée : un CSV colonnes lemma,pos,false_friend,cefr,zipf,aoa,surface_forms,
phrases,nb_phrases (format écrit par extract_word_contexts.py::write_csv) —
seules lemma/surface_forms/phrases sont utilisées ici. surface_forms est
joint par "/", phrases par " || " (mêmes séparateurs que l'extraction).

Sortie : un CSV, une ligne par sens distinct détecté (donc plusieurs lignes
pour un même lemme polysémique) — colonnes lemme,false_friend,sense,
definition_en,translations,example, représentation directe du modèle
LemmeAnalysis. translations est une liste jointe par " | ".

Important — invariant demandé : le nombre de LemmeAnalysis renvoyées pour un
lemme est le nombre de SENS DISTINCTS attestés dans ses phrases, jamais le
nombre de phrases (ex. 15 phrases mais 3 sens -> 3 LemmeAnalysis).

Pièges catgpt/DSPy identifiés avant l'écriture de ce script (voir le plan) :
  - llm_litellm_catgpt._CatGptLLM.completion() lit CATGPT_BASE_URL/
    CATGPT_API_TOKEN/CATGPT_TIMEOUT directement dans pipeline/config.py — un
    api_key/api_base passé à dspy.LM(...) est sans effet sur cette gateway,
    seules les variables d'environnement CATGPT_* comptent.
  - Le handler force response_format={"type":"json_object"} sur CHAQUE appel
    HTTP vers la gateway, indépendamment de ce que demande l'adaptateur DSPy.
    litellm ne reconnaît pas "response_format" comme paramètre supporté pour
    un provider custom (get_supported_openai_params renvoie None pour
    "catgpt/...") : dspy.JSONAdapter retombe donc sur le format ChatAdapter
    (texte à marqueurs [[ ## champ ## ]]) plutôt que sur un JSON schema
    natif. Validé empiriquement avant d'écrire ce script (voir le rapport
    d'exploration) : la gateway obéit correctement aux instructions
    textuelles de ChatAdapter malgré le hint transport json_object.
  - import litellm charge .env, qui contient PROVIDER=chatgpt — sans effet
    ici : ce script épingle son modèle catgpt/<CATGPT_MODEL> explicitement.

Usage :
    uv run python POC/traitement_word/claude/translate_word_context.py
    uv run python POC/traitement_word/claude/translate_word_context.py \
        --in POC/traitement_word/claude/tests/word_context_test.csv \
        --out POC/traitement_word/claude/tests/word_analysis_test.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import dspy
from pydantic import BaseModel, Field

ROOT = Path("C:/DOCS/_perso/vocab-filter")
sys.path.insert(0, str(ROOT))

from pipeline import config, llm_litellm_catgpt  # noqa: E402

DEFAULT_IN_PATH = Path(__file__).parent / "word_contexts.csv"
DEFAULT_OUT_PATH = Path(__file__).parent / "word_analysis.csv"

MAX_TOKENS = 16000
SENSE_MAX_WORDS = 3
DEFINITION_MAX_WORDS = 35


# --------------------------------------------------------------------------
# Modèles DSPy (entrée / sortie)
# --------------------------------------------------------------------------

class LemmeWithContext(BaseModel):
    """Un lemme retenu par extract_word_contexts.py et son contexte d'usage
    réel dans le livre — ce que le LLM reçoit en entrée."""

    lemme: str = Field(description="Lemme anglais (forme canonique, non fléchie).")
    surface_forms: list[str] = Field(
        description="Formes de surface fléchies sous lesquelles ce lemme apparaît dans le livre."
    )
    phrases_list: list[str] = Field(
        description="Phrases réelles du livre contenant une occurrence de ce lemme "
                     "(sous une de ses formes de surface)."
    )


class LemmeAnalysis(BaseModel):
    """Un SENS distinct de ce lemme, attesté par au moins une des phrases
    reçues — ce que le LLM renvoie, un objet par sens (pas par phrase)."""

    lemme: str = Field(description="Recopié à l'identique depuis l'entrée, jamais reformulé ni traduit.")
    false_friend: bool = Field(
        description="True si CE SENS précis du mot anglais est un faux-ami classique pour un "
                    "francophone (ex. 'actually' au sens 'en fait' n'est pas 'actuellement') — "
                    "jugé sens par sens, pas pour le lemme entier : un même mot peut avoir un "
                    "sens faux-ami et un autre sens qui ne l'est pas."
    )
    sense: str = Field(
        description="Descriptif très court de ce sens précis, 3 MOTS MAXIMUM (ex. 'narrow passage', "
                    "'become aware of'). Sert uniquement à distinguer ce sens des autres sens du "
                    "même lemme dans la sortie, pas une définition."
    )
    definition_en: str = Field(
        description="Définition détaillée en anglais de CE sens précis, dense, sans fioriture, "
                    "35 MOTS MAXIMUM. Doit être autosuffisante et discriminante : un embedding "
                    "calculé UNIQUEMENT sur ce texte doit permettre de retrouver d'autres mots "
                    "ayant pratiquement le même sens — pas de tournure d'introduction du style "
                    "'This word means...', va directement à la définition."
    )
    translations: list[str] = Field(
        description="Traductions françaises correspondant précisément à CE sens (jamais à un autre "
                    "sens du même lemme anglais), triées de la plus courante à la plus rare."
    )
    example: str = Field(
        description="Exactement UNE des phrases de phrases_list illustrant ce sens, recopiée "
                    "VERBATIM caractère pour caractère (jamais reformulée, jamais raccourcie, "
                    "jamais inventée)."
    )


class AnalyseLemmeSenses(dspy.Signature):
    """Tu es lexicographe bilingue anglais-français. On te donne un lemme
    anglais avec ses formes de surface et la liste de TOUTES les phrases
    d'un livre où il apparaît. Ta tâche : regrouper ces occurrences par SENS
    DISTINCT (pas par phrase — si 15 phrases illustrent le même sens, elles
    ne comptent que pour UN SEUL LemmeAnalysis), puis pour chaque sens
    distinct produire une analyse complète (voir les champs de
    LemmeAnalysis). Un lemme monosémique dans ces phrases ne produit qu'UNE
    analyse, même avec de nombreuses phrases. Ne retiens que des sens
    réellement attestés par au moins une phrase reçue — n'invente pas de
    sens absent du contexte fourni, et n'invente jamais de phrase d'exemple
    hors de phrases_list."""

    entree: LemmeWithContext = dspy.InputField()
    analyses: list[LemmeAnalysis] = dspy.OutputField(
        description="Une instance par SENS DISTINCT attesté dans phrases_list — jamais une par "
                    "phrase. Liste à un seul élément si un seul sens est attesté, quel que soit "
                    "le nombre de phrases reçues."
    )


# --------------------------------------------------------------------------
# Câblage LM (CatGPT via l'adaptateur LiteLLM de prod)
# --------------------------------------------------------------------------

def configure_dspy() -> None:
    llm_litellm_catgpt.register()
    lm = dspy.LM(
        model=f"catgpt/{config.CATGPT_MODEL}",
        api_key=config.CATGPT_API_TOKEN,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    dspy.configure(lm=lm)


# --------------------------------------------------------------------------
# Lecture / écriture CSV
# --------------------------------------------------------------------------

def read_word_contexts(path: Path) -> list[LemmeWithContext]:
    entries: list[LemmeWithContext] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            surface_forms = [s for s in row["surface_forms"].split("/") if s]
            phrases_list = [p for p in row["phrases"].split(" || ") if p]
            entries.append(LemmeWithContext(
                lemme=row["lemma"].strip(),
                surface_forms=surface_forms,
                phrases_list=phrases_list,
            ))
    return entries


def write_analyses_csv(analyses: list[LemmeAnalysis], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemme", "false_friend", "sense", "definition_en", "translations", "example"])
        for a in analyses:
            writer.writerow([
                a.lemme,
                "true" if a.false_friend else "false",
                a.sense,
                a.definition_en,
                " | ".join(a.translations),
                a.example,
            ])


# --------------------------------------------------------------------------
# Boucle d'analyse
# --------------------------------------------------------------------------

def analyze_lemmas(
    entries: list[LemmeWithContext], analyser,
) -> tuple[list[LemmeAnalysis], dict[str, int]]:
    stats = {
        "lemmas": 0, "lemmas_failed": 0, "analyses": 0,
        "bad_example": 0, "sense_too_long": 0, "definition_too_long": 0,
        "lemma_mismatch": 0,
    }
    results: list[LemmeAnalysis] = []

    for entry in entries:
        stats["lemmas"] += 1
        print(f"[{stats['lemmas']}/{len(entries)}] {entry.lemme} "
              f"({len(entry.phrases_list)} phrase(s))...")
        try:
            prediction = analyser(entree=entry)
        except Exception as exc:  # dégrade : une ligne en échec n'arrête pas le run
            print(f"  échec : {exc!r}")
            stats["lemmas_failed"] += 1
            continue

        analyses = prediction.analyses
        print(f"  -> {len(analyses)} sens distinct(s)")
        for analysis in analyses:
            stats["analyses"] += 1
            if analysis.lemme.strip().casefold() != entry.lemme.strip().casefold():
                print(f"  ATTENTION lemme renvoyé différent : {analysis.lemme!r} != {entry.lemme!r}")
                stats["lemma_mismatch"] += 1
            if analysis.example not in entry.phrases_list:
                print(f"  ATTENTION example absent de phrases_list : {analysis.example!r}")
                stats["bad_example"] += 1
            if len(analysis.sense.split()) > SENSE_MAX_WORDS:
                print(f"  ATTENTION sense > {SENSE_MAX_WORDS} mots : {analysis.sense!r}")
                stats["sense_too_long"] += 1
            if len(analysis.definition_en.split()) > DEFINITION_MAX_WORDS:
                print(f"  ATTENTION definition_en > {DEFINITION_MAX_WORDS} mots : {analysis.definition_en!r}")
                stats["definition_too_long"] += 1
        results.extend(analyses)

    return results, stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN_PATH),
                         help="Chemin du CSV d'entrée (défaut : word_contexts.csv)")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV de sortie (défaut : word_analysis.csv)")
    parser.add_argument("--limit", type=int, default=0,
                         help="Plafond de lemmes traités (0 = tous, défaut)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    print(f"Entrée : {in_path}")
    entries = read_word_contexts(in_path)
    if args.limit > 0:
        entries = entries[:args.limit]
    print(f"{len(entries)} lemme(s) à analyser.")

    configure_dspy()
    analyser = dspy.ChainOfThought(AnalyseLemmeSenses)

    analyses, stats = analyze_lemmas(entries, analyser)
    write_analyses_csv(analyses, out_path)

    print()
    print("=== Récapitulatif ===")
    print(f"Lemmes traités             : {stats['lemmas']}")
    print(f"Lemmes en échec (sautés)   : {stats['lemmas_failed']}")
    print(f"Analyses (sens) produites  : {stats['analyses']}")
    print(f"  dont lemme renvoyé != entrée : {stats['lemma_mismatch']}")
    print(f"  dont example hors phrases_list : {stats['bad_example']}")
    print(f"  dont sense > {SENSE_MAX_WORDS} mots : {stats['sense_too_long']}")
    print(f"  dont definition_en > {DEFINITION_MAX_WORDS} mots : {stats['definition_too_long']}")
    print()
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
