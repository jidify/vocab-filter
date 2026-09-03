"""POC — pour chaque candidat MWE de mwe_contexts.csv (sortie de
POC/traitement_mwe/claude/extract_mwe_contexts.py), reconstruit la véritable
unité lexicalisée, la type, évalue sa compositionnalité/conventionnalité/
difficulté, regroupe ses phrases par SENS DISTINCT et produit pour chaque
sens une définition anglaise dense, ses traductions françaises et un exemple
verbatim.

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production — même statut que translate_word_context.py, dont
il reprend le câblage DSPy/CatGPT et les conventions CSV.

Utilise DSPy (dspy.ChainOfThought) au-dessus de la gateway CatGPT locale, via
l'adaptateur LiteLLM déjà écrit pour la prod (pipeline/llm_litellm_catgpt.py)
— voir translate_word_context.py::configure_dspy(), recopié ici à l'identique
(mêmes pièges catgpt/DSPy documentés là-bas : CATGPT_BASE_URL/API_TOKEN/
TIMEOUT lus depuis pipeline/config.py, pas depuis dspy.LM(...) ; réponse en
ChatAdapter texte à marqueurs, pas JSON schema natif).

Entrée : un CSV colonnes canonical_form,sources,surface_forms,contexte_en,
nb_phrases (format écrit par extract_mwe_contexts.py::write_csv) — seuls
canonical_form/surface_forms/contexte_en sont utilisés ici. surface_forms est
joint par "/", contexte_en par " || " (mêmes séparateurs que l'extraction).

Sortie : un CSV, une ligne par SENS DISTINCT détecté (donc plusieurs lignes
pour un même candidat polysémique, ou aucune si le candidat n'est pas une
MWE valide) — colonnes extracted_form,lexicalized_form,mwe_type,
compositionality,conventionality,difficulty_for_non_native,sense,
definition_en,translations,example, représentation directe de MweAnalysis.
translations est une liste jointe par " | ". PAS de colonne false_friend :
absente de la structure de sortie confirmée par l'utilisateur (voir le plan),
malgré la demande initiale qui l'incluait — écart assumé, à réintégrer plus
tard si besoin.

Champs de sortie — rôle (condensé du prompt de référence phraséologique
fourni par l'utilisateur ; voir les docstrings Pydantic ci-dessous pour le
détail complet) :
  - extracted_form  : écho EXACT du candidat d'entrée (`entree.lemme`),
    jamais reformulé — invariant vérifié par check_analysis ci-dessous.
  - lexicalized_form : la plus petite unité complète et indépendamment
    lexicalisée justifiée par le contexte — peut différer de
    extracted_form quand celui-ci n'est qu'un fragment (ex. la ligne réelle
    "rid of", 1 phrase "let's get rid of some of this", doit reconstruire
    "get rid of").
  - mwe_type, compositionality, conventionality, difficulty_for_non_native :
    typage phraséologique de lexicalized_form (jamais de extracted_form).
  - sense, definition_en, translations, example : mêmes rôles que dans
    LemmeAnalysis (translate_word_context.py), pour CE sens précis.

IMPORTANT — invariant demandé (identique au script mots) : le nombre de
MweAnalysis renvoyées pour un candidat est le nombre de SENS DISTINCTS
réellement attestés par le contexte fourni, jamais un sens par phrase, et
JAMAIS un sens par complément/paraphrase différent (voir le "test
d'invariance sémantique" dans AnalyseMweSenses ci-dessous — le prompt de
référence est explicite : mieux vaut regrouper à tort que sur-séparer). Un
candidat jugé non lexicalisé (aucune phrase ne le confirme comme MWE)
produit une liste VIDE, ce n'est pas une erreur.

Architecture MINIMALE pour ce premier passage (décision actée avec
l'utilisateur, contrairement à translate_word_context.py) : UN APPEL LLM PAR
LIGNE du CSV d'entrée, écriture en streaming, PAS de lots, PAS de reprise/
--restart. Comme extracted_form est un écho garanti par construction du
prompt, un futur passage en lots n'aura aucune ambiguïté de rattachement
analyse<->ligne d'entrée (contrairement à ce qu'aurait donné un rattachement
sur un "lemme" potentiellement reconstruit). À étendre plus tard seulement si
le passage sur les 522 lignes réelles de mwe_contexts.csv s'avère trop lent
avec la gateway catgpt (pilotée par navigateur, donc lente).

Usage :
    uv run python POC/traitement_mwe/claude/translate_mwe_context.py \
        --in POC/traitement_mwe/claude/tests/mwe_contexts_tests.csv \
        --out POC/traitement_mwe/claude/tests/mwe_analysis_test.csv
    uv run python POC/traitement_mwe/claude/translate_mwe_context.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dspy
from pydantic import BaseModel, Field

ROOT = Path("C:/DOCS/_perso/vocab-filter")
sys.path.insert(0, str(ROOT))

from pipeline import config, llm_litellm_catgpt  # noqa: E402

DEFAULT_IN_PATH = ROOT / "POC" / "traitement_mwe" / "claude" / "mwe_contexts.csv"
DEFAULT_OUT_PATH = Path(__file__).parent / "mwe_analysis.csv"

MAX_TOKENS = 16000
SENSE_MAX_WORDS = 3
DEFINITION_MAX_WORDS = 35

CSV_HEADER = [
    "extracted_form", "lexicalized_form", "mwe_type", "compositionality",
    "conventionality", "difficulty_for_non_native", "sense", "definition_en",
    "translations", "example",
]

MweType = Literal[
    "idiom", "phrasal_verb", "proverb", "semi_fixed", "collocation",
    "compound", "formulaic_expression", "other",
]
Compositionality = Literal[
    "fully_compositional", "mostly_compositional", "partially_compositional", "non_compositional",
]
Conventionality = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------
# Modèles DSPy (entrée / sortie)
# --------------------------------------------------------------------------

class MweWithContext(BaseModel):
    """Un candidat MWE extrait par extract_mwe_contexts.py (S0->S1->S2, sans
    jugement sémantique) et son contexte d'usage réel dans le livre — ce que
    le LLM reçoit en entrée. Correspond au `mwe_candidate` du prompt de
    référence : une simple HYPOTHÈSE, jamais une MWE garantie valide."""

    lemme: str = Field(
        description="Candidat MWE brut proposé par le système d'extraction (colonne "
                    "canonical_form de mwe_contexts.csv) — peut être incomplet, mal segmenté, "
                    "trop large, trop étroit, ambigu, un simple fragment d'une MWE plus longue, "
                    "une séquence compositionnelle ordinaire, ou ne pas être une MWE du tout."
    )
    surface_forms: list[str] = Field(
        description="Formes de surface associées à ce candidat par le système d'extraction."
    )
    phrases_list: list[str] = Field(
        description="Phrases réelles du livre contenant ce candidat ou une de ses formes de "
                    "surface — seule source d'autorité linguistique pour l'analyse."
    )


class MweAnalysis(BaseModel):
    """Un SENS distinct réellement attesté par au moins une des phrases
    reçues — ce que le LLM renvoie, un objet par sens (pas par phrase, pas
    par complément différent). Champs et rôle repris du prompt de référence
    phraséologique fourni par l'utilisateur (sections 3 à 19)."""

    extracted_form: str = Field(
        description="Le candidat d'entrée (entree.lemme), conservé EXACTEMENT tel quel — "
                    "jamais reformulé, jamais corrigé, jamais traduit."
    )
    lexicalized_form: str = Field(
        description="La plus petite unité complète et INDÉPENDAMMENT lexicalisée justifiée par "
                    "le contexte. Si extracted_form n'est qu'un fragment d'une MWE lexicalisée "
                    "plus grande (ex. 'rid of' dans 'let's get rid of some of this'), reconstruis "
                    "cette MWE complète ('get rid of') — mais n'y inclus JAMAIS un sujet, objet, "
                    "complément, modificateur ou complément infinitif syntaxique ordinaire (ex. "
                    "'I never get to see her' reste 'get to', jamais 'get to see'). Normalise les "
                    "véritables slots lexicaux variables en 'someone'/'something'/'one's' (jamais "
                    "le mot particulier d'une phrase). Retire le marqueur infinitif 'to' sauf s'il "
                    "est réellement un élément lexical de l'expression (ex. 'give up', pas "
                    "'to give up')."
    )
    mwe_type: MweType = Field(
        description="Type de lexicalized_form (jamais de extracted_form), exactement un parmi : "
                    "idiom (sens substantiellement figuratif/non compositionnel, ex. 'kick the "
                    "bucket') ; phrasal_verb (verbe+particule lexicalisé, ex. 'give up' — pas "
                    "toute séquence verbe+préposition ou verbe+to+infinitif) ; proverb (vérité "
                    "générale/conseil) ; semi_fixed (véritables slots lexicaux variables, ex. "
                    "'be up to someone' — pas un simple complément syntaxique variable) ; "
                    "collocation (association lexicale conventionnelle, sens compositionnel, ex. "
                    "'heavy rain') ; compound (unité nominale multi-mots, ex. 'credit card') ; "
                    "formulaic_expression (formule discursive/sociale récurrente, ex. 'you know "
                    "what') ; other si aucune autre catégorie ne convient. N'utilise jamais idiom "
                    "comme catégorie fourre-tout pour une expression difficile."
    )
    compositionality: Compositionality = Field(
        description="Degré auquel le sens de lexicalized_form est déductible de ses mots pris "
                    "individuellement par un non-natif qui les connaît sans connaître déjà "
                    "l'expression : fully_compositional (déductible directement), "
                    "mostly_compositional (largement déductible, usage un peu conventionnalisé), "
                    "partially_compositional (indices partiels seulement), non_compositional "
                    "(sens conventionnel non déductible). Ne pas confondre avec la complexité "
                    "grammaticale."
    )
    conventionality: Conventionality = Field(
        description="Degré de lexicalisation/conventionnalisation de lexicalized_form (pas de "
                    "extracted_form) en anglais contemporain : low, medium ou high. Une candidate "
                    "fragmentaire peut avoir une forme extraite peu autonome alors que sa forme "
                    "lexicalisée reconstruite a une conventionnalité élevée."
    )
    difficulty_for_non_native: int = Field(
        ge=1, le=5,
        description="Difficulté (1 = très facile à déduire des mots individuels, 5 = très "
                    "difficile/fortement opaque) à retrouver le sens conventionnel de "
                    "lexicalized_form à partir de ses mots individuels. Ne pas attribuer "
                    "automatiquement un score élevé aux phrasal verbs/semi-fixed/collocations : "
                    "un phrasal verb transparent peut valoir 1 ou 2, un idiome opaque 5."
    )
    sense: str = Field(
        description="Descriptif très court de ce sens précis, 3 MOTS MAXIMUM (ex. 'eliminate', "
                    "'have opportunity', 'be angry'). Décrit la contribution sémantique de la MWE "
                    "elle-même, jamais le sens global de la phrase, et n'utilise jamais 'idiom'/"
                    "'phrasal verb'/'figurative meaning' comme valeur."
    )
    definition_en: str = Field(
        description="Définition sémantique précise et autonome de ce sens, 35 MOTS MAXIMUM. "
                    "Compréhensible indépendamment de la MWE, sans fioriture, sans expliquer qu'il "
                    "s'agit d'une expression (pas de 'An expression used when...'), adaptée comme "
                    "entrée autonome pour un modèle d'embedding sémantique : un embedding calculé "
                    "UNIQUEMENT sur ce texte doit permettre de retrouver d'autres mots/expressions "
                    "de sens proche."
    )
    translations: list[str] = Field(
        description="Traductions françaises correspondant précisément à CE sens (jamais à un "
                    "autre sens de la même expression), équivalents naturels plutôt que mot à mot."
    )
    example: str = Field(
        description="EXACTEMENT une des phrases de phrases_list illustrant ce sens, copiée "
                    "VERBATIM caractère pour caractère (jamais paraphrasée, raccourcie, corrigée, "
                    "normalisée, traduite ou reconstruite)."
    )


class AnalyseMweSenses(dspy.Signature):
    """Tu es un expert en linguistique anglaise, phraséologie, lexicographie
    et NLP. On te donne un candidat MWE (expression polylexicale) extrait
    AUTOMATIQUEMENT, avec ses formes de surface et TOUTES les phrases d'un
    livre où il apparaît. Ce candidat n'est qu'une HYPOTHÈSE : ne suppose
    JAMAIS qu'il constitue une MWE valide.

    Pour chaque phrase, détermine occurrence par occurrence quelle
    expression linguistique est réellement réalisée : une MWE autonome, un
    fragment d'une MWE lexicalisée plus longue, une partie d'une autre MWE,
    une séquence compositionnelle ordinaire, une variante morphologique
    d'une MWE, ou une séquence non pertinente. La fréquence ou la
    cooccurrence dans le corpus ne constitue JAMAIS, à elle seule, une
    preuve de lexicalisation. Si la candidate est un fragment, reconstruis
    la plus petite MWE complète et indépendamment lexicalisée qui explique
    l'occurrence — mais n'y inclus jamais un simple complément syntaxique
    productif (sujet, objet, complément infinitif ordinaire) : une séquence
    verbe+to+infinitif n'est pas automatiquement une MWE plus longue.

    REGROUPEMENT DE SENS — sois CONSERVATEUR. Ne crée un nouveau sens que si
    la CONTRIBUTION SÉMANTIQUE DE LA MWE ELLE-MÊME diffère réellement, jamais
    seulement parce que le complément, le sujet, le temps, la paraphrase ou
    le contexte global de la phrase diffère. Applique le test d'invariance :
    si tu remplaces le complément/contexte par un autre élément compatible
    et que la MWE garde essentiellement la même contribution sémantique,
    regroupe les occurrences sous UN SEUL sens. Il vaut mieux regrouper deux
    occurrences qui appartiennent probablement au même sens que créer
    artificiellement deux sens différents.

    Si aucune phrase reçue ne permet d'établir une véritable MWE (ou un
    fragment d'une MWE lexicalisée plus grande) correspondant au candidat,
    renvoie une liste VIDE — ne renvoie jamais une MWE spéculative simplement
    parce qu'une interprétation est théoriquement possible.

    Pour chaque sens distinct réellement attesté, produis une analyse
    complète (voir les champs de MweAnalysis). N'invente jamais un sens, une
    forme lexicalisée ou une phrase d'exemple absents des phrases reçues."""

    entree: MweWithContext = dspy.InputField()
    analyses: list[MweAnalysis] = dspy.OutputField(
        description="Une instance par SENS DISTINCT réellement attesté dans phrases_list — "
                    "jamais une par phrase, jamais une par complément différent. Liste VIDE si "
                    "aucune phrase ne confirme une véritable MWE (ou un fragment d'une MWE "
                    "lexicalisée plus grande) correspondant au candidat."
    )


# --------------------------------------------------------------------------
# Câblage LM (CatGPT via l'adaptateur LiteLLM de prod) — identique à
# translate_word_context.py::configure_dspy()
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
# Lecture CSV d'entrée
# --------------------------------------------------------------------------

@dataclass
class ContextRow:
    entry: MweWithContext


def read_mwe_contexts(path: Path) -> list[ContextRow]:
    rows: list[ContextRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            surface_forms = [s for s in row["surface_forms"].split("/") if s]
            phrases_list = [p for p in row["contexte_en"].split(" || ") if p]
            entry = MweWithContext(
                lemme=row["canonical_form"].strip(),
                surface_forms=surface_forms,
                phrases_list=phrases_list,
            )
            rows.append(ContextRow(entry=entry))
    return rows


# --------------------------------------------------------------------------
# Écriture CSV (streaming, pas de reprise — voir docstring du module)
# --------------------------------------------------------------------------

def ensure_csv_with_header(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_analyses_csv(analyses: list[MweAnalysis], out_path: Path) -> None:
    """Ajoute des lignes à un CSV déjà créé par ensure_csv_with_header.
    encoding="utf-8" (SANS -sig) ici, volontairement : "utf-8-sig" émet un
    \\ufeff en tête de CHAQUE écriture, y compris en mode 'a' — l'utiliser
    ici réécrirait un BOM au milieu du fichier à chaque ligne ajoutée (même
    piège que translate_word_context.py::append_analyses_csv)."""
    if not analyses:
        return
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for a in analyses:
            writer.writerow([
                a.extracted_form,
                a.lexicalized_form,
                a.mwe_type,
                a.compositionality,
                a.conventionality,
                a.difficulty_for_non_native,
                a.sense,
                a.definition_en,
                " | ".join(a.translations),
                a.example,
            ])


# --------------------------------------------------------------------------
# Contrôles a posteriori (déterministes, gratuits — n'altèrent pas la sortie)
# --------------------------------------------------------------------------

def check_analysis(analysis: MweAnalysis, entry: MweWithContext, stats: dict[str, int]) -> None:
    if analysis.extracted_form != entry.lemme:
        print(f"  ATTENTION extracted_form != candidat d'entrée : "
              f"{analysis.extracted_form!r} != {entry.lemme!r}")
        stats["extracted_form_mismatch"] += 1
    if analysis.example not in entry.phrases_list:
        print(f"  ATTENTION example hors phrases_list ({entry.lemme}) : {analysis.example!r}")
        stats["bad_example"] += 1
    if len(analysis.sense.split()) > SENSE_MAX_WORDS:
        print(f"  ATTENTION sense > {SENSE_MAX_WORDS} mots ({entry.lemme}) : {analysis.sense!r}")
        stats["sense_too_long"] += 1
    if len(analysis.definition_en.split()) > DEFINITION_MAX_WORDS:
        print(f"  ATTENTION definition_en > {DEFINITION_MAX_WORDS} mots ({entry.lemme}) : "
              f"{analysis.definition_en!r}")
        stats["definition_too_long"] += 1


# --------------------------------------------------------------------------
# Boucle principale : un appel LLM par ligne, écriture en streaming
# --------------------------------------------------------------------------

def new_stats(rows_total: int) -> dict[str, int]:
    return {
        "rows_total": rows_total, "rows_failed": 0, "rows_empty": 0,
        "analyses": 0, "extracted_form_mismatch": 0, "bad_example": 0,
        "sense_too_long": 0, "definition_too_long": 0,
    }


def run_rows(rows: list[ContextRow], out_path: Path, stats: dict[str, int]) -> None:
    analyser = dspy.ChainOfThought(AnalyseMweSenses)

    for row_idx, row in enumerate(rows, start=1):
        entry = row.entry
        print(f"[{row_idx}/{len(rows)}] {entry.lemme} ({len(entry.phrases_list)} phrase(s))...")

        try:
            prediction = analyser(entree=entry)
        except Exception as exc:  # dégrade : une ligne en échec n'arrête pas le run
            print(f"  échec : {exc!r}")
            stats["rows_failed"] += 1
            continue

        analyses = prediction.analyses
        if not analyses:
            print("  -> aucune MWE confirmée (liste vide)")
            stats["rows_empty"] += 1
            continue

        for analysis in analyses:
            stats["analyses"] += 1
            check_analysis(analysis, entry, stats)

        append_analyses_csv(analyses, out_path)
        print(f"  -> {len(analyses)} analyse(s) écrite(s)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN_PATH),
                         help="Chemin du CSV d'entrée (défaut : mwe_contexts.csv)")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV de sortie (défaut : mwe_analysis.csv) — écrasé à "
                              "chaque run, pas de reprise (voir docstring du module)")
    parser.add_argument("--limit", type=int, default=0,
                         help="Plafond de candidats considérés depuis l'entrée (0 = tous, défaut)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    print(f"Entrée : {in_path}")
    rows = read_mwe_contexts(in_path)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"{len(rows)} candidat(s) MWE au total.")

    stats = new_stats(rows_total=len(rows))

    configure_dspy()
    ensure_csv_with_header(out_path)
    run_rows(rows, out_path, stats)

    print()
    print("=== Récapitulatif ===")
    print(f"Candidats au total           : {stats['rows_total']}")
    print(f"  dont échecs d'appel LLM    : {stats['rows_failed']}")
    print(f"  dont MWE non confirmée (liste vide) : {stats['rows_empty']}")
    print(f"Analyses (sens) produites    : {stats['analyses']}")
    print(f"  dont extracted_form != entrée : {stats['extracted_form_mismatch']}")
    print(f"  dont example hors phrases_list : {stats['bad_example']}")
    print(f"  dont sense > {SENSE_MAX_WORDS} mots : {stats['sense_too_long']}")
    print(f"  dont definition_en > {DEFINITION_MAX_WORDS} mots : {stats['definition_too_long']}")
    print()
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
