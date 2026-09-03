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
seuls lemma/surface_forms/phrases/nb_phrases sont utilisés ici. surface_forms
est joint par "/", phrases par " || " (mêmes séparateurs que l'extraction).

Sortie : un CSV, une ligne par sens distinct détecté (donc plusieurs lignes
pour un même lemme polysémique) — colonnes lemme,false_friend,sense,
definition_en,translations,example, représentation directe du modèle
LemmeAnalysis. translations est une liste jointe par " | ".

Important — invariant demandé : le nombre de LemmeAnalysis renvoyées pour un
lemme est le nombre de SENS DISTINCTS attestés dans ses phrases, jamais le
nombre de phrases (ex. 15 phrases mais 3 sens -> 3 LemmeAnalysis). Cet
invariant tient que le lemme soit envoyé seul ou dans un lot (voir plus bas).

Traitement PAR LOTS (--batch-max-phrases, défaut 50) : au lieu d'un appel par
lemme, les lignes du CSV d'entrée sont regroupées en lots dont la somme des
nb_phrases reste sous ce seuil, pour réduire le nombre d'appels à la gateway
CatGPT (pilotée par navigateur, donc lente). Règle d'accumulation :
    lot = [ligne courante] ; total = nb_phrases(ligne)
    si total >= seuil -> le lot est cette seule ligne, on ferme
    sinon, tant que (total + nb_phrases(ligne suivante)) < seuil :
        ajouter la ligne suivante au lot ; total += nb_phrases(ligne suivante)
    (dès que total + nb_phrases(suivante) >= seuil, on ferme le lot SANS
    ajouter cette ligne suivante, qui démarre le lot d'après)
--batch-max-phrases 0 repasse en mode séquentiel (1 lemme = 1 appel, le
comportement d'origine de ce script) — chaque lot y est alors un singleton.
Un lot d'un seul lemme (par le seuil OU par --batch-max-phrases 0) appelle
directement AnalyseLemmeSenses ; un lot de plusieurs lemmes appelle
AnalyseLotLemmeSenses (même consigne, entrée/sortie en listes).

REPRISE : le CSV de sortie fait office de journal. Au démarrage, s'il existe
déjà, ses lemmes sont lus et exclus du traitement (--restart pour l'ignorer
et repartir de zéro). Les lignes sont écrites en streaming, un lot à la fois
— jamais en une seule passe finale — pour qu'une interruption ne perde que
ce qui n'a pas encore été écrit. Un lemme absent de la réponse d'un lot est
rejoué seul, en fin de run, via AnalyseLemmeSenses (rattrapage individuel).

Pièges catgpt/DSPy identifiés avant l'écriture de ce script (voir le plan) :
  - llm_litellm_catgpt._CatGptLLM.completion() lit CATGPT_BASE_URL/
    CATGPT_API_TOKEN/CATGPT_TIMEOUT directement dans pipeline/config.py — un
    api_key/api_base passé à dspy.LM(...) est sans effet sur cette gateway,
    seules les variables d'environnement CATGPT_* comptent. Le payload HTTP
    envoyé au handler ignore aussi max_tokens (jamais lu depuis
    optional_params) : MAX_TOKENS ci-dessous ne contraint donc rien côté
    catgpt, seulement côté d'autres providers LiteLLM éventuels.
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
  - Un lot de nombreux lemmes documentés dans le même appel augmente le
    risque de confusion entre lemmes proches (mesuré ailleurs dans le dépôt
    sur d'autres tâches : pipeline/config.py:322-333, réassignation S6
    dégradée à 40/lot, seul 10/lot validé) — comparer la sortie par lots au
    benchmark séquentiel (tests/word_analysis_test-sequential.csv) avant de
    lancer un run complet.

Usage :
    uv run python POC/pipeline/stages/translate_word_context.py
    uv run python POC/pipeline/stages/translate_word_context.py \
        --in POC/traitement_word/claude/tests/word_context_test.csv \
        --out POC/traitement_word/claude/tests/word_analysis_test-batch.csv
    # mode séquentiel (1 appel par lemme, comme avant --batch-max-phrases) :
    uv run python POC/pipeline/stages/translate_word_context.py --batch-max-phrases 0
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import dspy
from pydantic import BaseModel, Field

# POC/pipeline/stages/translate_word_context.py -> POC/ est le parent(2).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_pipeline import config, llm_litellm_catgpt  # noqa: E402

DEFAULT_IN_PATH = Path(__file__).parent / "word_contexts.csv"
DEFAULT_OUT_PATH = Path(__file__).parent / "word_analysis.csv"

MAX_TOKENS = 16000
SENSE_MAX_WORDS = 3
DEFINITION_MAX_WORDS = 35
DEFAULT_BATCH_MAX_PHRASES = 50

CSV_HEADER = ["lemme", "false_friend", "sense", "definition_en", "translations", "example"]


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


class AnalyseLotLemmeSenses(dspy.Signature):
    """Tu es lexicographe bilingue anglais-français. On te donne PLUSIEURS
    lemmes anglais INDÉPENDANTS les uns des autres, chacun avec ses formes
    de surface et la liste de TOUTES les phrases d'un livre où il apparaît.
    Traite CHAQUE lemme séparément, exactement comme si tu ne recevais que
    lui : ne fusionne JAMAIS les sens de deux lemmes différents, même s'ils
    se ressemblent ou partagent un sens proche. Pour chaque lemme, regroupe
    ses occurrences par SENS DISTINCT (pas par phrase) puis produis une
    analyse complète par sens (voir les champs de LemmeAnalysis) — un lemme
    monosémique ne produit qu'UNE analyse, même avec de nombreuses phrases.
    Ne retiens que des sens réellement attestés par au moins une phrase reçue
    POUR CE LEMME — n'invente pas de sens absent de son contexte, et
    n'invente jamais de phrase d'exemple hors de la phrases_list de ce même
    lemme. Renvoie TOUTES les analyses de TOUS les lemmes du lot dans une
    seule liste plate : chaque analyse porte son propre champ `lemme`, qui
    sert à la rattacher au bon lemme d'entrée."""

    lot: list[LemmeWithContext] = dspy.InputField(
        description="Lemmes indépendants à analyser dans ce lot — ne jamais les confondre ni "
                    "mélanger leurs sens entre eux."
    )
    analyses: list[LemmeAnalysis] = dspy.OutputField(
        description="Liste PLATE de toutes les analyses de TOUS les lemmes du lot — au moins une "
                    "par lemme reçu, plusieurs si un lemme est polysémique."
    )


# --------------------------------------------------------------------------
# Câblage LM (CatGPT via l'adaptateur LiteLLM de prod)
# --------------------------------------------------------------------------

def configure_dspy(no_cache: bool = False) -> None:
    """dspy.LM(..., cache=True) est le défaut de DSPy — jamais désactivé
    avant l'ajout de --no-cache : le cache est PERSISTANT SUR DISQUE
    (~/.dspy_cache, diskcache), survit aux runs ET aux sessions. Avec
    LLM_TEMPERATURE=0.0, la clé de cache est stable : tout lemme/tout lot
    déjà envoyé une fois est rejoué depuis le cache sans jamais rappeler
    catgpt. --no-cache force cache=False pour un test à blanc garanti."""
    llm_litellm_catgpt.register()
    lm = dspy.LM(
        model=f"catgpt/{config.CATGPT_MODEL}",
        api_key=config.CATGPT_API_TOKEN,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        cache=not no_cache,
    )
    dspy.configure(lm=lm)


# --------------------------------------------------------------------------
# Lecture CSV d'entrée + constitution des lots
# --------------------------------------------------------------------------

@dataclass
class ContextRow:
    """Une ligne de word_contexts.csv — entry est ce qui part vers le LLM,
    nb_phrases sert uniquement à la constitution des lots (build_batches)."""

    entry: LemmeWithContext
    nb_phrases: int


def read_word_contexts(path: Path) -> list[ContextRow]:
    rows: list[ContextRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            surface_forms = [s for s in row["surface_forms"].split("/") if s]
            phrases_list = [p for p in row["phrases"].split(" || ") if p]
            entry = LemmeWithContext(
                lemme=row["lemma"].strip(),
                surface_forms=surface_forms,
                phrases_list=phrases_list,
            )
            rows.append(ContextRow(entry=entry, nb_phrases=int(row["nb_phrases"])))
    return rows


def build_batches(rows: list[ContextRow], max_phrases: int) -> list[list[ContextRow]]:
    """Regroupe des lignes consécutives tant que la somme de leurs
    nb_phrases reste sous max_phrases (voir la règle d'accumulation dans le
    docstring du module). max_phrases <= 0 -> un lot par ligne (mode
    séquentiel)."""
    if max_phrases <= 0:
        return [[r] for r in rows]

    batches: list[list[ContextRow]] = []
    i, n = 0, len(rows)
    while i < n:
        batch = [rows[i]]
        total = rows[i].nb_phrases
        i += 1
        while i < n and total + rows[i].nb_phrases < max_phrases:
            batch.append(rows[i])
            total += rows[i].nb_phrases
            i += 1
        batches.append(batch)
    return batches


# --------------------------------------------------------------------------
# Lecture des lemmes déjà traités (reprise) / écriture incrémentale
# --------------------------------------------------------------------------

def read_done_lemmas(out_path: Path) -> set[str]:
    """Lemmes déjà présents (colonne `lemme`, casefold) dans un CSV de
    sortie d'un run précédent — sert de journal de reprise, voir le
    docstring du module."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemme = (row.get("lemme") or "").strip()
            if lemme:
                done.add(lemme.casefold())
    return done


def ensure_csv_with_header(out_path: Path) -> None:
    """Crée le fichier avec son en-tête s'il n'existe pas encore — jamais
    appelé sur un fichier existant (la reprise s'appuie dessus)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def append_analyses_csv(analyses: list[LemmeAnalysis], out_path: Path) -> None:
    """Ajoute des lignes à un CSV déjà créé par ensure_csv_with_header.
    encoding="utf-8" (SANS -sig) ici, volontairement : "utf-8-sig" émet un
    \\ufeff en tête de CHAQUE écriture, y compris en mode 'a' — l'utiliser
    ici réécrirait un BOM au milieu du fichier à chaque lot ajouté."""
    if not analyses:
        return
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
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
# Contrôles a posteriori (déterministes, gratuits — n'altèrent pas la sortie)
# --------------------------------------------------------------------------

def check_analysis(analysis: LemmeAnalysis, entry: LemmeWithContext, stats: dict[str, int]) -> None:
    if analysis.lemme.strip().casefold() != entry.lemme.strip().casefold():
        print(f"  ATTENTION lemme renvoyé différent : {analysis.lemme!r} != {entry.lemme!r}")
        stats["lemma_mismatch"] += 1
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
# Boucle d'analyse : lots, réconciliation, rattrapage individuel
# --------------------------------------------------------------------------

def new_stats(lemmas_total: int, lemmas_skipped_done: int) -> dict[str, int]:
    return {
        "lemmas_total": lemmas_total, "lemmas_skipped_done": lemmas_skipped_done,
        "batches": 0, "batches_failed": 0,
        "analyses": 0, "lemma_mismatch": 0, "bad_example": 0,
        "sense_too_long": 0, "definition_too_long": 0,
        "lemmas_retried": 0, "lemmas_still_missing": 0,
    }


def run_batches(
    batches: list[list[ContextRow]], out_path: Path, stats: dict[str, int],
) -> None:
    unit_analyser = dspy.ChainOfThought(AnalyseLemmeSenses)
    lot_analyser = dspy.ChainOfThought(AnalyseLotLemmeSenses)
    pending_retry: list[LemmeWithContext] = []

    for batch_idx, batch in enumerate(batches, start=1):
        entries_by_key = {r.entry.lemme.casefold(): r.entry for r in batch}
        total_phrases = sum(r.nb_phrases for r in batch)
        lemme_list = ", ".join(r.entry.lemme for r in batch)
        print(f"[lot {batch_idx}/{len(batches)}] {len(batch)} lemme(s), "
              f"{total_phrases} phrase(s) : {lemme_list}")
        stats["batches"] += 1

        try:
            if len(batch) == 1:
                prediction = unit_analyser(entree=batch[0].entry)
            else:
                prediction = lot_analyser(lot=[r.entry for r in batch])
        except Exception as exc:  # dégrade : un lot en échec n'arrête pas le run
            print(f"  échec de lot : {exc!r}")
            stats["batches_failed"] += 1
            pending_retry.extend(r.entry for r in batch)
            continue

        analyses = prediction.analyses
        seen_keys: set[str] = set()
        kept: list[LemmeAnalysis] = []
        for analysis in analyses:
            key = analysis.lemme.strip().casefold()
            entry = entries_by_key.get(key)
            if entry is None:
                print(f"  ATTENTION analyse hors-lot ignorée : lemme={analysis.lemme!r}")
                continue
            seen_keys.add(key)
            stats["analyses"] += 1
            check_analysis(analysis, entry, stats)
            kept.append(analysis)

        missing = [r.entry for r in batch if r.entry.lemme.casefold() not in seen_keys]
        if missing:
            print(f"  ATTENTION {len(missing)} lemme(s) absent(s) de la réponse : "
                  f"{', '.join(e.lemme for e in missing)}")
            pending_retry.extend(missing)

        append_analyses_csv(kept, out_path)
        print(f"  -> {len(kept)} analyse(s) écrite(s)")

    if not pending_retry:
        return

    print()
    print(f"=== Rattrapage individuel de {len(pending_retry)} lemme(s) ===")
    for entry in pending_retry:
        stats["lemmas_retried"] += 1
        print(f"[rattrapage {stats['lemmas_retried']}/{len(pending_retry)}] {entry.lemme}...")
        try:
            prediction = unit_analyser(entree=entry)
        except Exception as exc:
            print(f"  échec : {exc!r}")
            stats["lemmas_still_missing"] += 1
            continue

        analyses = [
            a for a in prediction.analyses
            if a.lemme.strip().casefold() == entry.lemme.strip().casefold()
        ]
        if not analyses:
            print(f"  ATTENTION toujours aucune analyse pour {entry.lemme!r}")
            stats["lemmas_still_missing"] += 1
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
                         help="Chemin du CSV d'entrée (défaut : word_contexts.csv)")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV de sortie (défaut : word_analysis.csv) — sert "
                              "aussi de journal de reprise, voir --restart")
    parser.add_argument("--limit", type=int, default=0,
                         help="Plafond de lemmes considérés depuis l'entrée (0 = tous, défaut)")
    parser.add_argument("--batch-max-phrases", type=int, default=DEFAULT_BATCH_MAX_PHRASES,
                         help="Nombre de phrases visé par lot avant appel groupé à catgpt "
                              "(défaut : 50 ; 0 = mode séquentiel, un appel par lemme)")
    parser.add_argument("--restart", action="store_true",
                         help="Ignore et réécrit le CSV de sortie existant au lieu de reprendre "
                              "là où le run précédent s'est arrêté")
    parser.add_argument("--no-cache", action="store_true",
                         help="Désactive le cache disque persistant de DSPy (~/.dspy_cache) : "
                              "force un appel catgpt réel pour chaque lot/lemme, sans jamais "
                              "rejouer une réponse d'un run précédent (voir configure_dspy)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    if args.restart and out_path.exists():
        out_path.unlink()

    print(f"Entrée : {in_path}")
    rows = read_word_contexts(in_path)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"{len(rows)} lemme(s) au total.")

    done = read_done_lemmas(out_path)
    todo = [r for r in rows if r.entry.lemme.casefold() not in done]
    if done:
        print(f"{len(rows) - len(todo)} lemme(s) déjà présent(s) dans {out_path} -> sauté(s).")

    stats = new_stats(lemmas_total=len(rows), lemmas_skipped_done=len(rows) - len(todo))

    if not todo:
        print("Rien à faire : tous les lemmes demandés sont déjà présents dans le CSV de sortie.")
        return 0

    batches = build_batches(todo, args.batch_max_phrases)
    print(f"{len(todo)} lemme(s) à traiter, regroupés en {len(batches)} lot(s) "
          f"(seuil ~{args.batch_max_phrases} phrase(s)/lot).")

    configure_dspy(no_cache=args.no_cache)
    ensure_csv_with_header(out_path)
    run_batches(batches, out_path, stats)

    print()
    print("=== Récapitulatif ===")
    print(f"Lemmes au total              : {stats['lemmas_total']}")
    print(f"  dont déjà traités (sautés) : {stats['lemmas_skipped_done']}")
    print(f"Lots envoyés                 : {stats['batches']} ({stats['batches_failed']} en échec)")
    print(f"Lemmes rattrapés individuellement : {stats['lemmas_retried']}")
    print(f"  dont toujours manquants    : {stats['lemmas_still_missing']}")
    print(f"Analyses (sens) produites    : {stats['analyses']}")
    print(f"  dont lemme renvoyé != entrée : {stats['lemma_mismatch']}")
    print(f"  dont example hors phrases_list : {stats['bad_example']}")
    print(f"  dont sense > {SENSE_MAX_WORDS} mots : {stats['sense_too_long']}")
    print(f"  dont definition_en > {DEFINITION_MAX_WORDS} mots : {stats['definition_too_long']}")
    print()
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
