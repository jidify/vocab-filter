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

Traitement PAR LOTS (--batch-max-phrases, défaut 50), même principe que
translate_word_context.py::build_batches : au lieu d'un appel par candidat,
les lignes du CSV d'entrée sont regroupées en lots dont la somme des
nb_phrases reste sous ce seuil. Règle d'accumulation :
    lot = [ligne courante] ; total = nb_phrases(ligne)
    si total >= seuil -> le lot est cette seule ligne, on ferme
    sinon, tant que (total + nb_phrases(ligne suivante)) < seuil :
        ajouter la ligne suivante au lot ; total += nb_phrases(ligne suivante)
    (dès que total + nb_phrases(suivante) >= seuil, on ferme le lot SANS
    ajouter cette ligne suivante, qui démarre le lot d'après)
--batch-max-phrases 0 repasse en mode séquentiel (1 candidat = 1 appel,
chaque lot y est alors un singleton). Comme extracted_form est un écho
EXACT garanti par construction du prompt (jamais reformulé), le rattachement
analyse<->ligne d'entrée dans un lot se fait dessus, sans ambiguïté même
quand lexicalized_form diffère du candidat.

Un lot d'un seul candidat (par le seuil OU par --batch-max-phrases 0) appelle
directement AnalyseMweSenses ; un lot de plusieurs candidats appelle
AnalyseLotMweSenses (même consigne, entrée/sortie en listes).

CAS PARTICULIER MWE, absent du script mots : un candidat peut légitimement
ne produire AUCUNE analyse (liste vide = non lexicalisé, voir §21 du prompt
de référence), ce qui rend une liste plate de sortie ambiguë en lot — un
candidat absent de la réponse d'un lot peut soit avoir été confirmé vide,
soit simplement oublié par le modèle, indiscernables depuis l'extérieur du
lot. Résolu ainsi :
  - un lot d'UN SEUL candidat est TOUJOURS entièrement résolu par son propre
    appel (AnalyseMweSenses) : une liste vide y est directement une réponse
    de confiance (pas de rattrapage, pas de gaspillage d'appel) ;
  - dans un lot de PLUSIEURS candidats, tout candidat absent de la réponse
    (AnalyseLotMweSenses) est rejoué INDIVIDUELLEMENT en fin de run — seul un
    appel individuel (AnalyseMweSenses) fait foi pour confirmer une liste
    vide.
Un candidat confirmé vide (par un appel individuel, direct ou de rattrapage)
est journalisé dans un second CSV compagnon (--empty-out, défaut
mwe_analysis_empty.csv, colonne extracted_form) — nécessaire à la REPRISE
ci-dessous, puisqu'une ligne sans aucune analyse n'écrit sinon rien dans le
CSV principal et serait donc indiscernable, au redémarrage, d'une ligne
jamais traitée.

REPRISE : les deux CSV de sortie (principal + vide) font office de journal.
Au démarrage, les candidats déjà présents dans l'un OU l'autre sont exclus du
traitement (--restart pour ignorer les deux et repartir de zéro). Les lignes
sont écrites en streaming, un lot à la fois — jamais en une seule passe
finale — pour qu'une interruption ne perde que ce qui n'a pas encore été
écrit.

Usage :
    uv run python POC/traitement_mwe/claude/translate_mwe_context.py \
        --in POC/traitement_mwe/claude/tests/mwe_contexts_tests.csv \
        --out POC/traitement_mwe/claude/tests/mwe_analysis_test.csv
    uv run python POC/traitement_mwe/claude/translate_mwe_context.py
    # mode séquentiel (1 appel par candidat, comme avant --batch-max-phrases) :
    uv run python POC/traitement_mwe/claude/translate_mwe_context.py --batch-max-phrases 0
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

# POC/traitement_mwe/claude/translate_mwe_context.py -> POC/ est le parent(2).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_pipeline import config, llm_litellm_catgpt  # noqa: E402

DEFAULT_IN_PATH = ROOT / "traitement_mwe" / "claude" / "mwe_contexts.csv"
DEFAULT_OUT_PATH = Path(__file__).parent / "mwe_analysis.csv"
DEFAULT_EMPTY_OUT_PATH = Path(__file__).parent / "mwe_analysis_empty.csv"

MAX_TOKENS = 16000
SENSE_MAX_WORDS = 3
DEFINITION_MAX_WORDS = 35
DEFAULT_BATCH_MAX_PHRASES = 50

CSV_HEADER = [
    "extracted_form", "lexicalized_form", "mwe_type", "compositionality",
    "conventionality", "difficulty_for_non_native", "sense", "definition_en",
    "translations", "example",
]
EMPTY_CSV_HEADER = ["extracted_form"]

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


class AnalyseLotMweSenses(dspy.Signature):
    """Tu es un expert en linguistique anglaise, phraséologie, lexicographie
    et NLP. On te donne PLUSIEURS candidats MWE INDÉPENDANTS les uns des
    autres, chacun avec ses formes de surface et TOUTES les phrases d'un
    livre où il apparaît. Traite CHAQUE candidat séparément, exactement
    comme si tu ne recevais que lui : ne fusionne JAMAIS les sens de deux
    candidats différents, même s'ils se ressemblent. Aucun candidat n'est
    supposé être une MWE valide.

    Pour chaque candidat, applique les mêmes règles que pour un candidat
    unique : analyse occurrence par occurrence (fréquence/cooccurrence ne
    prouve jamais la lexicalisation) ; si le candidat n'est qu'un fragment,
    reconstruis la plus petite MWE complète et indépendamment lexicalisée
    justifiée par le contexte, sans jamais y inclure un simple complément
    syntaxique productif (sujet, objet, complément infinitif ordinaire) ;
    regroupe les sens de façon CONSERVATRICE — ne sépare deux occurrences en
    deux sens que si la contribution sémantique de la MWE elle-même diffère
    réellement, jamais seulement à cause d'un complément, d'un sujet ou
    d'une paraphrase différente (test d'invariance : si tu peux remplacer le
    complément sans changer la contribution sémantique de la MWE, regroupe
    sous un seul sens).

    Ne retiens que des sens réellement attestés par au moins une phrase
    reçue POUR CE CANDIDAT — n'invente pas de sens absent de son contexte,
    et n'invente jamais de phrase d'exemple hors de la phrases_list de ce
    même candidat. Si aucune phrase d'un candidat ne permet d'établir une
    véritable MWE le concernant, ce candidat ne produit AUCUNE analyse dans
    la liste de sortie (il n'y apparaît simplement pas) — ne renvoie jamais
    une MWE spéculative.

    Renvoie TOUTES les analyses de TOUS les candidats du lot dans une seule
    liste plate : chaque analyse porte son propre champ extracted_form,
    identique caractère pour caractère au candidat d'entrée correspondant
    (jamais reformulé) — c'est ce qui permet de la rattacher au bon candidat
    d'entrée."""

    lot: list[MweWithContext] = dspy.InputField(
        description="Candidats indépendants à analyser dans ce lot — ne jamais les confondre ni "
                    "mélanger leurs sens entre eux."
    )
    analyses: list[MweAnalysis] = dspy.OutputField(
        description="Liste PLATE de toutes les analyses de TOUS les candidats du lot — un "
                    "candidat non lexicalisé n'y contribue aucune ligne (pas de placeholder "
                    "vide), un candidat polysémique plusieurs."
    )


# --------------------------------------------------------------------------
# Câblage LM (CatGPT via l'adaptateur LiteLLM de prod) — identique à
# translate_word_context.py::configure_dspy(), plus --no-cache
# --------------------------------------------------------------------------

def configure_dspy(no_cache: bool = False) -> None:
    """dspy.LM(..., cache=True) est le défaut de DSPy — jamais désactivé
    avant l'ajout de --no-cache : le cache est PERSISTANT SUR DISQUE
    (~/.dspy_cache, diskcache), survit aux runs ET aux sessions. Avec
    LLM_TEMPERATURE=0.0, la clé de cache est stable : toute ligne/tout lot
    déjà envoyé une fois est rejoué depuis le cache SANS jamais rappeler
    catgpt, quel que soit le nombre de runs ultérieurs. Les signatures
    AnalyseMweSenses (séquentiel) et AnalyseLotMweSenses (lot) produisent
    des prompts différents donc des clés de cache différentes — pas de
    collision entre les deux modes — mais deux runs du MÊME mode sur les
    MÊMES candidats se partagent, eux, le même cache. --no-cache force
    cache=False pour un test à blanc garanti sans cache."""
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
# Lecture CSV d'entrée
# --------------------------------------------------------------------------

@dataclass
class ContextRow:
    """Une ligne de mwe_contexts.csv — entry est ce qui part vers le LLM,
    nb_phrases sert uniquement à la constitution des lots (build_batches)."""

    entry: MweWithContext
    nb_phrases: int


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
            rows.append(ContextRow(entry=entry, nb_phrases=int(row["nb_phrases"])))
    return rows


def build_batches(rows: list[ContextRow], max_phrases: int) -> list[list[ContextRow]]:
    """Regroupe des lignes consécutives tant que la somme de leurs
    nb_phrases reste sous max_phrases (voir la règle d'accumulation dans le
    docstring du module — identique à translate_word_context.py::
    build_batches). max_phrases <= 0 -> un lot par ligne (mode séquentiel)."""
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
# Lecture des candidats déjà traités (reprise) / écriture incrémentale
# --------------------------------------------------------------------------

def read_done_candidates(out_path: Path, empty_out_path: Path) -> set[str]:
    """Candidats déjà présents (colonne `extracted_form`, casefold) dans le
    CSV principal OU le CSV compagnon des candidats confirmés vides d'un run
    précédent — sert de journal de reprise, voir le docstring du module.
    Les deux fichiers sont nécessaires : un candidat confirmé non lexicalisé
    n'écrit jamais de ligne dans le CSV principal (aucune analyse), il serait
    donc indiscernable d'un candidat jamais traité sans le CSV compagnon."""
    done: set[str] = set()
    for path in (out_path, empty_out_path):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = (row.get("extracted_form") or "").strip()
                if value:
                    done.add(value.casefold())
    return done


def ensure_csv_with_header(path: Path, header: list[str]) -> None:
    """Crée `path` avec son en-tête s'il n'existe pas encore — jamais appelé
    sur un fichier existant (la reprise s'appuie dessus)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header)


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


def append_empty_csv(candidates: list[str], empty_out_path: Path) -> None:
    """Journalise les candidats confirmés non lexicalisés (liste vide) par
    un appel INDIVIDUEL (jamais un appel de lot, voir le docstring du
    module) — seul moyen de reprise fiable pour ces lignes, qui n'écrivent
    sinon rien dans le CSV principal. Même piège BOM-en-mode-append que
    append_analyses_csv : encoding="utf-8" sans -sig ici."""
    if not candidates:
        return
    with empty_out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for candidate in candidates:
            writer.writerow([candidate])


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
# Boucle d'analyse : lots, réconciliation par extracted_form, rattrapage
# individuel (seul chemin fiable pour confirmer une liste vide en lot)
# --------------------------------------------------------------------------

def new_stats(candidates_total: int, candidates_skipped_done: int) -> dict[str, int]:
    return {
        "candidates_total": candidates_total, "candidates_skipped_done": candidates_skipped_done,
        "batches": 0, "batches_failed": 0,
        "analyses": 0, "extracted_form_mismatch": 0, "bad_example": 0,
        "sense_too_long": 0, "definition_too_long": 0,
        "candidates_confirmed_empty": 0,
        "candidates_retried": 0, "candidates_still_missing": 0,
    }


def _handle_single_candidate(
    entry: MweWithContext, analyser, out_path: Path, empty_out_path: Path, stats: dict[str, int],
) -> None:
    """Résout ENTIÈREMENT un candidat via un appel individuel
    (AnalyseMweSenses) : contrairement à un lot de plusieurs candidats, une
    liste vide reçue ici est une réponse de confiance (pas de rattrapage
    nécessaire) — voir le docstring du module."""
    try:
        prediction = analyser(entree=entry)
    except Exception as exc:
        print(f"  échec : {exc!r}")
        stats["candidates_still_missing"] += 1
        return

    analyses = prediction.analyses
    if not analyses:
        print("  -> aucune MWE confirmée (liste vide)")
        stats["candidates_confirmed_empty"] += 1
        append_empty_csv([entry.lemme], empty_out_path)
        return

    for analysis in analyses:
        stats["analyses"] += 1
        check_analysis(analysis, entry, stats)
    append_analyses_csv(analyses, out_path)
    print(f"  -> {len(analyses)} analyse(s) écrite(s)")


def run_batches(
    batches: list[list[ContextRow]], out_path: Path, empty_out_path: Path, stats: dict[str, int],
) -> None:
    unit_analyser = dspy.ChainOfThought(AnalyseMweSenses)
    lot_analyser = dspy.ChainOfThought(AnalyseLotMweSenses)
    pending_retry: list[MweWithContext] = []

    for batch_idx, batch in enumerate(batches, start=1):
        total_phrases = sum(r.nb_phrases for r in batch)
        candidate_list = ", ".join(r.entry.lemme for r in batch)
        print(f"[lot {batch_idx}/{len(batches)}] {len(batch)} candidat(s), "
              f"{total_phrases} phrase(s) : {candidate_list}")
        stats["batches"] += 1

        if len(batch) == 1:
            # Lot d'un seul candidat : toujours entièrement résolu ici, même
            # si le résultat est une liste vide — jamais de rattrapage
            # redondant (voir le docstring du module).
            _handle_single_candidate(batch[0].entry, unit_analyser, out_path, empty_out_path, stats)
            continue

        entries_by_key = {r.entry.lemme.casefold(): r.entry for r in batch}
        try:
            prediction = lot_analyser(lot=[r.entry for r in batch])
        except Exception as exc:  # dégrade : un lot en échec n'arrête pas le run
            print(f"  échec de lot : {exc!r}")
            stats["batches_failed"] += 1
            pending_retry.extend(r.entry for r in batch)
            continue

        analyses = prediction.analyses
        seen_keys: set[str] = set()
        kept: list[MweAnalysis] = []
        for analysis in analyses:
            key = analysis.extracted_form.strip().casefold()
            entry = entries_by_key.get(key)
            if entry is None:
                print(f"  ATTENTION analyse hors-lot ignorée : extracted_form={analysis.extracted_form!r}")
                continue
            seen_keys.add(key)
            stats["analyses"] += 1
            check_analysis(analysis, entry, stats)
            kept.append(analysis)

        # Absent de la réponse : soit confirmé vide, soit oublié par le
        # modèle — indiscernable en lot (voir le docstring du module) ;
        # rejoué individuellement ci-dessous, seul chemin qui tranche.
        missing = [r.entry for r in batch if r.entry.lemme.casefold() not in seen_keys]
        if missing:
            print(f"  {len(missing)} candidat(s) absent(s) de la réponse (à confirmer "
                  f"individuellement) : {', '.join(e.lemme for e in missing)}")
            pending_retry.extend(missing)

        append_analyses_csv(kept, out_path)
        print(f"  -> {len(kept)} analyse(s) écrite(s)")

    if not pending_retry:
        return

    print()
    print(f"=== Rattrapage individuel de {len(pending_retry)} candidat(s) ===")
    for entry in pending_retry:
        stats["candidates_retried"] += 1
        print(f"[rattrapage {stats['candidates_retried']}/{len(pending_retry)}] {entry.lemme}...")
        _handle_single_candidate(entry, unit_analyser, out_path, empty_out_path, stats)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN_PATH),
                         help="Chemin du CSV d'entrée (défaut : mwe_contexts.csv)")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT_PATH),
                         help="Chemin du CSV de sortie (défaut : mwe_analysis.csv) — sert aussi "
                              "de journal de reprise, voir --restart")
    parser.add_argument("--empty-out", dest="empty_out_path", default=str(DEFAULT_EMPTY_OUT_PATH),
                         help="Chemin du CSV des candidats confirmés non lexicalisés (défaut : "
                              "mwe_analysis_empty.csv) — second journal de reprise, voir "
                              "docstring du module")
    parser.add_argument("--limit", type=int, default=0,
                         help="Plafond de candidats considérés depuis l'entrée (0 = tous, défaut)")
    parser.add_argument("--batch-max-phrases", type=int, default=DEFAULT_BATCH_MAX_PHRASES,
                         help="Nombre de phrases visé par lot avant appel groupé à catgpt "
                              "(défaut : 50 ; 0 = mode séquentiel, un appel par candidat)")
    parser.add_argument("--restart", action="store_true",
                         help="Ignore et réécrit les CSV de sortie existants (principal + "
                              "candidats vides) au lieu de reprendre là où le run précédent "
                              "s'est arrêté")
    parser.add_argument("--no-cache", action="store_true",
                         help="Désactive le cache disque persistant de DSPy (~/.dspy_cache) : "
                              "force un appel catgpt réel pour chaque lot/candidat, sans jamais "
                              "rejouer une réponse d'un run précédent (voir configure_dspy)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    empty_out_path = Path(args.empty_out_path)

    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    if args.restart:
        for path in (out_path, empty_out_path):
            if path.exists():
                path.unlink()

    print(f"Entrée : {in_path}")
    rows = read_mwe_contexts(in_path)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"{len(rows)} candidat(s) MWE au total.")

    done = read_done_candidates(out_path, empty_out_path)
    todo = [r for r in rows if r.entry.lemme.casefold() not in done]
    if done:
        print(f"{len(rows) - len(todo)} candidat(s) déjà présent(s) dans {out_path.name}/"
              f"{empty_out_path.name} -> sauté(s).")

    stats = new_stats(candidates_total=len(rows), candidates_skipped_done=len(rows) - len(todo))

    if not todo:
        print("Rien à faire : tous les candidats demandés sont déjà présents dans les CSV de sortie.")
        return 0

    batches = build_batches(todo, args.batch_max_phrases)
    print(f"{len(todo)} candidat(s) à traiter, regroupés en {len(batches)} lot(s) "
          f"(seuil ~{args.batch_max_phrases} phrase(s)/lot).")

    configure_dspy(no_cache=args.no_cache)
    ensure_csv_with_header(out_path, CSV_HEADER)
    ensure_csv_with_header(empty_out_path, EMPTY_CSV_HEADER)
    run_batches(batches, out_path, empty_out_path, stats)

    print()
    print("=== Récapitulatif ===")
    print(f"Candidats au total            : {stats['candidates_total']}")
    print(f"  dont déjà traités (sautés)  : {stats['candidates_skipped_done']}")
    print(f"Lots envoyés                  : {stats['batches']} ({stats['batches_failed']} en échec)")
    print(f"Candidats rattrapés individuellement : {stats['candidates_retried']}")
    print(f"  dont toujours manquants     : {stats['candidates_still_missing']}")
    print(f"Candidats confirmés non lexicalisés (liste vide) : {stats['candidates_confirmed_empty']}")
    print(f"Analyses (sens) produites     : {stats['analyses']}")
    print(f"  dont extracted_form != entrée : {stats['extracted_form_mismatch']}")
    print(f"  dont example hors phrases_list : {stats['bad_example']}")
    print(f"  dont sense > {SENSE_MAX_WORDS} mots : {stats['sense_too_long']}")
    print(f"  dont definition_en > {DEFINITION_MAX_WORDS} mots : {stats['definition_too_long']}")
    print()
    print(f"-> {out_path}")
    print(f"-> {empty_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
