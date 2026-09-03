"""POC — pour chaque expression anglaise (mot ou MWE) du CSV fusionné +
localisé produit par POC/pipeline/stages/localize_words_and_mwe.py, demande à
catgpt 2 ou 3 distracteurs français pour un jeu de traduction à choix
multiples : des mots/expressions FR courants, plausibles comme réponse
erronée, mais qui ne sont EN AUCUN CAS une traduction possible de l'expression
source, quel que soit le sens.

Ne touche à rien dans pipeline/ ni pipeline_out/ : script autonome, jetable,
hors pipeline de production. Reprend le câblage DSPy/CatGPT de
translate_word_context.py::configure_dspy() (mêmes pièges catgpt/DSPy
documentés là-bas : CATGPT_BASE_URL/API_TOKEN/TIMEOUT lus depuis
poc_pipeline/config.py, pas depuis dspy.LM(...) ; réponse en ChatAdapter texte
à marqueurs, pas JSON schema natif).

Entrée : le CSV fusionné + localisé (OUTPUT_COLUMNS de
localize_words_and_mwe.py, 21 colonnes) — voir INPUT_COLUMNS ci-dessous, copie
documentée de ce schéma. Seules trois colonnes sont lues : `type` (word/mwe),
`lemme` (expression source pour type=word), `lexicalized_form` (expression
source pour type=mwe), et `translations` pour le garde-fou anti-traduction.
L'exemple fourni (inputs/vocabulary_input_example.csv) n'a PAS de ligne
d'en-tête ; read_input_csv détecte les deux cas (voir sa docstring).

DÉDOUBLONNAGE AVANT tout appel LLM, décidé avec l'utilisateur : puisqu'aucun
distracteur ne peut correspondre à une traduction quel qu'en soit le sens, les
distracteurs ne dépendent pas du sens — une même expression apparaissant sur
plusieurs lignes (plusieurs sens distincts) ne produit qu'UNE ligne de sortie.
La clé de dédoublonnage est (type, expression.casefold()) — jamais les champs
FR (translations), qui ne servent qu'au garde-fou ci-dessous, jamais de signal
de fusion.

SORTIE — un répertoire PAR FICHIER D'ENTRÉE, jamais un chemin libre, même
principe que POC/pipeline/build_vocabulary_to_learn_pipeline.py (voir sa
docstring : un --out à chemin libre permet à un copier-coller de commande mal
adapté d'écraser silencieusement le résultat d'un autre fichier) :

    POC/distractor/out/<slug-du-fichier-d'entrée>/  (ou --out-dir)
        <slug>-distractors.csv   <- résultat final (CSV_HEADER ci-dessous),
                                     une ligne par expression UNIQUE,
                                     colonnes type, expression, distractors,
                                     nb_distractors — distractors est une
                                     liste jointe par " | " (même convention
                                     que translations dans
                                     translate_word_context.py / le CSV
                                     d'entrée)
        audit/
            distractors_rejected.csv  <- distracteurs rejetés par le garde-fou
                                          anti-traduction pendant CE run,
                                          colonnes expression,cause — cause
                                          est le distracteur rejeté LUI-MÊME
                                          (ex. "silence" pour "beat"), pas une
                                          explication : réutilisable tel quel
                                          dans une consigne de re-soumission
                                          au LLM ("ne pas utiliser <cause>",
                                          voir CONTRÔLES ci-dessous ; la
                                          re-soumission elle-même N'EST PAS
                                          implémentée — ce fichier ne fait
                                          QUE journaliser, décision utilisateur
                                          explicite)

slug = slugify(nom du fichier d'entrée sans extension) — copié tel quel de
build_vocabulary_to_learn_pipeline.py::slugify (minuscules, séparateurs
collapsés en '_'). Ex. "my_file_xxx.csv" -> répertoire "my_file_xxx/",
résultat "my_file_xxx/my_file_xxx-distractors.csv".

Deux paires de signatures DSPy (dspy.ChainOfThought), jamais mélangées dans un
lot : ProposeDistracteursMot/ProposeDistracteursLotMot pour type=word,
ProposeDistracteursMwe/ProposeDistracteursLotMwe pour type=mwe — la variante
MWE étend explicitement la vérification des sens au sens littéral ET au sens
idiomatique/figuré de la locution (voir leurs docstrings ci-dessous, qui
portent le prompt réel).

Traitement PAR LOTS (--batch-size, défaut 50 expressions), homogènes (word et
mwe jamais mélangés dans un même appel) : contrairement aux translate_*, il
n'y a pas de phrases à pondérer ici, le lot se découpe simplement toutes les
--batch-size expressions. --batch-size 0 repasse en mode séquentiel (1
expression = 1 appel).

RÉSERVE (dégradation qualité en gros lot) : le docstring de
translate_word_context.py note une dégradation observée en production dès 40
éléments/lot. La contrainte ici est plus dure (écarter tout distracteur qui
serait une traduction possible dans N'IMPORTE QUEL sens), donc potentiellement
plus sensible à la taille de lot — le défaut reste 50 comme demandé par
l'utilisateur, à comparer en pratique à --batch-size 0 avant un run complet
(voir le plan).

CONTRÔLES a posteriori (check_distractors, motif check_analysis du script
mots) :
  - expression renvoyée != expression d'entrée (casefold) -> comptage seul
  - GARDE-FOU TRADUCTIONS : tout distracteur qui coïncide (strip+casefold)
    avec une des translations connues de cette expression (union de tous ses
    sens) est RETIRÉ de la liste — seul contrôle qui modifie la sortie, la
    contrainte anti-traduction étant le cœur de la demande et l'entrée nous
    donnant gratuitement une partie de la réponse. Journalisé À LA FOIS sur
    stdout (ATTENTION) ET dans out/<slug>/audit/distractors_rejected.csv
    (colonnes expression,cause — voir SORTIE ci-dessus) : PAS de fichier
    d'audit avant cette entrée du docstring — un run réel qui déclenche ce
    garde-fou perdait la trace de l'expression concernée dès que le seul
    ATTENTION stdout défilait hors de l'écran (cas vécu sur "beat"/"silence").
  - distracteurs vides ou dupliqués dans la même ligne -> retirés + comptés,
    JAMAIS journalisés dans distractors_rejected.csv (rien d'utile à faire
    éviter au LLM dans ces deux cas — juste du bruit de formatage).
  - après retraits, hors de [2, 3] -> rattrapage individuel une fois (avec le
    MÊME prompt, sans tenir compte des causes journalisées ci-dessus — voir
    RÉ-SOUMISSION plus bas), puis écrit quand même (jamais perdu
    silencieusement) et compté à part.

RÉ-SOUMISSION AU LLM AVEC LES CAUSES JOURNALISÉES : PAS IMPLÉMENTÉE
(décision utilisateur explicite) — distractors_rejected.csv existe pour
qu'un futur run puisse relire ce journal et rejouer chaque expression
concernée avec une consigne supplémentaire ("ne pas utiliser <cause>"), mais
ce script ne le fait pas lui-même aujourd'hui. Une expression dont TOUS les
distracteurs ont été rejetés (ex. "beat" -> 0 distracteur) reste donc dans le
CSV de sortie avec un compte < 2, sans nouvelle tentative automatique au-delà
du rattrapage individuel déjà décrit ci-dessus (qui rejoue le MÊME prompt,
sans exclure les causes) — à corriger manuellement ou via un futur run outillé
pour ça.

REPRISE : le CSV de sortie (<slug>/<slug>-distractors.csv) fait office de
journal (colonne `expression`, casefold) — --restart pour ignorer et repartir
de zéro : supprime TOUT le répertoire <slug>/ (résultat + audit/ de CE run),
jamais le cache persistant ni son historique inter-run (voir ci-dessous).
Écriture EN STREAMING, un lot à la fois (jamais une seule passe finale).

CACHE PERSISTANT DE DISTRACTEURS (--cache-path, défaut
cache/distractors_cache.csv — SIBLING de out/, PAS dedans), décidé avec
l'utilisateur : une expression déjà traitée pour UN fichier ne redemande
jamais d'appel LLM pour un AUTRE fichier — les distracteurs d'une expression
ne dépendent que de l'expression elle-même (voir DÉDOUBLONNAGE ci-dessus).
C'est le SEUL état qui survit délibérément en dehors de out/<slug>/ : le
mettre sous out/<slug>/ le rendrait spécifique à un seul fichier d'entrée et
casserait la réutilisation inter-fichiers qui est tout l'intérêt de ce cache.
Ni --restart ni --out-dir ne l'affectent. Même format que la sortie
principale (CSV_HEADER), même clé de lecture que le dédoublonnage
((type, expression.casefold())) ; fichier APPEND-ONLY — une expression
recalculée y ajoute une NOUVELLE ligne plutôt que de réécrire l'ancienne ;
read_cache ne garde que la DERNIÈRE occurrence de chaque clé (les entrées
plus anciennes restent dans le fichier, à titre d'historique, mais ne sont
plus lues).

LA SEULE VÉRIFICATION faite sur une valeur en cache avant réutilisation est le
GARDE-FOU ANTI-TRADUCTION (check_distractors ci-dessus), rejoué contre les
`translations` connues de cette expression DANS LE FICHIER COURANT (elles
peuvent différer d'un fichier à l'autre — un nouveau fichier peut attester un
sens donc une traduction absente lors du calcul initial). Aucun autre
contrôle n'est refait (les doublons/bornes ont déjà été validés au moment du
calcul initial). Si AU MOINS UN distracteur en cache correspond désormais à
une traduction connue, l'entrée en cache est ENTIÈREMENT rejetée (pas de
filtrage partiel) : l'expression repart en calcul LLM comme si elle n'était
pas en cache, et le conflit est journalisé — un ATTENTION sur stdout, PLUS une
ligne dans out/<slug>/audit/distractors_rejected.csv (MÊME fichier, MÊMES
colonnes expression,cause que pour les rejets de calcul frais ci-dessus — le
fait générateur diffère, cache ou LLM, mais l'info utile est identique : "ce
distracteur-là est disqualifié pour cette expression"), motif déjà utilisé
dans le pipeline de prod pour ses journaux d'anomalies par livre (ex.
transient/audit/localisation_unmatched.csv de
build_vocabulary_to_learn_pipeline.py) — CE fichier-là, contrairement au
cache, EST vidé par --restart (il fait partie de out/<slug>/, propre à CE
run).

Tout résultat frais (cache manquant OU invalidé), une fois passé par
check_distractors, est ajouté À LA FOIS au CSV de sortie et au cache — un
futur fichier pourra le réutiliser. --ignore-cache saute la LECTURE du cache
(force un appel LLM pour toute expression, même déjà en cache) sans désactiver
son ÉCRITURE.

Usage :
    uv run python POC/distractor/generate_distractors.py
    uv run python POC/distractor/generate_distractors.py --in <fusionné.csv>
    # -> POC/distractor/out/<slug>/<slug>-distractors.csv
    uv run python POC/distractor/generate_distractors.py --dry-run
    # mode séquentiel (1 appel par expression) :
    uv run python POC/distractor/generate_distractors.py --batch-size 0
    # forcer un recalcul LLM en ignorant le cache en lecture :
    uv run python POC/distractor/generate_distractors.py --ignore-cache
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import dspy
from pydantic import BaseModel, Field

# POC/distractor/generate_distractors.py -> POC/ est le parent(1).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poc_pipeline import config, llm_litellm_catgpt  # noqa: E402

DEFAULT_IN_PATH = Path(__file__).parent / "inputs" / "vocabulary_input_example.csv"
# Racine des résultats, un sous-répertoire par fichier d'entrée (voir SORTIE
# dans le docstring du module et RunPaths ci-dessous) — jamais de --out à
# chemin libre, même principe que build_vocabulary_to_learn_pipeline.py.
DEFAULT_OUT_ROOT = Path(__file__).parent / "out"
# Cache PERSISTANT inter-run/inter-FICHIER — voir CACHE PERSISTANT dans le
# docstring du module. SIBLING de out/, délibérément PAS dedans (ni --restart
# ni --out-dir ne doivent pouvoir l'affecter).
DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "distractors_cache.csv"

MAX_TOKENS = 4000
DEFAULT_BATCH_SIZE = 50
MIN_DISTRACTORS = 2
MAX_DISTRACTORS = 3

# Copie documentée du schéma écrit par
# POC/pipeline/stages/localize_words_and_mwe.py::OUTPUT_COLUMNS (INPUT_COLUMNS
# + LOCATION_COLUMNS de ce script-là) — sert de repli positionnel quand le CSV
# d'entrée n'a pas de ligne d'en-tête (cas de l'exemple fourni, voir
# read_input_csv) et de garde-fou de schéma quand il en a une.
INPUT_COLUMNS = [
    "type", "lemme", "extracted_form", "lexicalized_form", "mwe_type", "sense",
    "definition_en", "translations", "false_friend", "compositionality",
    "conventionality", "difficulty_for_non_native", "example",
    "zone_ids", "zone_ordinals", "zone_ranges_pct",
    "first_zone", "last_zone", "nb_occurrences", "nb_zones", "nb_segments",
]

CSV_HEADER = ["type", "expression", "distractors", "nb_distractors"]

# Journal d'audit des distracteurs rejetés par le garde-fou anti-traduction
# (calcul frais OU relecture de cache invalidée — voir CONTRÔLES et CACHE
# PERSISTANT dans le docstring du module) — une ligne par distracteur rejeté.
# `cause` est le distracteur rejeté LUI-MÊME (pas une explication) : pensé
# pour être injecté tel quel dans une future consigne de re-soumission au
# LLM ("ne pas utiliser <cause>") — re-soumission NON implémentée ici.
AUDIT_CSV_HEADER = ["expression", "cause"]


# --------------------------------------------------------------------------
# Résolution des chemins de sortie à partir du fichier d'entrée — voir
# SORTIE dans le docstring du module.
# --------------------------------------------------------------------------

def slugify(stem: str) -> str:
    """Copié à l'identique de
    build_vocabulary_to_learn_pipeline.py::slugify — nomme le répertoire de
    sortie du fichier d'entrée traité : minuscules, séparateurs (espaces,
    tirets...) collapsés en un seul '_'."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug or "vocabulaire"


@dataclass
class RunPaths:
    """Chemins de sortie d'UN run, tous dérivés de output_root
    (out_root/slug) — voir SORTIE dans le docstring du module."""

    output_root: Path
    slug: str

    def __post_init__(self) -> None:
        self.out = self.output_root / f"{self.slug}-distractors.csv"
        self.audit = self.output_root / "audit"
        self.rejected = self.audit / "distractors_rejected.csv"


# --------------------------------------------------------------------------
# Modèles DSPy (entrée / sortie) — 1 champ en entrée, 2 en sortie, imposés
# --------------------------------------------------------------------------

class Expression(BaseModel):
    """Un mot ou une expression (MWE) anglaise source — ce que le LLM reçoit
    en entrée. Jamais de sens ni de traduction fournis : les distracteurs
    doivent être valables quel que soit le sens de l'expression."""

    expression: str = Field(description="Mot ou expression anglaise source.")


class ExpressionDistractors(BaseModel):
    """Les distracteurs français d'une expression — ce que le LLM renvoie,
    un objet par expression reçue."""

    expression: str = Field(description="Recopiée à l'identique depuis l'entrée, jamais reformulée ni traduite.")
    distractors: list[str] = Field(
        description="2 ou 3 distracteurs français, triés du meilleur au moins bon. "
                    "Jamais une traduction possible de expression, quel qu'en soit le sens."
    )


# --------------------------------------------------------------------------
# Signatures DSPy — mots
# --------------------------------------------------------------------------

class ProposeDistracteursMot(dspy.Signature):
    """Tu es lexicographe bilingue anglais-français, concepteur de jeux de
    traduction à choix multiples. Pour le mot anglais reçu (entree.expression),
    propose 2 ou 3 distracteurs en français.

    Contraintes strictes :
      - Le distracteur doit être un mot français courant et réel.
      - Il ne doit en aucun cas être une traduction possible du mot source,
        quel que soit le contexte.
      - Vérifie tous les sens, usages, nuances, registres, expressions
        idiomatiques et emplois figurés possibles du mot source avant de
        retenir un distracteur.
      - Le distracteur ne doit être ni un synonyme, ni un quasi-synonyme, ni
        une traduction contextuelle possible du mot source.
      - Il doit néanmoins être plausible comme réponse erronée dans un jeu de
        traduction : idéalement un mot courant que le joueur pourrait
        raisonnablement choisir.
      - Privilégie des mots de fréquence et de difficulté similaires au mot
        source.
      - Évite les mots trop rares, archaïques, techniques ou manifestement
        absurdes.
      - Si un mot peut être une traduction possible, même dans un contexte
        très spécifique, élimine-le.
      - Si tu hésites sur un distracteur, ne le propose pas et cherche une
        alternative plus sûre.

    Classe tes propositions par qualité décroissante (la meilleure en
    premier) dans distractors. Ne renvoie que les 2 ou 3 meilleurs
    distracteurs.

    CONSIGNE IMPORTANTE ET IMPÉRATIVE : les distracteurs doivent être
    suffisamment éloignés sémantiquement du mot source pour qu'ils ne
    puissent pas être défendus comme une traduction, mais suffisamment
    plausibles pour constituer de vrais pièges dans un QCM."""

    entree: Expression = dspy.InputField()
    resultat: ExpressionDistractors = dspy.OutputField()


class ProposeDistracteursLotMot(dspy.Signature):
    """Même tâche que ProposeDistracteursMot, appliquée à PLUSIEURS mots
    anglais INDÉPENDANTS les uns des autres. Traite CHAQUE mot séparément,
    exactement comme si tu ne recevais que lui : ne mélange JAMAIS les
    distracteurs de deux mots différents, même s'ils se ressemblent ou
    partagent un sens proche. Applique à chaque mot les mêmes contraintes
    strictes que ProposeDistracteursMot (distracteur jamais une traduction
    possible du mot source, quel qu'en soit le sens ou le registre ; 2 ou 3
    distracteurs, classés par qualité décroissante). Renvoie UNE entrée par
    mot reçu, dans une seule liste plate — le champ expression de chaque
    entrée sert à la rattacher au bon mot d'entrée."""

    lot: list[Expression] = dspy.InputField(
        description="Mots indépendants à traiter dans ce lot — ne jamais mélanger leurs distracteurs entre eux."
    )
    resultats: list[ExpressionDistractors] = dspy.OutputField(
        description="Une entrée par mot du lot, dans le même ordre si possible, jamais moins d'entrées que de mots reçus."
    )


# --------------------------------------------------------------------------
# Signatures DSPy — MWE (expressions à plusieurs mots)
# --------------------------------------------------------------------------

class ProposeDistracteursMwe(dspy.Signature):
    """Tu es expert en phraséologie et lexicographie bilingue anglais-français,
    concepteur de jeux de traduction à choix multiples. Pour l'expression
    anglaise reçue (entree.expression — une locution, un verbe à particule,
    une expression figée), propose 2 ou 3 distracteurs en français.

    Contraintes strictes :
      - Le distracteur doit être un mot ou une expression française courante
        et réelle.
      - Il ne doit en aucun cas être une traduction possible de l'expression
        source, ni de son sens LITTÉRAL (mot à mot), ni de son sens
        IDIOMATIQUE/FIGURÉ, quel que soit le contexte.
      - Vérifie tous les sens, usages, nuances, registres et emplois possibles
        de l'expression source (au sens propre comme au sens figuré) avant de
        retenir un distracteur.
      - Le distracteur ne doit être ni un synonyme, ni un quasi-synonyme, ni
        une traduction contextuelle possible de l'expression source, à aucun
        de ses sens.
      - Il doit néanmoins être plausible comme réponse erronée dans un jeu de
        traduction : idéalement un mot ou une expression courante que le
        joueur pourrait raisonnablement choisir.
      - Privilégie des distracteurs de fréquence et de difficulté similaires
        à l'expression source (un mot simple ou une courte expression, pas
        forcément une autre locution).
      - Évite les distracteurs trop rares, archaïques, techniques ou
        manifestement absurdes.
      - Si un mot ou une expression peut être une traduction possible, même
        dans un contexte très spécifique ou un registre familier, élimine-le.
      - Si tu hésites sur un distracteur, ne le propose pas et cherche une
        alternative plus sûre.

    Classe tes propositions par qualité décroissante (la meilleure en
    premier) dans distractors. Ne renvoie que les 2 ou 3 meilleurs
    distracteurs.

    CONSIGNE IMPORTANTE ET IMPÉRATIVE : les distracteurs doivent être
    suffisamment éloignés sémantiquement de l'expression source (sens propre
    ET sens figuré) pour qu'ils ne puissent pas être défendus comme une
    traduction, mais suffisamment plausibles pour constituer de vrais pièges
    dans un QCM."""

    entree: Expression = dspy.InputField()
    resultat: ExpressionDistractors = dspy.OutputField()


class ProposeDistracteursLotMwe(dspy.Signature):
    """Même tâche que ProposeDistracteursMwe, appliquée à PLUSIEURS
    expressions anglaises INDÉPENDANTES les unes des autres. Traite CHAQUE
    expression séparément, exactement comme si tu ne recevais qu'elle : ne
    mélange JAMAIS les distracteurs de deux expressions différentes, même
    si elles se ressemblent ou partagent un sens proche. Applique à chaque
    expression les mêmes contraintes strictes que ProposeDistracteursMwe
    (distracteur jamais une traduction possible de l'expression source, ni au
    sens littéral ni au sens idiomatique/figuré ; 2 ou 3 distracteurs, classés
    par qualité décroissante). Renvoie UNE entrée par expression reçue, dans
    une seule liste plate — le champ expression de chaque entrée sert à la
    rattacher à la bonne expression d'entrée."""

    lot: list[Expression] = dspy.InputField(
        description="Expressions indépendantes à traiter dans ce lot — ne jamais mélanger leurs distracteurs entre elles."
    )
    resultats: list[ExpressionDistractors] = dspy.OutputField(
        description="Une entrée par expression du lot, dans le même ordre si possible, jamais moins d'entrées que d'expressions reçues."
    )


# --------------------------------------------------------------------------
# Câblage LM (CatGPT via l'adaptateur LiteLLM de prod) — recopié à
# l'identique de translate_word_context.py::configure_dspy.
# --------------------------------------------------------------------------

def configure_dspy(no_cache: bool = False) -> None:
    """dspy.LM(..., cache=True) est le défaut de DSPy — jamais désactivé
    avant l'ajout de --no-cache : le cache est PERSISTANT SUR DISQUE
    (~/.dspy_cache, diskcache), survit aux runs ET aux sessions. Avec
    LLM_TEMPERATURE=0.0, la clé de cache est stable : toute expression déjà
    envoyée une fois est rejouée depuis le cache sans jamais rappeler catgpt.
    --no-cache force cache=False pour un test à blanc garanti."""
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
# Lecture CSV d'entrée + dédoublonnage
# --------------------------------------------------------------------------

@dataclass
class VocabEntry:
    """Une expression UNIQUE (type, expression) à traiter — translations
    porte l'union des traductions connues de TOUS les sens de cette
    expression dans le CSV d'entrée, pour le garde-fou anti-traduction."""

    type: str
    expression: str
    translations: set[str]


def _split_translations(raw: str) -> list[str]:
    return [t.strip() for t in raw.split("|") if t.strip()]


def read_input_csv(path: Path) -> tuple[list[VocabEntry], dict[str, int]]:
    """Lit le CSV fusionné + localisé et renvoie les expressions UNIQUES
    (dédoublonnées par (type, expression.casefold()), première occurrence
    conservée, jamais sur les champs FR — voir le docstring du module) dans
    l'ordre de première apparition, plus des compteurs de lecture.

    Détecte l'absence de ligne d'en-tête (cas de l'exemple fourni) : si la
    première cellule de la première ligne vaut "type", lecture en
    DictReader normal ; sinon repli positionnel sur INPUT_COLUMNS."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.reader(f))

    has_header = bool(raw_rows) and raw_rows[0][:1] == ["type"]
    data_rows = raw_rows[1:] if has_header else raw_rows

    entries_by_key: dict[tuple[str, str], VocabEntry] = {}
    order: list[tuple[str, str]] = []
    counters = {"lignes_lues": 0, "lignes_ignorees": 0}

    for raw_row in data_rows:
        counters["lignes_lues"] += 1
        row = dict(zip(INPUT_COLUMNS, raw_row))

        row_type = (row.get("type") or "").strip()
        if row_type == "word":
            expression = (row.get("lemme") or "").strip()
        elif row_type == "mwe":
            expression = (row.get("lexicalized_form") or "").strip()
        else:
            expression = ""

        if row_type not in ("word", "mwe") or not expression:
            counters["lignes_ignorees"] += 1
            continue

        translations = set(_split_translations(row.get("translations") or ""))
        key = (row_type, expression.casefold())
        if key in entries_by_key:
            entries_by_key[key].translations |= translations
        else:
            entries_by_key[key] = VocabEntry(type=row_type, expression=expression, translations=translations)
            order.append(key)

    counters["expressions_uniques"] = len(order)
    counters["doublons_ecartes"] = counters["lignes_lues"] - counters["lignes_ignorees"] - len(order)
    return [entries_by_key[k] for k in order], counters


def build_batches(entries: list[VocabEntry], batch_size: int) -> list[list[VocabEntry]]:
    """Découpe entries (déjà homogène en type — voir main()) en lots de
    batch_size expressions. batch_size <= 0 -> un lot par expression (mode
    séquentiel)."""
    if batch_size <= 0:
        return [[e] for e in entries]
    return [entries[i:i + batch_size] for i in range(0, len(entries), batch_size)]


# --------------------------------------------------------------------------
# Lecture des expressions déjà traitées (reprise) / écriture incrémentale
# --------------------------------------------------------------------------

def read_done_expressions(out_path: Path) -> set[str]:
    """Expressions déjà présentes (colonne `expression`, casefold) dans un
    CSV de sortie d'un run précédent — sert de journal de reprise."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expression = (row.get("expression") or "").strip()
            if expression:
                done.add(expression.casefold())
    return done


def ensure_csv_with_header(out_path: Path, header: list[str] = CSV_HEADER) -> None:
    """Crée le fichier avec son en-tête s'il n'existe pas encore — sur un
    fichier déjà existant, ne fait rien (la reprise ET le cache s'appuient
    là-dessus pour ne jamais réécrire ce qui est déjà présent). Réutilisé
    pour --out, le cache (même header) et l'audit du cache (header dédié)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header)


def read_cache(cache_path: Path) -> dict[tuple[str, str], ExpressionDistractors]:
    """Lit le cache persistant inter-run/inter-livre — voir CACHE PERSISTANT
    dans le docstring du module. Fichier APPEND-ONLY : si une même clé
    (type, expression.casefold()) apparaît plusieurs fois (recalcul après
    invalidation), seule la DERNIÈRE occurrence est conservée."""
    cache: dict[tuple[str, str], ExpressionDistractors] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip()
            expression = (row.get("expression") or "").strip()
            distractors = _split_translations(row.get("distractors") or "")
            if row_type not in ("word", "mwe") or not expression or not distractors:
                continue
            cache[(row_type, expression.casefold())] = ExpressionDistractors(
                expression=expression, distractors=distractors,
            )
    return cache


def append_audit_rows(rows: list[list[str]], audit_path: Path) -> None:
    """Ajoute des lignes au journal d'audit du cache — mêmes règles
    d'encodage que append_results_csv (utf-8 SANS -sig en append, voir
    ci-dessous)."""
    if not rows:
        return
    with audit_path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def append_results_csv(row_type: str, results: list[ExpressionDistractors], out_path: Path) -> None:
    """Ajoute des lignes à un CSV déjà créé par ensure_csv_with_header.
    encoding="utf-8" (SANS -sig) ici, volontairement : "utf-8-sig" émet un
    \ufeff en tête de CHAQUE écriture, y compris en mode 'a' — l'utiliser
    ici réécrirait un BOM au milieu du fichier à chaque lot ajouté."""
    if not results:
        return
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([row_type, r.expression, " | ".join(r.distractors), len(r.distractors)])


# --------------------------------------------------------------------------
# Contrôles a posteriori (garde-fou anti-traduction inclus — seul contrôle
# qui modifie la sortie, voir le docstring du module)
# --------------------------------------------------------------------------

def check_distractors(
    result: ExpressionDistractors, entry: VocabEntry, stats: dict[str, int], rejected_rows: list[list[str]],
) -> ExpressionDistractors:
    """rejected_rows reçoit une ligne [expression, cause] par distracteur
    écarté pour cause de traduction connue (voir AUDIT_CSV_HEADER) — PAS pour
    les distracteurs vides/dupliqués, qui n'ont rien d'utile à faire éviter
    au LLM (voir CONTRÔLES dans le docstring du module). Muté en place :
    l'appelant l'écrit dans out/<slug>/audit/distractors_rejected.csv."""
    if result.expression.strip().casefold() != entry.expression.strip().casefold():
        print(f"  ATTENTION expression renvoyée différente : {result.expression!r} != {entry.expression!r}")
        stats["expression_mismatch"] += 1

    known_translations = {t.casefold() for t in entry.translations}
    cleaned: list[str] = []
    seen: set[str] = set()
    for d in result.distractors:
        d = d.strip()
        if not d:
            stats["distracteurs_vides_retires"] += 1
            continue
        key = d.casefold()
        if key in known_translations:
            print(f"  ATTENTION distracteur écarté (traduction connue de {entry.expression!r}) : {d!r}")
            stats["distracteurs_traduction_retires"] += 1
            rejected_rows.append([entry.expression, d])
            continue
        if key in seen:
            stats["distracteurs_doublons_retires"] += 1
            continue
        seen.add(key)
        cleaned.append(d)

    if not (MIN_DISTRACTORS <= len(cleaned) <= MAX_DISTRACTORS):
        stats["hors_bornes_apres_controle"] += 1

    return ExpressionDistractors(expression=entry.expression, distractors=cleaned)


# --------------------------------------------------------------------------
# Cache persistant : réutilisation inter-run/inter-livre — voir CACHE
# PERSISTANT dans le docstring du module.
# --------------------------------------------------------------------------

def split_by_cache(
    entries: list[VocabEntry], cache: dict[tuple[str, str], ExpressionDistractors], stats: dict[str, int],
) -> tuple[list[tuple[str, ExpressionDistractors]], list[VocabEntry], list[list[str]]]:
    """Sépare entries en (from_cache, needs_llm, audit_rows) :
      - from_cache : (type, résultat) directement réutilisables sans appel
        LLM — une entrée en cache existe ET repasse le garde-fou
        anti-traduction pour CE livre (SEULE vérification faite sur une
        valeur en cache, voir le docstring du module).
      - needs_llm : expressions sans entrée en cache, OU dont l'entrée en
        cache a été REJETÉE EN BLOC (au moins un distracteur en cache
        correspond désormais à une traduction connue de cette expression
        dans ce livre) — aucun filtrage partiel, l'expression repart en
        calcul LLM comme si elle n'était pas en cache.
      - audit_rows : une ligne [expression, cause] par distracteur en cache
        rejeté (voir AUDIT_CSV_HEADER), à journaliser par l'appelant."""
    from_cache: list[tuple[str, ExpressionDistractors]] = []
    needs_llm: list[VocabEntry] = []
    audit_rows: list[list[str]] = []

    for entry in entries:
        cached = cache.get((entry.type, entry.expression.casefold()))
        if cached is None:
            needs_llm.append(entry)
            continue

        known_translations = {t.casefold(): t for t in entry.translations}
        conflicts = [
            (d, known_translations[d.strip().casefold()])
            for d in cached.distractors if d.strip().casefold() in known_translations
        ]
        if conflicts:
            stats["expressions_cache_invalidees"] += 1
            for distractor, translation in conflicts:
                print(f"  ATTENTION cache invalidé pour {entry.expression!r} ({entry.type}) : le "
                      f"distracteur {distractor!r} correspond désormais à la traduction connue "
                      f"{translation!r} -> recalcul via le LLM.")
                audit_rows.append([entry.expression, distractor])
            needs_llm.append(entry)
        else:
            stats["expressions_depuis_cache"] += 1
            from_cache.append((entry.type, ExpressionDistractors(expression=entry.expression, distractors=cached.distractors)))

    return from_cache, needs_llm, audit_rows


# --------------------------------------------------------------------------
# Boucle principale : lots, réconciliation, rattrapage individuel
# --------------------------------------------------------------------------

def new_stats(expressions_total: int, expressions_skipped_done: int) -> dict[str, int]:
    return {
        "expressions_total": expressions_total, "expressions_skipped_done": expressions_skipped_done,
        "expressions_depuis_cache": 0, "expressions_cache_invalidees": 0,
        "batches": 0, "batches_failed": 0,
        "resultats": 0, "expression_mismatch": 0,
        "distracteurs_vides_retires": 0, "distracteurs_doublons_retires": 0,
        "distracteurs_traduction_retires": 0, "hors_bornes_apres_controle": 0,
        "expressions_retentees": 0, "expressions_toujours_manquantes": 0,
    }


def run_batches(
    row_type: str, batches: list[list[VocabEntry]], out_path: Path, cache_path: Path,
    audit_path: Path, stats: dict[str, int],
) -> None:
    unit_analyser = dspy.ChainOfThought(ProposeDistracteursMot if row_type == "word" else ProposeDistracteursMwe)
    lot_analyser = dspy.ChainOfThought(ProposeDistracteursLotMot if row_type == "word" else ProposeDistracteursLotMwe)
    pending_retry: list[VocabEntry] = []

    for batch_idx, batch in enumerate(batches, start=1):
        entries_by_key = {e.expression.casefold(): e for e in batch}
        expr_list = ", ".join(e.expression for e in batch)
        print(f"[{row_type} lot {batch_idx}/{len(batches)}] {len(batch)} expression(s) : {expr_list}")
        stats["batches"] += 1

        try:
            if len(batch) == 1:
                prediction = unit_analyser(entree=Expression(expression=batch[0].expression))
                results = [prediction.resultat]
            else:
                prediction = lot_analyser(lot=[Expression(expression=e.expression) for e in batch])
                results = prediction.resultats
        except Exception as exc:  # dégrade : un lot en échec n'arrête pas le run
            print(f"  échec de lot : {exc!r}")
            stats["batches_failed"] += 1
            pending_retry.extend(batch)
            continue

        seen_keys: set[str] = set()
        kept: list[ExpressionDistractors] = []
        rejected_rows: list[list[str]] = []
        for result in results:
            key = result.expression.strip().casefold()
            entry = entries_by_key.get(key)
            if entry is None:
                print(f"  ATTENTION résultat hors-lot ignoré : expression={result.expression!r}")
                continue
            seen_keys.add(key)
            stats["resultats"] += 1
            kept.append(check_distractors(result, entry, stats, rejected_rows))

        missing = [e for e in batch if e.expression.casefold() not in seen_keys]
        if missing:
            print(f"  ATTENTION {len(missing)} expression(s) absente(s) de la réponse : "
                  f"{', '.join(e.expression for e in missing)}")
            pending_retry.extend(missing)

        append_results_csv(row_type, kept, out_path)
        append_results_csv(row_type, kept, cache_path)  # résultats frais -> réutilisables par un futur livre
        if rejected_rows:
            ensure_csv_with_header(audit_path, AUDIT_CSV_HEADER)
            append_audit_rows(rejected_rows, audit_path)
        print(f"  -> {len(kept)} résultat(s) écrit(s)"
              f"{f', {len(rejected_rows)} distracteur(s) rejeté(s) -> {audit_path}' if rejected_rows else ''}")

    if not pending_retry:
        return

    print()
    print(f"=== Rattrapage individuel de {len(pending_retry)} expression(s) ({row_type}) ===")
    for entry in pending_retry:
        stats["expressions_retentees"] += 1
        print(f"[rattrapage {stats['expressions_retentees']}/{len(pending_retry)}] {entry.expression}...")
        try:
            prediction = unit_analyser(entree=Expression(expression=entry.expression))
        except Exception as exc:
            print(f"  échec : {exc!r}")
            stats["expressions_toujours_manquantes"] += 1
            continue

        result = prediction.resultat
        if result.expression.strip().casefold() != entry.expression.strip().casefold():
            print(f"  ATTENTION toujours aucun résultat exploitable pour {entry.expression!r}")
            stats["expressions_toujours_manquantes"] += 1
            continue

        stats["resultats"] += 1
        rejected_rows: list[list[str]] = []
        kept = check_distractors(result, entry, stats, rejected_rows)
        append_results_csv(row_type, [kept], out_path)
        append_results_csv(row_type, [kept], cache_path)  # résultat frais -> réutilisable par un futur livre
        if rejected_rows:
            ensure_csv_with_header(audit_path, AUDIT_CSV_HEADER)
            append_audit_rows(rejected_rows, audit_path)
        print("  -> 1 résultat écrit"
              f"{f', {len(rejected_rows)} distracteur(s) rejeté(s) -> {audit_path}' if rejected_rows else ''}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN_PATH),
                         help="Chemin du CSV d'entrée (défaut : inputs/vocabulary_input_example.csv)")
    parser.add_argument("--out-dir", dest="out_root", default=str(DEFAULT_OUT_ROOT),
                         help="Répertoire RACINE des résultats (défaut : out/) — le résultat de CE "
                              "fichier va dans <out-dir>/<slug>/ (voir slugify dans le docstring du "
                              "module), jamais un chemin libre, pour ne jamais écraser le résultat "
                              "d'un autre fichier par erreur")
    parser.add_argument("--limit", type=int, default=0,
                         help="Plafond d'expressions uniques considérées depuis l'entrée (0 = toutes, défaut)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                         help="Nombre d'expressions par lot avant appel groupé à catgpt "
                              "(défaut : 50 ; 0 = mode séquentiel, un appel par expression)")
    parser.add_argument("--restart", action="store_true",
                         help="Supprime tout <out-dir>/<slug>/ (résultat + audit/ de CE run) au lieu "
                              "de reprendre là où le run précédent s'est arrêté — n'affecte JAMAIS le "
                              "cache persistant de distracteurs (--cache-path), voir --ignore-cache")
    parser.add_argument("--no-cache", action="store_true",
                         help="Désactive le cache disque persistant de DSPy (~/.dspy_cache) : "
                              "force un appel catgpt réel pour chaque lot/expression, sans jamais "
                              "rejouer une réponse d'un run précédent (voir configure_dspy). Sans "
                              "rapport avec --cache-path/--ignore-cache (cache des DISTRACTEURS, "
                              "réutilisable entre livres) — voir le docstring du module.")
    parser.add_argument("--cache-path", dest="cache_path", default=str(DEFAULT_CACHE_PATH),
                         help="Chemin du cache PERSISTANT de distracteurs (défaut : "
                              "cache/distractors_cache.csv) — réutilisé entre livres/runs, jamais "
                              "vidé par --restart, voir CACHE PERSISTANT dans le docstring du module")
    parser.add_argument("--rejected-out", dest="rejected_path", default=None,
                         help="Chemin du journal des distracteurs rejetés par le garde-fou "
                              "anti-traduction (défaut : <out-dir>/<slug>/audit/"
                              "distractors_rejected.csv, propre à CE run, vidé par --restart) — "
                              "colonnes expression,cause (cause = le distracteur rejeté lui-même) ; "
                              "couvre à la fois les rejets sur calcul frais et sur relecture de "
                              "cache invalidée")
    parser.add_argument("--ignore-cache", action="store_true",
                         help="Ignore le cache persistant de distracteurs EN LECTURE (force un appel "
                              "LLM pour toute expression, même déjà en cache) ; le cache est quand "
                              "même réécrit avec les résultats recalculés")
    parser.add_argument("--dry-run", action="store_true",
                         help="Lit, dédoublonne, consulte le cache et affiche le plan de lots sans "
                              "appeler le LLM ni rien écrire (ni --out, ni le cache, ni l'audit)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    cache_path = Path(args.cache_path)

    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    slug = slugify(in_path.stem)
    paths = RunPaths(output_root=Path(args.out_root) / slug, slug=slug)
    out_path = paths.out
    rejected_path = Path(args.rejected_path) if args.rejected_path else paths.rejected

    if args.restart and not args.dry_run and paths.output_root.exists():
        shutil.rmtree(paths.output_root)  # résultat + audit/ de CE run — jamais cache_path (persistant)

    print(f"Entrée : {in_path}  (slug : {slug})")
    print(f"Sortie : {paths.output_root}")
    entries, read_counters = read_input_csv(in_path)
    print(f"{read_counters['lignes_lues']} ligne(s) lue(s), "
          f"{read_counters['lignes_ignorees']} ignorée(s) (type inconnu ou expression vide), "
          f"{read_counters['expressions_uniques']} expression(s) unique(s), "
          f"{read_counters['doublons_ecartes']} doublon(s) écarté(s).")

    if args.limit > 0:
        entries = entries[:args.limit]
        print(f"--limit {args.limit} -> {len(entries)} expression(s) considérée(s).")

    words = [e for e in entries if e.type == "word"]
    mwes = [e for e in entries if e.type == "mwe"]
    print(f"  dont {len(words)} word / {len(mwes)} mwe")

    done = read_done_expressions(out_path)
    todo_words = [e for e in words if e.expression.casefold() not in done]
    todo_mwes = [e for e in mwes if e.expression.casefold() not in done]
    nb_skipped = (len(words) - len(todo_words)) + (len(mwes) - len(todo_mwes))
    if done:
        print(f"{nb_skipped} expression(s) déjà présente(s) dans {out_path} -> sautée(s).")

    stats = new_stats(expressions_total=len(entries), expressions_skipped_done=nb_skipped)

    cache = {} if args.ignore_cache else read_cache(cache_path)
    print(f"Cache : {cache_path} ({len(cache)} entrée(s) connue(s)"
          f"{' — ignoré en lecture (--ignore-cache)' if args.ignore_cache else ''}).")
    from_cache, needs_llm, audit_rows = split_by_cache(todo_words + todo_mwes, cache, stats)
    print(f"{len(from_cache)} expression(s) réutilisée(s) depuis le cache, "
          f"{stats['expressions_cache_invalidees']} entrée(s) de cache invalidée(s) "
          f"(garde-fou anti-traduction, voir rejected_path), "
          f"{len(needs_llm)} expression(s) à traiter par le LLM.")

    needs_llm_words = [e for e in needs_llm if e.type == "word"]
    needs_llm_mwes = [e for e in needs_llm if e.type == "mwe"]
    word_batches = build_batches(needs_llm_words, args.batch_size)
    mwe_batches = build_batches(needs_llm_mwes, args.batch_size)
    print(f"{len(needs_llm_words)} word(s) -> {len(word_batches)} lot(s), "
          f"{len(needs_llm_mwes)} mwe(s) -> {len(mwe_batches)} lot(s) "
          f"(seuil {args.batch_size} expression(s)/lot, lots jamais mixtes).")

    if args.dry_run:
        for label, batches in (("word", word_batches), ("mwe", mwe_batches)):
            for idx, batch in enumerate(batches, start=1):
                print(f"  [dry-run {label} lot {idx}/{len(batches)}] "
                      f"{', '.join(e.expression for e in batch)}")
        print("--dry-run : aucun appel LLM effectué, rien écrit (ni --out, ni le cache, ni l'audit).")
        return 0

    if not from_cache and not needs_llm:
        print("Rien à faire : toutes les expressions demandées sont déjà présentes dans le CSV de sortie.")
        return 0

    ensure_csv_with_header(out_path)
    ensure_csv_with_header(cache_path)

    if audit_rows:
        ensure_csv_with_header(rejected_path, AUDIT_CSV_HEADER)
        append_audit_rows(audit_rows, rejected_path)
        print(f"{len(audit_rows)} conflit(s) cache/traduction journalisé(s) -> {rejected_path}")

    if from_cache:
        from_cache_by_type: dict[str, list[ExpressionDistractors]] = {"word": [], "mwe": []}
        for row_type, result in from_cache:
            from_cache_by_type[row_type].append(result)
        for row_type, results in from_cache_by_type.items():
            append_results_csv(row_type, results, out_path)

    if needs_llm:
        configure_dspy(no_cache=args.no_cache)
        if word_batches:
            run_batches("word", word_batches, out_path, cache_path, rejected_path, stats)
        if mwe_batches:
            run_batches("mwe", mwe_batches, out_path, cache_path, rejected_path, stats)

    print()
    print("=== Récapitulatif ===")
    print(f"Expressions au total          : {stats['expressions_total']}")
    print(f"  dont déjà traitées (sautées) : {stats['expressions_skipped_done']}")
    print(f"  dont réutilisées depuis le cache : {stats['expressions_depuis_cache']}")
    print(f"  dont entrées de cache invalidées (-> recalcul LLM) : {stats['expressions_cache_invalidees']}")
    print(f"Lots envoyés                   : {stats['batches']} ({stats['batches_failed']} en échec)")
    print(f"Expressions rattrapées individuellement : {stats['expressions_retentees']}")
    print(f"  dont toujours manquantes    : {stats['expressions_toujours_manquantes']}")
    print(f"Résultats produits              : {stats['resultats']}")
    print(f"  dont expression renvoyée != entrée : {stats['expression_mismatch']}")
    print(f"  dont distracteurs vides retirés    : {stats['distracteurs_vides_retires']}")
    print(f"  dont distracteurs doublons retirés : {stats['distracteurs_doublons_retires']}")
    print(f"  dont distracteurs = traduction connue (retirés) : {stats['distracteurs_traduction_retires']}")
    print(f"  dont lignes hors bornes [{MIN_DISTRACTORS},{MAX_DISTRACTORS}] après contrôle : "
          f"{stats['hors_bornes_apres_controle']}")
    print()
    print(f"-> {out_path}")
    print(f"-> cache : {cache_path}")
    if rejected_path.exists():
        print(f"-> distracteurs rejetés (audit) : {rejected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
