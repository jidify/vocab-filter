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
    REJEU plus bas), puis écrit quand même (jamais perdu silencieusement) et
    compté à part.

REJEU DES DISTRACTEURS REJETÉS (--replay-rejected) : relit
out/<slug>/audit/distractors_rejected.csv, regroupe les causes par
expression, et redemande au LLM ses distracteurs pour CES expressions
précises via une variante des signatures ci-dessus
(ProposeDistracteurs{Mot,Mwe}{,Lot}Rejeu, champ supplémentaire entree.a_eviter
= les causes connues, jamais à reproposer). Mode AUTONOME : remplace
ENTIÈREMENT le flux normal pour ce run (pas de génération de nouvelles
expressions dans le même run) ; incompatible avec --restart (qui
supprimerait exactement ce que ce mode doit lire — voir main()). Traitement
PAR LOTS comme le flux normal (mêmes --batch-size, mêmes lots homogènes
word/mwe). Le garde-fou anti-traduction (check_distractors) reste actif sur
les résultats du rejeu : le LLM peut ignorer a_eviter, un distracteur qui
coïncide encore avec une traduction connue est de nouveau écarté et
journalisé.

Résultats du rejeu : une expression qui obtient >= MIN_DISTRACTORS
distracteurs valides REMPLACE sa ligne (déficiente) dans
<slug>-distractors.csv — SEULE opération du script qui réécrit ce fichier au
lieu d'y ajouter (update_output_rows : lit tout le CSV en mémoire, remplace,
réécrit ; acceptable pour cette correction ponctuelle, un livre entier restant
de l'ordre de quelques milliers de lignes) — et disparaît de
distractors_rejected.csv. Une expression encore sous MIN_DISTRACTORS y reste,
avec les causes constatées PENDANT ce rejeu (qui REMPLACENT les anciennes) —
sauf si l'appel LLM lui-même a échoué (lot en échec, expression absente de la
réponse même après rattrapage individuel), auquel cas ses causes D'ORIGINE
sont conservées telles quelles : aucune perte d'information faute d'avoir pu
retester (voir run_replay_batches). Écriture UNE SEULE FOIS en fin de rejeu
(pas de streaming incrémental ici, à la différence du flux normal — les
volumes attendus sont petits) : une interruption en cours de rejeu ne perd
rien d'irréversible, ni out_path ni distractors_rejected.csv ne sont modifiés
tant que le rejeu n'est pas allé à son terme ; relancer --replay-rejected
repart du même journal.

LIMITE CONNUE (assumée pour rester simple) : si un rejeu renvoie moins de
MIN_DISTRACTORS sans qu'aucun distracteur n'ait été rejeté cette fois (le LLM
en propose simplement trop peu, sans en reproposer d'invalide), l'expression
sort quand même de distractors_rejected.csv faute de cause à y écrire, malgré
un problème persistant — signalé au récapitulatif, pas traité autrement.

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
    # rejouer les distracteurs rejetés (voir REJEU ci-dessus) :
    uv run python POC/distractor/generate_distractors.py --in <fusionné.csv> --replay-rejected
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

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
# Signatures DSPy — REJEU (--replay-rejected, voir REJEU dans le docstring
# du module) : mêmes contraintes que les 4 signatures ci-dessus, plus une
# liste de distracteurs déjà rejetés à ne jamais reproposer.
# --------------------------------------------------------------------------

class ExpressionAvecExclusions(BaseModel):
    """Une expression à rejouer, avec les distracteurs déjà proposés pour
    elle et rejetés — ce que le LLM reçoit en entrée pour --replay-rejected."""

    expression: str = Field(description="Mot ou expression anglaise source, identique à un rejeu précédent.")
    a_eviter: list[str] = Field(
        description="Distracteurs déjà proposés pour cette expression et rejetés par un contrôle "
                    "automatique car ils correspondaient à une traduction connue — NE JAMAIS les "
                    "reproposer, y compris reformulés, au singulier/pluriel, conjugués ou en "
                    "synonyme évident de ces mots précis."
    )


class ProposeDistracteursMotRejeu(dspy.Signature):
    """Identique à ProposeDistracteursMot (même objectif, mêmes contraintes
    strictes rappelées ci-dessous), avec UNE contrainte supplémentaire :
    entree.a_eviter liste des distracteurs déjà proposés pour ce mot et
    rejetés par un contrôle automatique car ils correspondaient à une
    traduction connue du mot — NE JAMAIS reproposer un de ces mots précis,
    ni une variante évidente (accord, conjugaison, synonyme immédiat).

    Contraintes strictes (rappel, identiques à un premier essai) :
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
        source. Évite les mots trop rares, archaïques, techniques ou
        manifestement absurdes.
      - Si tu hésites sur un distracteur (y compris sur son lien avec un
        élément de a_eviter), ne le propose pas et cherche une alternative
        plus sûre.

    Classe tes propositions par qualité décroissante dans distractors. Ne
    renvoie que les 2 ou 3 meilleurs distracteurs — AUCUN ne doit figurer
    dans a_eviter, ni en être une variante évidente."""

    entree: ExpressionAvecExclusions = dspy.InputField()
    resultat: ExpressionDistractors = dspy.OutputField()


class ProposeDistracteursLotMotRejeu(dspy.Signature):
    """Même tâche que ProposeDistracteursMotRejeu, appliquée à PLUSIEURS mots
    anglais INDÉPENDANTS les uns des autres, chacun avec sa PROPRE liste
    a_eviter — ne jamais appliquer la liste a_eviter d'un mot à un autre mot
    du lot, et ne jamais mélanger les distracteurs de deux mots différents.
    Renvoie UNE entrée par mot reçu, dans une seule liste plate — le champ
    expression de chaque entrée sert à la rattacher au bon mot d'entrée."""

    lot: list[ExpressionAvecExclusions] = dspy.InputField(
        description="Mots indépendants à rejouer dans ce lot, chacun avec sa propre liste a_eviter."
    )
    resultats: list[ExpressionDistractors] = dspy.OutputField(
        description="Une entrée par mot du lot, jamais moins d'entrées que de mots reçus — aucun "
                    "distracteur ne doit figurer dans le a_eviter de SON mot."
    )


class ProposeDistracteursMweRejeu(dspy.Signature):
    """Identique à ProposeDistracteursMwe (même objectif, mêmes contraintes
    strictes rappelées ci-dessous, sens LITTÉRAL et IDIOMATIQUE/FIGURÉ),
    avec UNE contrainte supplémentaire : entree.a_eviter liste des
    distracteurs déjà proposés pour cette expression et rejetés par un
    contrôle automatique car ils correspondaient à une traduction connue —
    NE JAMAIS reproposer un de ces mots/expressions précis, ni une variante
    évidente.

    Contraintes strictes (rappel, identiques à un premier essai) :
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
        traduction.
      - Privilégie des distracteurs de fréquence et de difficulté similaires
        à l'expression source. Évite les distracteurs trop rares, archaïques,
        techniques ou manifestement absurdes.
      - Si tu hésites sur un distracteur (y compris sur son lien avec un
        élément de a_eviter), ne le propose pas et cherche une alternative
        plus sûre.

    Classe tes propositions par qualité décroissante dans distractors. Ne
    renvoie que les 2 ou 3 meilleurs distracteurs — AUCUN ne doit figurer
    dans a_eviter, ni en être une variante évidente."""

    entree: ExpressionAvecExclusions = dspy.InputField()
    resultat: ExpressionDistractors = dspy.OutputField()


class ProposeDistracteursLotMweRejeu(dspy.Signature):
    """Même tâche que ProposeDistracteursMweRejeu, appliquée à PLUSIEURS
    expressions anglaises INDÉPENDANTES les unes des autres, chacune avec sa
    PROPRE liste a_eviter — ne jamais appliquer la liste a_eviter d'une
    expression à une autre expression du lot, et ne jamais mélanger les
    distracteurs de deux expressions différentes. Renvoie UNE entrée par
    expression reçue, dans une seule liste plate — le champ expression de
    chaque entrée sert à la rattacher à la bonne expression d'entrée."""

    lot: list[ExpressionAvecExclusions] = dspy.InputField(
        description="Expressions indépendantes à rejouer dans ce lot, chacune avec sa propre liste a_eviter."
    )
    resultats: list[ExpressionDistractors] = dspy.OutputField(
        description="Une entrée par expression du lot, jamais moins d'entrées que d'expressions reçues "
                    "— aucun distracteur ne doit figurer dans le a_eviter de SON expression."
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


@dataclass
class ReplayEntry:
    """Une expression à rejouer (--replay-rejected, voir REJEU dans le
    docstring du module) : son VocabEntry d'origine (relu depuis --in, donc
    avec ses translations pour le garde-fou) et les causes déjà connues
    (distracteurs déjà proposés et rejetés pour elle)."""

    entry: VocabEntry
    excluded: list[str]


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


T = TypeVar("T")


def build_batches(entries: list[T], batch_size: int) -> list[list[T]]:
    """Découpe entries (déjà homogène en type — voir main()) en lots de
    batch_size expressions. batch_size <= 0 -> un lot par expression (mode
    séquentiel). Générique : réutilisé tel quel sur list[VocabEntry] (flux
    normal) et list[ReplayEntry] (--replay-rejected, voir plus bas)."""
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


def read_rejected(path: Path) -> dict[str, list[str]]:
    """Lit out/<slug>/audit/distractors_rejected.csv (colonnes
    expression,cause) et regroupe les causes par expression — utilisé
    UNIQUEMENT par --replay-rejected (voir REJEU dans le docstring du
    module). Fichier absent ou vide -> {}."""
    causes: dict[str, list[str]] = {}
    if not path.exists():
        return causes
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expression = (row.get("expression") or "").strip()
            cause = (row.get("cause") or "").strip()
            if expression and cause:
                causes.setdefault(expression, []).append(cause)
    return causes


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


def update_output_rows(out_path: Path, updates: dict[tuple[str, str], ExpressionDistractors]) -> int:
    """Réécrit out_path en remplaçant les lignes dont (type,
    expression.casefold()) est dans updates par leur nouveau contenu — SEULE
    fonction du script qui réécrit out_path au lieu d'y ajouter, utilisée
    UNIQUEMENT par --replay-rejected (voir REJEU dans le docstring du module ;
    le flux normal reste append-only en streaming, voir append_results_csv).
    Lit tout le fichier en mémoire (quelques centaines à quelques milliers de
    lignes pour un livre entier) — acceptable pour cette correction
    ponctuelle. Renvoie le nombre de lignes effectivement remplacées."""
    if not updates or not out_path.exists():
        return 0
    with out_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    nb_replaced = 0
    for row in rows:
        key = ((row.get("type") or "").strip(), (row.get("expression") or "").strip().casefold())
        result = updates.get(key)
        if result is not None:
            row["distractors"] = " | ".join(result.distractors)
            row["nb_distractors"] = str(len(result.distractors))
            nb_replaced += 1

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return nb_replaced


def rewrite_rejected_csv(path: Path, remaining_causes: dict[str, list[str]]) -> None:
    """Réécrit intégralement out/<slug>/audit/distractors_rejected.csv après
    un rejeu (--replay-rejected) : ne garde que les causes des expressions
    ENCORE déficientes après ce rejeu (résolues -> disparaissent du fichier).
    remaining_causes vide -> le fichier est supprimé (rien à signaler)."""
    rows = [[expression, cause] for expression, causes in remaining_causes.items() for cause in causes]
    if not rows:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(AUDIT_CSV_HEADER)
        writer.writerows(rows)


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
# Rejeu des distracteurs rejetés (--replay-rejected) — voir REJEU dans le
# docstring du module.
# --------------------------------------------------------------------------

def new_replay_stats(expressions_total: int, expressions_introuvables: int) -> dict[str, int]:
    return {
        "expressions_total": expressions_total, "expressions_introuvables": expressions_introuvables,
        "replay_batches": 0, "replay_batches_failed": 0,
        "replay_resultats": 0, "replay_corrigees": 0,
        "replay_retentees": 0, "replay_toujours_manquantes": 0,
        # Alimentés par check_distractors (réutilisé tel quel, voir plus bas).
        "expression_mismatch": 0, "distracteurs_vides_retires": 0, "distracteurs_doublons_retires": 0,
        "distracteurs_traduction_retires": 0, "hors_bornes_apres_controle": 0,
    }


def run_replay_batches(
    row_type: str, batches: list[list[ReplayEntry]], cache_path: Path, stats: dict[str, int],
) -> tuple[dict[tuple[str, str], ExpressionDistractors], dict[str, list[str]]]:
    """Pendant de run_batches pour --replay-rejected : mêmes principes (échec
    de lot -> pending_retry, réconciliation par expression.casefold(),
    rattrapage individuel en fin de run via la signature unitaire *Rejeu),
    mais construit ExpressionAvecExclusions (a_eviter = r.excluded) au lieu
    de Expression, alimente le cache normalement (append_results_csv, additif
    — voir CACHE PERSISTANT dans le docstring du module), et NE MODIFIE PAS
    out_path/distractors_rejected.csv directement : renvoie (updates,
    remaining_causes) que l'appelant applique UNE SEULE FOIS pour tout le
    run (word + mwe confondus) via update_output_rows/rewrite_rejected_csv."""
    unit_analyser = dspy.ChainOfThought(ProposeDistracteursMotRejeu if row_type == "word" else ProposeDistracteursMweRejeu)
    lot_analyser = dspy.ChainOfThought(ProposeDistracteursLotMotRejeu if row_type == "word" else ProposeDistracteursLotMweRejeu)
    pending_retry: list[ReplayEntry] = []

    updates: dict[tuple[str, str], ExpressionDistractors] = {}
    # Pessimiste par défaut : toute expression garde ses causes D'ORIGINE
    # tant qu'un rejeu effectif (réussi ou non) ne les a pas mises à jour —
    # une expression dont l'appel LLM échoue même après rattrapage individuel
    # n'est donc JAMAIS perdue du journal (voir le docstring du module).
    remaining_causes: dict[str, list[str]] = {
        r.entry.expression: list(r.excluded) for batch in batches for r in batch
    }

    for batch_idx, batch in enumerate(batches, start=1):
        entries_by_key = {r.entry.expression.casefold(): r for r in batch}
        expr_list = ", ".join(r.entry.expression for r in batch)
        print(f"[{row_type} rejeu {batch_idx}/{len(batches)}] {len(batch)} expression(s) : {expr_list}")
        stats["replay_batches"] += 1

        try:
            if len(batch) == 1:
                r0 = batch[0]
                prediction = unit_analyser(
                    entree=ExpressionAvecExclusions(expression=r0.entry.expression, a_eviter=r0.excluded)
                )
                results = [prediction.resultat]
            else:
                prediction = lot_analyser(lot=[
                    ExpressionAvecExclusions(expression=r.entry.expression, a_eviter=r.excluded) for r in batch
                ])
                results = prediction.resultats
        except Exception as exc:  # dégrade : un lot en échec n'arrête pas le run
            print(f"  échec de lot : {exc!r}")
            stats["replay_batches_failed"] += 1
            pending_retry.extend(batch)
            continue

        seen_keys: set[str] = set()
        for result in results:
            key = result.expression.strip().casefold()
            r = entries_by_key.get(key)
            if r is None:
                print(f"  ATTENTION résultat hors-lot ignoré : expression={result.expression!r}")
                continue
            seen_keys.add(key)
            stats["replay_resultats"] += 1
            rejected_rows: list[list[str]] = []
            kept = check_distractors(result, r.entry, stats, rejected_rows)
            updates[(row_type, r.entry.expression.casefold())] = kept
            if len(kept.distractors) >= MIN_DISTRACTORS:
                remaining_causes.pop(r.entry.expression, None)
                stats["replay_corrigees"] += 1
            else:
                remaining_causes[r.entry.expression] = [row[1] for row in rejected_rows]
            append_results_csv(row_type, [kept], cache_path)  # cache toujours additif, voir read_cache

        missing = [r for r in batch if r.entry.expression.casefold() not in seen_keys]
        if missing:
            print(f"  ATTENTION {len(missing)} expression(s) absente(s) de la réponse : "
                  f"{', '.join(r.entry.expression for r in missing)}")
            pending_retry.extend(missing)

    if not pending_retry:
        return updates, remaining_causes

    print()
    print(f"=== Rattrapage individuel de {len(pending_retry)} expression(s) rejouée(s) ({row_type}) ===")
    for r in pending_retry:
        stats["replay_retentees"] += 1
        print(f"[rattrapage rejeu {stats['replay_retentees']}/{len(pending_retry)}] {r.entry.expression}...")
        try:
            prediction = unit_analyser(
                entree=ExpressionAvecExclusions(expression=r.entry.expression, a_eviter=r.excluded)
            )
        except Exception as exc:
            print(f"  échec : {exc!r}")
            stats["replay_toujours_manquantes"] += 1
            continue  # remaining_causes garde les causes D'ORIGINE de r, jamais modifiées ci-dessus

        result = prediction.resultat
        if result.expression.strip().casefold() != r.entry.expression.strip().casefold():
            print(f"  ATTENTION toujours aucun résultat exploitable pour {r.entry.expression!r}")
            stats["replay_toujours_manquantes"] += 1
            continue

        stats["replay_resultats"] += 1
        rejected_rows: list[list[str]] = []
        kept = check_distractors(result, r.entry, stats, rejected_rows)
        updates[(row_type, r.entry.expression.casefold())] = kept
        if len(kept.distractors) >= MIN_DISTRACTORS:
            remaining_causes.pop(r.entry.expression, None)
            stats["replay_corrigees"] += 1
        else:
            remaining_causes[r.entry.expression] = [row[1] for row in rejected_rows]
        append_results_csv(row_type, [kept], cache_path)
        print("  -> 1 résultat rejoué")

    return updates, remaining_causes


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
    parser.add_argument("--replay-rejected", action="store_true",
                         help="Mode REJEU (voir REJEU dans le docstring du module) : relit "
                              "--rejected-out, regroupe les causes par expression, et redemande au "
                              "LLM ses distracteurs pour CES expressions en lui interdisant "
                              "explicitement de reproposer les causes connues. REMPLACE le flux "
                              "normal (pas de génération de nouvelles expressions dans ce run) ; "
                              "incompatible avec --restart")
    return parser.parse_args()


def run_replay_mode(
    entries: list[VocabEntry], rejected_path: Path, out_path: Path, cache_path: Path, args: argparse.Namespace,
) -> int:
    """Flux complet de --replay-rejected — voir REJEU dans le docstring du
    module. Appelé par main() en remplacement ENTIER du flux normal (aucune
    nouvelle expression générée dans ce run)."""
    causes_by_expr = read_rejected(rejected_path)
    if not causes_by_expr:
        print(f"Rien à rejouer : {rejected_path} absent ou vide.")
        return 0

    # entries_by_expr : si une même expression existe à la fois en word et en
    # mwe dans --in (cas structurellement possible mais non observé en
    # pratique), seule la dernière rencontrée est retenue ici — limite
    # assumée pour rester simple (voir le docstring du module).
    entries_by_expr = {e.expression.casefold(): e for e in entries}
    replay_entries: list[ReplayEntry] = []
    introuvables: list[str] = []
    for expression, causes in causes_by_expr.items():
        entry = entries_by_expr.get(expression.casefold())
        if entry is None:
            introuvables.append(expression)
            continue
        replay_entries.append(ReplayEntry(entry=entry, excluded=causes))

    print(f"{len(causes_by_expr)} expression(s) à rejouer trouvée(s) dans {rejected_path}.")
    if introuvables:
        print(f"  ATTENTION {len(introuvables)} expression(s) introuvable(s) dans --in (ignorée(s)) : "
              f"{', '.join(introuvables)}")

    replay_words = [r for r in replay_entries if r.entry.type == "word"]
    replay_mwes = [r for r in replay_entries if r.entry.type == "mwe"]
    print(f"  dont {len(replay_words)} word / {len(replay_mwes)} mwe à rejouer")

    word_batches = build_batches(replay_words, args.batch_size)
    mwe_batches = build_batches(replay_mwes, args.batch_size)
    print(f"{len(replay_words)} word(s) -> {len(word_batches)} lot(s), "
          f"{len(replay_mwes)} mwe(s) -> {len(mwe_batches)} lot(s) "
          f"(seuil {args.batch_size} expression(s)/lot, lots jamais mixtes).")

    if args.dry_run:
        for label, batches in (("word", word_batches), ("mwe", mwe_batches)):
            for idx, batch in enumerate(batches, start=1):
                detail = ", ".join(f"{r.entry.expression} (éviter : {', '.join(r.excluded)})" for r in batch)
                print(f"  [dry-run rejeu {label} lot {idx}/{len(batches)}] {detail}")
        print("--dry-run : aucun appel LLM effectué, rien écrit.")
        return 0

    if not replay_entries:
        return 0

    stats = new_replay_stats(expressions_total=len(replay_entries), expressions_introuvables=len(introuvables))

    configure_dspy(no_cache=args.no_cache)
    ensure_csv_with_header(cache_path)

    updates: dict[tuple[str, str], ExpressionDistractors] = {}
    remaining_causes: dict[str, list[str]] = {}
    if word_batches:
        u, rc = run_replay_batches("word", word_batches, cache_path, stats)
        updates.update(u)
        remaining_causes.update(rc)
    if mwe_batches:
        u, rc = run_replay_batches("mwe", mwe_batches, cache_path, stats)
        updates.update(u)
        remaining_causes.update(rc)

    nb_replaced = update_output_rows(out_path, updates)
    if nb_replaced < len(updates):
        print(f"  ATTENTION {len(updates) - nb_replaced} résultat(s) rejoué(s) sans ligne "
              f"correspondante dans {out_path} (pas remplacé(s)).")
    rewrite_rejected_csv(rejected_path, remaining_causes)

    print()
    print("=== Récapitulatif du rejeu ===")
    print(f"Expressions à rejouer           : {stats['expressions_total']}")
    print(f"  dont introuvables dans --in   : {stats['expressions_introuvables']}")
    print(f"Lots envoyés                    : {stats['replay_batches']} ({stats['replay_batches_failed']} en échec)")
    print(f"Rattrapages individuels         : {stats['replay_retentees']}")
    print(f"  dont toujours manquants       : {stats['replay_toujours_manquantes']}")
    print(f"Résultats obtenus                : {stats['replay_resultats']}")
    print(f"  dont corrigés (>= {MIN_DISTRACTORS} distracteurs valides) : {stats['replay_corrigees']}")
    print(f"  dont distracteurs = traduction connue (retirés) : {stats['distracteurs_traduction_retires']}")
    print(f"Lignes remplacées dans {out_path.name}    : {nb_replaced}")
    print(f"Expressions encore dans {rejected_path.name} : {len(remaining_causes)}")
    print()
    print(f"-> {out_path}")
    print(f"-> cache : {cache_path}")
    if rejected_path.exists():
        print(f"-> distracteurs encore rejetés (audit) : {rejected_path}")
    else:
        print(f"-> {rejected_path.name} supprimé : plus aucune expression rejetée.")
    return 0


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    cache_path = Path(args.cache_path)

    if not in_path.exists():
        print(f"CSV d'entrée introuvable : {in_path}")
        return 1

    if args.replay_rejected and args.restart:
        print("--replay-rejected et --restart sont incompatibles : --restart supprimerait "
              "out/<slug>/ (résultat ET distractors_rejected.csv), exactement ce que "
              "--replay-rejected doit lire. Lance les deux séparément si besoin.")
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

    if args.replay_rejected:
        return run_replay_mode(entries, rejected_path, out_path, cache_path, args)

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
