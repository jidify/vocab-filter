"""Catalogue de variantes de prompt par tâche LLM — Lot U4 du plan
d'unification (fix_pipeline/multi_models/report_multi_models.md §4bis).

Le champ ``custom_prompt`` d'une ``TaskDescriptor``/``TaskLlmConfig``
(``pipeline/llm_tasks.py``) porte le NOM d'une entrée de ``PROMPT_VARIANTS``
— jamais le texte du prompt directement, parce que toutes les tâches
S3/S5 ont déjà la même structure à 4 volets (system unitaire, template
unitaire, system lot, template lot) qu'une simple chaîne ne pourrait pas
couvrir. Poser ``VOCAB_LLM_<TASK_ID>=provider/modèle;prompt=<nom>``
sélectionne une variante ; chaque champ de ``PromptOverride`` est optionnel
— laisser ``None`` conserve le texte standard (en dur dans le module qui
porte la tâche) pour cette partie précise.

Contrat de placeholders : chaque site d'appel documente, dans son propre
module, le dictionnaire de champs qu'il fournit à ``render()``. Un
placeholder absent de ce dictionnaire lève ``PromptVariantError`` — jamais
un ``KeyError`` nu ni un rendu silencieusement tronqué.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


class PromptVariantError(ValueError):
    """Variante de prompt inconnue, ou placeholder manquant dans un template."""


@dataclass(frozen=True)
class PromptOverride:
    system: str | None = None
    user_template: str | None = None
    batch_system: str | None = None
    batch_template: str | None = None
    schema_variant: str = "default"


class _StrictFormatDict(dict):
    def __missing__(self, key):
        raise PromptVariantError(f"placeholder inconnu dans le template : {{{key}}}")


def render(template: str, fields: Mapping[str, object]) -> str:
    """``template.format_map`` avec un placeholder manquant transformé en
    ``PromptVariantError`` explicite plutôt qu'un ``KeyError`` nu."""
    try:
        return template.format_map(_StrictFormatDict(fields))
    except PromptVariantError:
        raise
    except (KeyError, IndexError) as exc:
        raise PromptVariantError(f"placeholder invalide dans le template : {exc}") from exc


# ============================================================
# S3-judge-occurrence — variante "tags" (catgpt) : reprend les critères de
# discrimination enrichis de fix_pipeline/evaluate_s3_judges.py (prompt
# validé sur le corpus contrastif), sortie compacte — ``evidence`` en 1 à 2
# étiquettes fermées plutôt qu'un indice en texte libre (c'est le format
# libre qui rendait ce champ coûteux chez catgpt, voir report_multi_models.md
# §4bis), pas de champ ``reason`` séparé (mwe_judge._calibrate_occurrence ne
# le compte plus dans ``complete``, pour les deux variantes — voir
# mwe_judge.py). Placeholders unitaires : canonical_form, context, surface
# (alias de idiom/sentence, voir mwe_judge._occurrence_prompt).
# ============================================================

EVIDENCE_TAGS = frozenset({
    "substitution_impossible", "sens_specialise", "polarite_inversee",
    "structure_libre", "collocation_contrainte",
})

_S3_OCC_TAGS_SYSTEM = (
    "Tu es linguiste, spécialiste de l'anglais lexicalisé pour l'enseignement "
    "à des apprenants francophones avancés. Classe CETTE occurrence précise "
    "d'une expression candidate, jamais l'expression en général."
)

_S3_OCC_TAGS_USER = """Expression candidate : "{canonical_form}"
Phrase : "{context}"
Span détecté : "{surface}"

Classe CETTE occurrence dans exactement une catégorie :
- "idiome" : sens conventionnel non compositionnel, notamment si la lecture
  littérale contredit le sens réellement communiqué ou inverse sa polarité ;
- "phrasal_verb" : verbe et particule/préposition forment une unité verbale
  lexicalisée dont le sens ou la construction est spécialisé ;
- "semi_fige" : collocation contrainte ou formulation conventionnelle, mais
  dont le sens global reste compositionnel et compatible avec les mots ;
- "littéral" : combinaison syntaxique libre et compositionnelle dans ce contexte ;
- "incertain" : le contexte ne permet pas de trancher sans forcer l'analyse.
En cas de conflit, la non-compositionnalité prime sur le caractère seulement figé.

Réponds uniquement avec ce JSON compact, sans explication ni champ supplémentaire :
{{"label":"<catégorie>","canonical_form":"<canon>","pos":"<NOUN|VERB|ADJ|ADV|OTHER>","contextual_paraphrase":"<paraphrase anglaise>","confidence":<0.0-1.0>,"evidence":["<1 à 2 étiquettes parmi : substitution_impossible, sens_specialise, polarite_inversee, structure_libre, collocation_contrainte>"],"wordnet_sense_id":"<sens WordNet exact si fourni et applicable, sinon null>"}}"""

_S3_OCC_TAGS_BATCH_SYSTEM = _S3_OCC_TAGS_SYSTEM + (
    " Tu reçois plusieurs occurrences indépendantes : ne les fusionne jamais et "
    "renvoie une décision séparée pour chaque occurrence_id."
)

_S3_OCC_TAGS_BATCH_USER = """Classe séparément les {count} occurrences suivantes.

{items}

Réponds uniquement avec un objet JSON compact, sans explication ni champ supplémentaire :
{{"decisions":[{{"occurrence_id":"<id exact>","label":"<catégorie>","canonical_form":"<canon>","pos":"<NOUN|VERB|ADJ|ADV|OTHER>","contextual_paraphrase":"<paraphrase anglaise>","confidence":<0.0-1.0>,"evidence":["<1 à 2 étiquettes>"],"wordnet_sense_id":"<sens WordNet exact ou null>"}}]}}
Il doit y avoir exactement une décision par occurrence_id, dans le même ordre."""

PROMPT_VARIANTS: Mapping[str, PromptOverride] = MappingProxyType({
    "s3-occurrence-tags": PromptOverride(
        system=_S3_OCC_TAGS_SYSTEM, user_template=_S3_OCC_TAGS_USER,
        batch_system=_S3_OCC_TAGS_BATCH_SYSTEM, batch_template=_S3_OCC_TAGS_BATCH_USER,
        schema_variant="tags",
    ),
})
