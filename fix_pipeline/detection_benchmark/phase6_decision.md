# Q0-3 — Phase 6 : décision d'architecture

## Décision

**Issue 1 du plan : `rules_plus` suffit.** Pas de piste LLM en S1. spaCy
reste conservé comme générateur de candidats (via `pipeline.multi_token`
et `pipeline.vpc`), mais **augmenté** par les générateurs `rules_plus`
(PARSEME, WordNet, patron de phrasal verb séparable, règles de bornes) —
jamais remplacé, jamais donné de pouvoir de rejet sur ces nouveaux
candidats.

Cette décision **ne modifie rien au pipeline de production** — c'est le
constat de ce document, pas sa mise en œuvre. La reprise de S1-2 pour
intégrer `rules_plus` (aujourd'hui un module de benchmark sous
`fix_pipeline/detection_benchmark/rules_plus.py`) est un chantier séparé,
à planifier ensuite, explicitement hors périmètre de ce prompt.

## Ce qui a été lu pour trancher

- `fix_pipeline/plan_detection_benchmark_funnel.md` (Phase 6 et règles
  transverses).
- `fix_pipeline/detection_benchmark/phase2_baselines_report.md` (deux
  baselines spaCy).
- `fix_pipeline/detection_benchmark/phase3_rules_plus_report.md`
  (`rules_plus`, critère d'arrêt n°1 atteint).
- Phase 4 (`phase4_llm_probe_report.md`) et Phase 5
  (`phase5_external_validation_report.md`) : **absents** —
  voir la section dédiée ci-dessous pour ce que ça signifie pour chacune,
  ce n'est pas la même situation pour les deux.

## Le chemin qui mène à la décision, phase par phase

### Phase 0-1 (corpus et scorer) — non re-questionnés ici

Corpus gold v0 gelé (99 segments, 109 spans, commit `7bf07f2`), scorer
Phase 1 validé par son propre auto-test (score du corpus gold contre
lui-même = 100% partout, 0% de capture des `hard_negative`). Ce document
n'y revient pas : il s'appuie sur les rapports Phase 2/3 qui les utilisent
déjà tous les deux.

### Phase 2 — les deux baselines spaCy

`pipeline.multi_token` seul : rappel MWE exact **7,0%**, phrasal verbs
séparables **0%**. L'ensemble réel du pipeline actuel (`multi_token` ∪
idiomatch ∪ VPC ∪ mots simples, **la vraie baseline** selon le plan) :
rappel global exact **46,3%** / chevauchement **97,6%**, rappel MWE exact
**43,7%**, phrasal verbs séparables exact **57,1%**, `protective_span`
exact **50%**, 10/27 `hard_negative` capturés. Constat clé retenu pour la
Phase 3 : l'écart entre rappel exact et rappel par chevauchement (46,3%
vs 97,6%) montre que le goulot est **les bornes**, pas la couverture — la
quasi-totalité des spans gold sont "vus" par au moins un détecteur, mal
délimités.

### Phase 3 — `rules_plus`, critère d'arrêt n°1 atteint

`rules_plus` = union(baseline 2, PARSEME + WordNet + patron de phrasal
verb séparable + règles de bornes trait d'union/possessif/ponctuation de
dialogue/frontières de proposition), spaCy sans aucun pouvoir de rejet.
**3 des 4 seuils du critère d'arrêt n°1 franchis** (un seul suffisait) :

| Seuil | Exigence | Mesuré (baseline2 → `rules_plus`) | Franchi ? |
|---|---|---:|:---:|
| Rappel MWE exact | +10 pts | 43,7% → 67,6% (**+23,9 pts**) | **OUI** |
| Phrasal verbs séparables exact | +15 pts | 57,1% → 85,7% (**+28,6 pts**) | **OUI** |
| Erreurs structurelles spaCy connues corrigées | ≥75% | **7/8 = 87,5%** | **OUI** |
| Rappel global exact, sans explosion | ≥95% | 46,3% → 70,7% | non |

*(Chiffres après une itération sur le rappel documentée dans le rapport
Phase 3 — deux affinements ciblés : un garde-fou contre une erreur de
tagging spaCy sur un token coordonné, et le rejeu du lexique custom déjà
validé en production à une fenêtre plus large que le `slop=2` d'idiomatch.
La version initiale franchissait déjà les 3 mêmes seuils, marges plus
faibles : +21,1 / +21,4 / 87,5%.)*

Effets secondaires mesurés dans le même rapport : `protective_span` exact
50%→**100%**, `boundary_accuracy` globale 47,5%→**72,5%**, sur-génération
maîtrisée (+8,3% de candidats/1000 tokens, 498,7 vs 460,7 — pas
d'explosion), `hard_negative` capturés 10/27→13/27 (+3, tous documentés
et attribués au scanner de phrasal verbs — dont un cas que le corpus gold
qualifie lui-même de "genuinely ambiguous"). Une limite assumée et
documentée : `ground-floor/basement duplex tenement apartment` (seg75)
reste non détectée, faute de tout candidat de base spaCy sur ce segment à
étendre.

Conséquence directe, appliquée dans le rapport Phase 3 lui-même : **la
Phase 4 n'a pas été lancée**, conformément à la règle du plan ("Phase 4 —
à lancer seulement si `rules_plus` reste nettement sous le gate").

### Phase 4 — absente par construction, pas par oubli

Le plan gate explicitement la Phase 4 sur l'échec de `rules_plus` à la
Phase 3 ("critère d'arrêt n°1 non atteint"). Ici il est atteint, et
largement (marges de +21 points sur deux des trois seuils francs). Aucun
LLM n'a tourné, y compris "pour voir" — respecté à la lettre dans
`phase3_run_rules_plus.py` (aucun import `ollama`/`litellm`/appel LLM).
Rien à documenter de plus : l'absence de rapport Phase 4 EST la
documentation de cette branche du plan.

### Phase 5 — absente, mais pas court-circuitée par un critère d'arrêt propre à elle

Différence importante avec la Phase 4 : la condition de déclenchement de
la Phase 5 ("Seulement si une architecture bat clairement spaCy en local
— Phase 3 ou 4") **est remplie** par le résultat de la Phase 3 ci-dessus.
La Phase 5 n'a cependant pas été exécutée dans ce prompt — validation sur
STREUSLE/PARSEME EN hors périmètre de la tâche demandée ici (lecture des
rapports existants + rédaction de la décision, explicitement pas de
nouveau run). **Ce document ne peut donc pas s'appuyer sur son critère
d'arrêt n°3** (généralisation hors du livre, indépendance des règles
propres à *The Humans*) : la décision ci-dessous porte sur l'architecture
à privilégier pour S1, pas sur une validation finale prête pour un autre
livre. À traiter dans le chantier de mise en œuvre séparé mentionné plus
haut, avant d'étendre `rules_plus` au-delà de *The Humans* — pas avant de
choisir l'architecture elle-même, ce que ce document tranche.

## Pourquoi pas les deux autres issues

- **Issue 2 (`rules_plus` + LLM)** : écartée. Le plan ne l'envisage que
  si la Phase 4 a mesuré un gain net du LLM au-dessus de `rules_plus`
  seul — la Phase 4 n'a pas eu lieu, précisément parce que `rules_plus`
  seul a déjà franchi le gate. Rouvrir cette piste maintenant reviendrait
  à ignorer le critère d'arrêt n°1 déjà appliqué en Phase 3.
- **Issue 3 (conserver spaCy imparfait, déplacer l'effort vers S3)** :
  écartée. Cette issue suppose qu'"aucune solution ne bat clairement la
  baseline" — le contraire de ce que mesure la Phase 3 (gains de +21
  points sur deux familles de spans directement citées comme priorité
  projet, MWE > NER — voir `plan_detection_benchmark_funnel.md`, section
  Contexte). Le report vers S3 (arbitrage sémantique des faux positifs)
  reste pertinent à terme — les 13/27 `hard_negative` de `rules_plus` en
  sont la preuve — mais comme travail COMPLÉMENTAIRE une fois `rules_plus`
  en place, pas comme alternative à son adoption.

## Ce que cette décision ne couvre pas

- La mise en œuvre (reprise de S1-2 pour intégrer `rules_plus` au
  pipeline de production, remplacement de la comparaison `analyze_segments`
  ad hoc de Phase 2/3 par un chemin de production propre) — chantier
  séparé, à planifier.
- La validation externe (Phase 5, STREUSLE/PARSEME EN) avant d'étendre au
  delà de *The Humans*.
- L'écart au gate du pilote (Phase 0 : rappel global ≥95%, 100% sur les
  cas critiques nommés) — `rules_plus` ne l'atteint pas encore seul
  (70,7% de rappel exact global après une première itération, voir le
  rapport Phase 3). Cet écart de RAPPEL se comble en continuant à affiner
  `rules_plus` lui-même (bornes/lexiques supplémentaires — l'itération
  déjà faite en Phase 3 en est un exemple concret) ; la précision des
  candidats ajoutés (les `hard_negative` que `rules_plus` capture, un
  problème distinct du rappel) relève elle de S3 (`pipeline/mwe_judge.py`,
  jugement occurrence par occurrence — voir
  `fix_pipeline/plan_action_fix_pipeline.md`, section 3). Aucun des deux
  chantiers n'est un signe que l'architecture choisie ici est la mauvaise.
