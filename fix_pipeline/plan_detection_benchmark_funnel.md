# Q0-3 — Benchmark des architectures de détection (plan entonnoir)

## Contexte

Constat du benchmark spaCy précédent (`pipeline_out/spacy_quick_compare/report.md`) : aucun des modèles `sm`/`lg`/`trf` ne suffit seul, et leurs pires erreurs (troncature sur trait d'union, span cassé par la ponctuation de dialogue) sont des limites structurelles de l'analyse en dépendances, pas des limites de taille de modèle. Un « autre expert » a proposé un plan complet de comparaison d'architectures (spaCy / lexiques+patrons / LLM local / ensemble) avant de toucher à `pipeline.multi_token`. Ce document reprend ce plan, recadré en **entonnoir** : comparaison locale très rapide d'abord, approfondissement (LLM, corpus externes) seulement si une alternative bat clairement spaCy.

**Priorité projet (à respecter à toutes les phases)** : la qualité **MWE** (composés, phrasal verbs, idiomes) compte beaucoup plus que la qualité **NER**. Des erreurs NER sont acceptables ; des erreurs MWE beaucoup moins. Ne pas investir de temps sur la précision/rappel NER au-delà de ce qui est gratuit.

## Machine à utiliser pour les tests

La machine à utiliser pour tester les modèles est la machine `ip=192.168.1.28` avec l'utilisateur `user=danta` CAR c'est **cette machine qui possede le GPU RTX 5090**


## Corpus gold local — déjà construit, ne pas agrandir

`fix_pipeline/gold_corpus/the_humans_gold_v0.jsonl` (commit `7bf07f2`) : **99 segments, 109 spans (82 positifs / 27 `hard_negative`)**, offsets vérifiés mécaniquement contre `pipeline.corpus.load_segments()`. Répartition des positifs : `phrasal_verb_separable` 28, `idiom` 17, `phrasal_verb_inseparable` 14, `nominal_compound` 12, `simple_word` 7, `multi_token_entity` 4 — déjà penché MWE, cohérent avec la priorité ci-dessus. Détails, taxonomie et limites : `fix_pipeline/gold_corpus/README.md`.

Jugé **suffisant pour un pilote**. Ne pas l'agrandir maintenant — l'ajout de corpus externes (STREUSLE, PARSEME EN) n'intervient qu'en Phase 5, et seulement en validation légère, pas en remplacement.

L'évaluation doit séparer **3 fonctions différentes**, sans quoi la comparaison est injuste :

- `simple_word` : détection d'un mot pédagogiquement intéressant ;
- MWE/composés/idiomes : extraction d'une unité lexicale ;
- `multi_token_entity` : surtout protection contre l'export de fragments (ex. « York » seul).

Comparer uniquement `pipeline.multi_token` aux 82 positifs serait injuste : ce composant n'est pas censé détecter les mots simples et ne couvre actuellement qu'une partie des MWE.

## Modules de production pertinents (repérés, pour gagner du temps en Phase 2)

- `pipeline/multi_token.py` — détecteur composés/entités actuel, schéma de candidats propre, écrit `multi_token_candidates.jsonl` (`config.MULTI_TOKEN_CANDIDATES_PATH`).
- `pipeline/vpc/` (`adapter.py`, `service.py`, `detectors/phrasal_verbs.py`, `domain/`) — détecteur VPC (verb-particle constructions), écrit **séparément** `vpc_candidates.jsonl` (`config.VPC_CANDIDATES_PATH`) avec un schéma différent (champs `*_char_span`).
- `pipeline/mwe.py`, `mwe_alignment.py`, `mwe_judge.py`, `mwe_stores.py`, `custom_lexicon.py` — lexiques/idiomatch/MWE ; schéma non confirmé unifié avec les deux précédents.
- `pipeline/analyze.py::analyze_segments()` orchestre ces sinks séparément — **il n'existe pas aujourd'hui de sortie unique normalisée combinant tous les détecteurs actuels.** Phase 2 doit donc prévoir un petit adaptateur de normalisation (spans → `(start_char, end_char, surface)`), pas une commande unique.

## Phase 0 — Verrouiller le benchmark local

Durée visée : quelques heures.

- Relire les 109 annotations existantes.
- Ajouter à chaque span positif un rôle explicite, en plus de la catégorie linguistique déjà présente :
  - `lexical_candidate` (unité à extraire pour le vocabulaire) ;
  - `protective_span` (span dont le rôle est surtout d'empêcher un export de fragment erroné, ex. entités) ;
  - `pedagogical_word` (mot simple à signaler).
- Conserver les catégories linguistiques actuelles (`simple_word`, `nominal_compound`, `multi_token_entity`, `phrasal_verb_separable`, `phrasal_verb_inseparable`, `idiom`, `hard_negative`) — le rôle est un champ en plus, pas un remplacement.
- Distinguer explicitement les spans imbriqués acceptables (ex. `duplex tenement apartment` inclus dans `ground-floor/basement duplex tenement apartment`, idx 75 du corpus) : les deux sont gold, à des granularités différentes — ce n'est pas une erreur de l'un ou l'autre, à noter clairement dans le span le plus court.
- Geler ce corpus en v0 (plus aucune modification de contenu après cette phase) avant d'exécuter la moindre alternative.

**Gate retenu pour ce pilote — pas 99% :**

- rappel global ≥ 95% ;
- 100% sur les cas critiques nommés (les six expressions historiques du benchmark spaCy + les segments `known_difficult` portés depuis ce benchmark) ;
- aucun effondrement d'une catégorie (à documenter au cas par cas en Phase 1/2, pas de seuil chiffré unique) ;
- résultat toujours détaillé par rôle **et** par catégorie, jamais un seul chiffre global.

Justification du 95% : avec seulement 82 positifs, un seul cas manqué donne déjà 98,78% de rappel — un gate à 99% équivaut presque à exiger zéro erreur sur un corpus de cette taille. 99% reste l'objectif final, sur un corpus plus grand (voir Phase 5).

## Phase 1 — Construire un évaluateur de spans minimal

Durée visée : une journée maximum.

Format d'entrée commun pour chaque détecteur (un objet JSON par span produit) :

```json
{
  "segment_idx": 75,
  "surface": "New York City",
  "start_char": 73,
  "end_char": 86,
  "category": "multi_token_entity",
  "source": "spacy"
}
```

Pour les phrasal verbs séparables, le format doit pouvoir représenter des membres discontinus : `full_span` peut couvrir toute la construction, mais `member_spans` doit identifier séparément le verbe et la particule.

Métriques à produire :

- rappel exact des spans ;
- rappel avec chevauchement (**diagnostic uniquement, jamais assimilé à une réussite** : `floor apartment` chevauche `ground-floor apartment` mais représente précisément l'erreur qu'on veut détecter, pas un succès partiel) ;
- exactitude des bornes ;
- rappel par catégorie ;
- rappel par rôle ;
- taux de capture des 27 `hard_negative` ;
- nombre de candidats produits par 1000 tokens (proxy de sur-génération) ;
- temps d'exécution.

**Métrique volontairement absente — ne pas l'ajouter :** pas de « précision exacte » globale (candidats hors gold comptés comme faux positifs). Le corpus n'est pas exhaustif (109 spans annotés sur 2535 segments) : un span produit par un détecteur mais absent du gold n'est pas nécessairement une erreur, il peut s'agir d'une unité réelle non couverte par l'échantillonnage. Le seul signal de précision interprétable ici est le taux de capture des `hard_negative`, conçus explicitement comme pièges.

Sortie : le scorer lui-même dans `fix_pipeline/detection_benchmark/scorer.py` (tracké, réutilisable). Test de cohérence minimal : scorer le gold corpus contre lui-même doit donner 100% partout.

## Phase 2 — Établir deux baselines spaCy

Durée visée : quelques heures + 1-2h pour l'adaptateur de normalisation (voir section Modules ci-dessus).

Mesurer séparément :

1. `pipeline.multi_token` seul ;
2. l'ensemble des détecteurs actuels du pipeline (spaCy NER/compound, détecteur VPC, idiomatch et lexiques existants, extraction des mots simples) — **la vraie baseline**. Le remplacement doit battre le pipeline actuel, pas seulement `multi_token`.

Le rapport doit donner au minimum :

- rappel global ;
- rappel MWE ;
- rappel des phrasal verbs séparables ;
- rappel des `protective_span` ;
- erreurs de bornes ;
- faux positifs sur les 27 pièges (`hard_negative`).

Sortie : rapport dans `fix_pipeline/detection_benchmark/phase2_baselines_report.md` ; artefacts de run bruts (candidats produits) dans `pipeline_out/detection_benchmark/` (regénérables, non trackés).

## Phase 3 — Essayer d'abord l'alternative la plus simple (`rules_plus`)

Durée maximale : un à deux jours.

Construire une variante `rules_plus` sans LLM :

- lexiques MWE existants ;
- WordNet pour les formes multi-mots, comme source de candidats seulement (jamais de pouvoir de rejet) ;
- lexique PARSEME ou ressources compatibles ;
- patrons de phrasal verbs séparables ;
- règles de bornes pour : traits d'union, possessifs, ponctuation de dialogue, frontières de propositions ;
- union avec les sorties spaCy, sans donner à spaCy de pouvoir de rejet.

Objectif : voir si des ajouts déterministes corrigent l'essentiel des trous à faible coût conceptuel — pas encore la précision finale.

### Critère d'arrêt n°1

Ne pas poursuivre `rules_plus` comme remplaçant s'il n'apporte **aucun** des gains suivants :

- +10 points de rappel absolu sur les MWE ;
- +15 points sur les phrasal verbs séparables ;
- correction d'au moins 75% des erreurs structurelles spaCy connues ;
- ou rappel global ≥ 95% sans explosion du nombre de candidats.

Documenter la décision (continuer vers Phase 4 / s'arrêter ici) avec les chiffres à l'appui dans `fix_pipeline/detection_benchmark/phase3_rules_plus_report.md`.

## Phase 4 — Probe LLM local, strictement isolé

À lancer **seulement si** `rules_plus` reste nettement sous le gate (critère d'arrêt n°1 non atteint). Durée maximale initiale : une journée, sans intégration au pipeline.

- Un seul modèle local raisonnable sur la RTX 5090 (ne pas comparer plusieurs familles à ce stade).
- Lui demander uniquement de proposer des spans structurés, sans traduction ni décision pédagogique.
- Tester trois configurations : LLM seul ; `rules_plus` seul ; union `rules_plus` + LLM.
- Le LLM ne doit pas inventer d'offsets : il retourne la surface et éventuellement les tokens membres ; le runner réaligne mécaniquement sur le texte. Une proposition non alignable est une erreur, pas un candidat à corriger silencieusement.

### Critère d'arrêt n°2

Abandonner la piste LLM si, sur le corpus local :

- elle gagne moins de 5 points de rappel MWE sur `rules_plus` ;
- elle ne corrige pas substantiellement les phrasal verbs séparables ;
- elle multiplie fortement les faux positifs ;
- ou son gain vient seulement de cas déjà facilement couverts par les lexiques.

Documenter dans `fix_pipeline/detection_benchmark/phase4_llm_probe_report.md`.

## Phase 5 — Validation externe légère

**Seulement si** une architecture bat clairement spaCy en local (Phase 3 ou 4) — pas avant, pas systématique.

- Retenir seulement **STREUSLE** (segmentation lexicale générale, MWE nominales/verbales, supersenses) et **PARSEME EN** (MWE verbales, en particulier constructions discontinues).
- Différer **DiMSUM** (largement redondant avec STREUSLE pour ce pilote) et **MWE-CWI / CompLex** (difficulté pédagogique — plus utiles pour S7 que pour S1, cohérent avec la priorité détection > jugement pédagogique à ce stade).
- Ne pas importer intégralement ces corpus : un adaptateur par format + un sous-ensemble gold fixe suffisent pour vérifier l'absence de sur-apprentissage sur *The Humans*.

### Critère d'arrêt n°3

Une architecture ne peut remplacer le chemin actuel que si :

- elle bat spaCy sur le corpus local ;
- le gain subsiste sur STREUSLE ou PARSEME ;
- elle ne dépend pas de règles propres au livre ;
- ses sorties restent auditables et alignées.

Documenter dans `fix_pipeline/detection_benchmark/phase5_external_validation_report.md`.

## Phase 6 — Décision d'architecture

Lire tous les rapports de phase précédents. Trois issues possibles seulement :

1. **`rules_plus` suffit** → solution la plus simple retenue, pas de LLM en S1.
2. **`rules_plus` + LLM apporte un gain net** → LLM utilisé comme générateur complémentaire, jamais comme source exclusive.
3. **aucune solution ne bat clairement la baseline** → conserver spaCy comme générateur imparfait, déplacer l'effort vers le jugement contextuel S3 (arbitrage des faux positifs et hypothèses concurrentes).

Sortie : `fix_pipeline/detection_benchmark/phase6_decision.md`. Cette phase **ne modifie pas** le pipeline de production — la mise en œuvre du choix (reprise de S1-2) est un chantier séparé, hors périmètre de ce document.

## Règles transverses (à respecter à toutes les phases)

- Ne jamais modifier le pipeline de production tant qu'une phase n'a pas explicitement validé un remplacement (Phase 6 uniquement, et encore : Phase 6 documente la décision, ne l'implémente pas).
- Ne jamais faire lire le gold corpus par le pipeline de production ni le copier dans un magasin permanent.
- Respecter strictement les critères d'arrêt numériques : ne pas poursuivre une phase juste parce qu'elle est « presque » au seuil — documenter la décision (continuer/arrêter) avec les chiffres à l'appui.
- Artefacts de run bruts (candidats produits par un détecteur) → `pipeline_out/detection_benchmark/` (gitignored, regénérable). Rapports de décision et code réutilisable (scorer, adaptateurs) → `fix_pipeline/detection_benchmark/` (tracké).
- Chaque phase produit un rapport écrit avant de passer à la suivante — c'est ce rapport qui sert de contexte au prompt de la phase suivante après un `/clear`.

## Ordre de travail immédiat

1. Ajouter les trois rôles au corpus local (Phase 0).
2. Implémenter le scorer de spans (Phase 1).
3. Mesurer le pipeline actuel complet, les deux baselines (Phase 2).
4. Tester `rules_plus` (Phase 3).
5. Décider, chiffres à l'appui, si le probe LLM mérite d'être lancé (Phase 4).
6. Valider seulement le gagnant sur STREUSLE et PARSEME EN (Phase 5).
