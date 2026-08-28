# Plan d'action — qualité de `vocab.csv` équivalente au benchmark

## 0. Objectif, règle d'or et mesure de départ

L'objectif est que le pipeline produise automatiquement `pipeline_out/vocab.csv` avec une qualité lexicale, sémantique, traductionnelle et pédagogique équivalente à `pipeline_out/vocab_corrige.csv`. Le benchmark est une **vérité d'évaluation** : il ne doit jamais être lu par le pipeline de production, copié dans les magasins permanents ni transformé en liste d'exceptions propre à *The Humans*.

Ordre impératif des travaux :

1. préserver une occurrence correcte et son contexte ;
2. détecter l'unité lexicale complète ;
3. décider le POS et le sens par occurrence ;
4. regrouper uniquement les occurrences de même sens ;
5. traduire ce sens verrouillé ;
6. décider si l'unité représente une difficulté pédagogique ;
7. exporter ou envoyer en révision, sans suppression silencieuse.

Baseline observée le 2026-08-28 (les nombres doivent être recalculés par l'outil d'évaluation, car les deux CSV contiennent des formes canoniques dupliquées) :

- `vocab.csv` : 1 091 lignes ; benchmark : 1 052 lignes ;
- comparaison grossière `(canonical_form, unit_type)` : 43 unités seulement dans le résultat courant, dont 34 MWE et 9 mots ; 11 MWE seulement dans le benchmark ;
- parmi les lignes appariables : 98 MWE sans POS, 126 MWE avec un `sense_id` différent, 38 définitions MWE corrigées, 14 mots avec POS/sens corrigé et 99 mots dont la traduction officielle vide est remplie dans le benchmark ;
- 53 lignes courantes ont une forme canonique exactement identique à leur traduction officielle, dont `affection` et `intelligible`.

Définition opérationnelle de « qualité équivalente » pour le gate final (Q0-1 doit calculer ces valeurs ; les variantes françaises jugées synonymes par la métrique souple comptent comme correctes) :

| Dimension | Seuil final minimal |
|---|---:|
| précision des unités conservées | 97 % |
| rappel des unités du benchmark | 97 % |
| exactitude POS des unités appariées | 99 % |
| exactitude d'identité de sens | 97 % |
| définition compatible avec le sens/contextes | 98 % |
| traduction officielle présente | 100 % |
| traduction sémantiquement acceptable | 98 % |
| cohérence canon–sens–définition–FR | 100 % sur les cas non révisés |
| gates critiques nommés dans S7-4 | 100 % |
| disparitions silencieuses d'occurrences | 0 |

Ces seuils ne permettent pas de compenser un défaut critique par de bons résultats ailleurs. Une unité incertaine peut être envoyée en révision et sortir du dénominateur « non révisé », mais Q0-1 doit aussi publier le taux de révision : il ne doit pas dépasser 5 % des candidats finaux, afin d'éviter d'atteindre artificiellement la précision en différant toutes les décisions difficiles.

### Correction Q0-1 — construire l'évaluateur de référence

- **Pré-requis** : les deux CSV existent ; aucune correction fonctionnelle n'a encore besoin d'être appliquée.
- **Description** : créer un comparateur reproductible qui normalise Unicode/casse/apostrophes, gère les homonymes et compare séparément inventaire, spans/surfaces, POS, identité de sens, définition, traduction et décision garder/exclure. Il doit produire un JSON de métriques et un rapport Markdown, sans modifier le benchmark.
- **Résultat attendu** : `pipeline_out/fix_quality_metrics.json` et `pipeline_out/fix_quality_report.md` donnent la baseline, les écarts par étape et une liste de cas nommés. Les métriques minimales sont : précision/rappel des unités, précision MWE, rappel MWE, exactitude POS, exactitude du sens, exactitude de la définition, couverture FR, cohérence sens–définition–FR, précision de la sélection pédagogique et taille de la file de révision.
- **Vérifications** : le rapport retrouve au minimum les écarts `come to`, `let someone go`, `burn out`, `latch`, `affection`, `intelligible`, `facility`, `frosting`, `York` et les 99 traductions officielles manquantes. Un test avec deux lignes homonymes prouve que le comparateur ne les écrase pas dans un dictionnaire indexé seulement par `canonical_form`.
- **Gate qualité** : la baseline générée est stable sur deux exécutions et toute différence avec les comptes ci-dessus est expliquée par le rapport.

### Correction Q0-2 — créer un corpus de régression stratifié

- **Pré-requis** : Q0-1 validée.
- **Description** : extraire des artefacts existants un jeu de cas minimal, sans utiliser les réponses du benchmark à l'exécution du pipeline. Les attentes vivent uniquement dans les tests. Strates : MWE fusionnées, MWE manquées, polysémie MWE, mauvais POS/lemme, sens WordNet erroné, entités/composés, `aucun_sens_adapte`, traduction transparente, traduction pending.
- **Résultat attendu** : une suite de tests ciblés donne une localisation de régression par étape et un test end-to-end compare le CSV final au benchmark.
- **Vérifications** : chaque anomalie citée en Q0-1 appartient à au moins une strate ; les tests savent tourner hors réseau avec réponses LLM figées et disposent séparément d'un mode d'évaluation réel non déterministe.
- **Gate qualité** : tous les défauts connus échouent avant correction pour la bonne raison ; aucun test ne valide une simple chaîne propre au livre sans invariant généralisable.

## 1. Correction de S1 — analyse des occurrences

### Correction S1-1 — préserver forme, lemme, POS et alternatives d'analyse

- **Pré-requis** : Q0-1 et Q0-2 validées ; `pipeline_out/occurrences.jsonl` reste la source auditée.
- **Description** : ne plus faire du couple spaCy `(lemma, wn_pos)` une vérité irrévocable. Conserver dans chaque occurrence la surface, le POS spaCy, le lemme spaCy, les alternatives plausibles issues de la morphologie et les indicateurs de forme fléchie/nominalisée. Ajouter un identifiant de version d'analyse.
- **Résultat attendu** : S5 pourra ouvrir un inventaire alternatif sans reparcourir le livre. `frosting`, `creeping`, `facilities`, `stressing` et `bitch` conservent les analyses permettant respectivement les sens `frosting.n.01`, `creeping.s.01`, `facilities.n.02`, `de_stress.v.01` et `bitch.n.01`.
- **Vérifications** : dans `occurrences.jsonl`, chaque cas possède sa surface exacte et au moins l'analyse alternative attendue ; les offsets pointent toujours exactement dans le segment source ; les tests existants de zones et d'inventaire restent verts.
- **Qualité estimée** : prépare la correction des 14 erreurs POS/sens mesurées, sans gain final tant que S5 n'utilise pas les alternatives.

### Correction S1-2 — annoter les composés et entités multi-tokens

- **Pré-requis** : S1-1 validée.
- **Description** : produire des candidats de composé/entité avec spans exacts, sans supprimer les tokens simples à S1. Combiner dépendances spaCy, NER et patrons nominaux ; conserver score et provenance.
- **Résultat attendu** : `New York`, `Virgin Mary`, `ranch dip`, `observation deck`, `nursing home` et `crystal ball` sont visibles comme spans candidats dans un artefact auditable, tout en gardant leurs tokens composants disponibles jusqu'à S4.
- **Vérifications** : les spans et offsets sont exacts ; `York`, `Virgin`, `ranch`, `observation`, `nursing` et `crystal` ne peuvent plus être validés comme unités autonomes sans que le candidat couvrant soit examiné ; aucune réservation n'a encore lieu.
- **Qualité estimée** : couvre directement les 6 faux mots/composés les plus visibles du benchmark.

### Correction S1-3 — fiabiliser et enrichir les candidats VPC

- **Pré-requis** : S1-1 validée ; contrat VPC existant compris et tests `test_vpc_contract.py` verts.
- **Description** : conserver tous les VPC syntaxiquement plausibles avec spans exacts et contexte syntaxique, y compris ceux absents de PARSEME. Ajouter les compléments nécessaires pour distinguer sens et constructions (`burn out` avec sujet humain ou ampoule, particules, objets, attributs).
- **Résultat attendu** : `burn out` apparaît quatre fois avec des traits contextuels distinctifs ; les candidats séparables `get worked up`, `steer clear of`, `put to rest` et `tighten one's belt` sont proposés par S1 ou explicitement routés vers le détecteur MWE général de S2.
- **Vérifications** : `vpc_candidates.jsonl` conserve décision, preuve, tokens membres et compléments ; absence de PARSEME n'est jamais une preuve de littéralité ; les quatre occurrences de `burn out` sont retrouvées.

## 2. Correction de S2 — génération des candidats MWE

### Correction S2-1 — empêcher les canons génériques de capturer des constructions différentes

- **Pré-requis** : S1 complet et artefacts d'occurrences/spans fiables.
- **Description** : modifier l'alignement `idiomatch` pour distinguer slots autorisés et mots lexicaux interposés. Une correspondance ne peut pas réduire arbitrairement `come back to earth`, `come home to` ou `come talk to` à `come to`. La surface complète et une signature syntaxique sont obligatoires.
- **Résultat attendu** : `mwe_candidates.jsonl` ne contient plus un type `come to` réunissant les six surfaces actuelles ; `look to` ne contient plus `look, love to`, `go for` ne contient plus `gonna sleep for`, `get better` ne contient plus `get a better` et `back up` ne contient plus `back to picking up`.
- **Vérifications** : assertions négatives sur ces mélanges ; assertions positives séparées pour `come back to earth` et les occurrences réellement lexicalisées ; aucun token intermédiaire significatif n'est perdu dans l'identité candidate.
- **Qualité estimée** : élimine la famille d'erreurs de fusion la plus destructrice, dont le cas initial `come to`.

### Correction S2-2 — augmenter le rappel des MWE du benchmark

- **Pré-requis** : S2-1 validée afin que le gain de rappel ne recrée pas de sur-fusion.
- **Description** : fusionner trois sources de candidats : lexiques MWE, syntaxe VPC/composés et proposition contextuelle contrôlée. Permettre les slots possessifs/pronominaux et flexions, avec provenance explicite.
- **Résultat attendu** : les candidats couvrent `let it go`, `come back to earth`, `get worked up`, `at ease`, `burn out`, `put to rest`, `steer clear of`, `could care less` et `tighten our belts` sous un canon approprié.
- **Vérifications** : présence des neuf familles dans `mwe_candidates.jsonl`, avec surface et offsets exacts ; chaque nouveau patron comporte au moins un test négatif compositionnel ; le rappel MWE contre le benchmark augmente sans baisse de précision supérieure au seuil défini par Q0-1.

### Correction S2-3 — normaliser l'identité d'occurrence sans décider du sens

- **Pré-requis** : S2-1 et S2-2 validées.
- **Description** : séparer explicitement `candidate_form`, `observed_surface`, `member_spans`, `full_span`, `syntactic_signature` et `candidate_sources`. Ne pas attribuer encore une définition unique ni un sens.
- **Résultat attendu** : deux détecteurs peuvent fusionner la même occurrence physique, mais deux analyses lexicales différentes restent deux hypothèses concurrentes jusqu'à S3.
- **Vérifications** : `let it go` peut être candidat à la fois sous une construction générique et sous l'idiome exact sans écrasement ; les collisions sont résolues par identifiant de span + hypothèse, pas uniquement par span.

## 3. Correction de S3 — jugement lexical et désambiguïsation MWE

### Correction S3-1 — juger chaque occurrence avant le type

- **Pré-requis** : S2 complet ; exemples Q0-2 disponibles.
- **Description** : remplacer le verdict global fondé sur les trois premières occurrences par une décision occurrence par occurrence pour tous les candidats sémantiquement variables, quelle que soit leur source. Sortie minimale : lexicalisé/littéral/incertain, canon proposé, POS, paraphrase contextuelle et confiance calibrable.
- **Résultat attendu** : `let it go`, `let him go` et `let's go` reçoivent des décisions distinctes ; les quatre `burn out` ne partagent pas mécaniquement un verdict ; `could care less` est lexicalisé ; `ask for`, `at home` et autres combinaisons transparentes peuvent être rejetés ou différés au filtre pédagogique.
- **Vérifications** : `mwe_decisions.jsonl` contient une décision par occurrence et jamais seulement une décision de type pour un groupe hétérogène ; tests sur les cas positifs et négatifs ; aucune confiance autodéclarée du LLM n'est utilisée seule comme preuve.
- **Qualité estimée** : corrige les 34 faux MWE retirés du benchmark et récupère une partie des 11 MWE manquantes, sous réserve de S3-2.

### Correction S3-2 — regrouper les occurrences par sens

- **Pré-requis** : S3-1 validée.
- **Description** : regrouper après jugement, sur canon + POS + sens/paraphrase compatible. Utiliser les sens DBnary/WordNet lorsqu'ils correspondent exactement ; sinon créer un identifiant custom stable et versionné. Une catégorie telle que `phrasal_verb` reste un `unit_type/label`, jamais un `sense_id`.
- **Résultat attendu** : `burn out` possède des unités de sens distinctes pour « s'épuiser » et « griller/s'éteindre » ; `come back to earth` a sa propre clé ; `let it go` n'est pas confondu avec licencier ou laisser partir quelqu'un.
- **Vérifications** : identifiants différents pour deux sens et identifiant identique pour deux occurrences réellement synonymes ; le champ POS est rempli ; aucune MWE n'a `idiome`, `phrasal_verb` ou `semi_fige` comme seul `sense_id`.
- **Qualité estimée** : traite les 98 POS MWE vides et les 126 identités MWE divergentes observées.

### Correction S3-3 — sélectionner la définition correspondant au contexte

- **Pré-requis** : S3-2 validée et inventaires de sens accessibles.
- **Description** : supprimer le choix `senses[0]`. Comparer toutes les définitions candidates aux occurrences du cluster ; autoriser une définition custom si aucune entrée exacte n'existe.
- **Résultat attendu** : les définitions de `break up`, `bring up`, `check in`, `get a grip`, `give out`, `keep up`, `look after`, `turn off` et `work out` correspondent au livre.
- **Vérifications** : les 38 divergences de définition MWE sont mesurées ; aucune ligne n'a une traduction qui contredit sa définition ; tests contrastifs entre les sens concurrents.

### Correction S3-4 — rendre le cache sûr et évaluer le modèle juge

- **Pré-requis** : schéma de décision S3-1 à S3-3 stabilisé.
- **Description** : clé de cache = version du prompt + modèle + version du schéma + canon + signature/contexte pertinent. Invalider les anciennes décisions globales incompatibles. Comparer le modèle local et un modèle frontière sur Q0-2 ; retenir le moins coûteux qui satisfait les seuils, avec escalade du résidu difficile.
- **Résultat attendu** : une ancienne erreur telle que `could care less = littéral` ne survit pas à un changement de protocole ; un rapport donne précision/rappel par modèle et par strate.
- **Vérifications** : modification de version provoque bien un nouveau jugement ; panne LLM produit `incertain`, jamais une décision persistée ; le modèle choisi atteint les seuils S3 définis dans Q0-1.

## 4. Correction de S4 — sélection et inventaire lexical figé

### Correction S4-1 — adopter une unité `(canon, POS, sense_id)` pour mots et MWE

- **Pré-requis** : S3 complet ; sens MWE stabilisés.
- **Description** : unifier le schéma des mots et MWE. Les formes de surface ne sont agrégées que pour une même clé sémantique. Conserver la liste exacte des occurrence IDs et zones.
- **Résultat attendu** : `selected_mwe.jsonl` contient POS, véritable `sense_id`, définition choisie et occurrences homogènes ; `lexical_inventory.jsonl` référence ces clés de sens.
- **Vérifications** : aucune union de surfaces entre sens différents ; `burn out` a plusieurs clés ; `come back to earth`, `come home to` et `come talk to` ne partagent pas de clé ; digest d'inventaire cohérent.

### Correction S4-2 — réserver les spans sans perdre les alternatives utiles

- **Pré-requis** : S4-1 validée.
- **Description** : réserver seulement les membres/spans des occurrences MWE confirmées. Une MWE rejetée rend ses tokens aux mots simples ; une hypothèse incertaine reste révisable et ne supprime rien.
- **Résultat attendu** : les tokens de `latch` restent disponibles ; les composants d'une fausse MWE réapparaissent ; un composé confirmé empêche en revanche `York` ou `Virgin` d'être exporté seuls dans cette occurrence.
- **Vérifications** : tests de couverture exacts, discontinus et concurrents ; conservation occurrence par occurrence ; aucun filtre au niveau du lemme entier.

### Correction S4-3 — différer les exclusions pédagogiques dépendantes du sens

- **Pré-requis** : S4-1 validée.
- **Description** : S4 ne doit exclure que le bruit certain. CEFR, transparence bilingue et difficulté sont conservés comme signaux ; la décision finale attend S6/S7, où le sens et la traduction sont connus.
- **Résultat attendu** : une forme banale avec sens rare peut atteindre S5 ; les métadonnées nécessaires au filtre final sont présentes sans décision prématurée.
- **Vérifications** : tests avec un même lemme ayant un sens trivial et un sens difficile ; aucune exclusion de type n'efface le second.

## 5. Correction de S5 — désambiguïsation des mots simples

### Correction S5-1 — intégrer la réassignation conjointe lemme/POS/sens

- **Pré-requis** : alternatives S1-1 disponibles et inventaire S4 figé.
- **Description** : ouvrir d'abord l'inventaire attendu, puis les alternatives de lemme/POS si aucun sens ne convient ou si le contexte les favorise. Intégrer cette étape dans `run_pipeline.py`, au lieu de laisser `sense_fr_reassign` hors du chemin standard.
- **Résultat attendu** : `frosting`, `creeping`, `facilities`, `stressing` et `bitch` peuvent recevoir les identités du benchmark ; chaque réassignation conserve l'analyse initiale et sa justification.
- **Vérifications** : présence des sens attendus dans `senses.jsonl` ou l'artefact de réassignation ; aucune invention d'un ID absent des inventaires ; tests de non-régression sur les 14 cas connus.

### Correction S5-2 — remplacer le seuil de marge brut par une politique calibrée

- **Pré-requis** : S5-1 validée ; corpus Q0-2 annoté.
- **Description** : calibrer l'arbitrage sur les erreurs réelles. Prendre en compte compatibilité du POS, qualité de localisation de cible, cohérence bilingue, entropie des candidats, nature composé/entité et accord de modèles. Une grande marge GlossBERT ne vaut pas preuve absolue.
- **Résultat attendu** : `facility`, `plow`, `haggard`, `spa`, `barely`, `poke`, `roll` et `touch` sont arbitrés ou correctement sélectionnés ; `needs_review` reflète une incertitude mesurée.
- **Vérifications** : matrice de confusion avant/après ; exactitude du sens sur Q0-2 ; calibration par tranches de confiance ; aucune régression des tests `verify_senses_regression.py`.

### Correction S5-3 — faire de `aucun_sens_adapte` une bifurcation, pas une suppression

- **Pré-requis** : S5-1 et S5-2 validées.
- **Description** : lorsqu'aucun sens ne convient, tenter successivement autre POS/lemme, composé/MWE couvrant, clé custom justifiée, puis file de révision. Ne jamais supprimer dans `build_records()` une occurrence marquée `needs_review`.
- **Résultat attendu** : `latch` reste dans les artefacts aval avec les candidats `latch.n.01/.02` ou une décision révisable ; les usages réellement hors inventaire sont visibles dans un artefact dédié.
- **Vérifications** : `latch` est présent dans `senses.jsonl`, l'inventaire aval, la file de révision si non résolu, et finalement `vocab.csv` une fois résolu ; aucune ligne `aucun_sens_adapte` ne disparaît sans trace et raison comptabilisée.

### Correction S5-4 — traiter les entités et composés avant le filtre WordNet final

- **Pré-requis** : S1-2 et S4-2 validées.
- **Description** : utiliser les spans candidats et le contexte pour empêcher l'assignation d'un sens WordNet simple à un fragment de composé/entité. Étendre le filtre au-delà des seuls `instance_hypernyms`.
- **Résultat attendu** : `York=maison d'York`, `Virgin=personne vierge`, `ranch=exploitation`, `observation=acte d'observer` et `nursing=profession` ne sont plus exportés pour les contextes composés observés.
- **Vérifications** : assertions sur `New York`, `Virgin Mary`, `ranch dip`, `observation deck`, `nursing home`, `crystal ball` ; les occurrences autonomes éventuelles restent traitées indépendamment.

## 6. Correction de S6 — traduction du sens verrouillé

### Correction S6-1 — imposer une entrée cohérente au traducteur

- **Pré-requis** : S3/S5 produisent canon, POS, sens, définition et occurrences homogènes.
- **Description** : traduire une clé de sens, jamais une catégorie MWE ni un groupe hétérogène. Fournir définition validée, deux contextes représentatifs au minimum si disponibles et traduction de phrase. Faire vérifier explicitement la cohérence sens–définition–FR.
- **Résultat attendu** : aucune compensation silencieuse du type « mauvaise définition, bonne traduction ». `look after`, `give out`, `turn off` et `work out` ont des triplets cohérents.
- **Vérifications** : taux de contradiction nul sur les fixtures ; `sense_fit=mismatch/doubtful` empêche le verrouillage automatique ; les preuves sont conservées.

### Correction S6-2 — résoudre ou router toutes les traductions `pending`

- **Pré-requis** : S6-1 validée.
- **Description** : utiliser ressources indépendantes, second jugement ciblé et contexte pour résoudre le pending. Si la décision reste incertaine, router en révision ; ne pas présenter une ligne vide comme finalisée.
- **Résultat attendu** : les 99 traductions remplies dans le benchmark (`fit`, `watering`, `flush`, `overtone`, `settle`, `sound`, etc.) sont soit correctement remplies, soit explicitement absentes du CSV final et présentes dans la file de révision.
- **Vérifications** : couverture `meaning_fr_official` de 100 % dans `vocab.csv` ; toute entrée pending se trouve dans `sense_fr_review.csv`/`review_queue.csv` avec contexte et proposition ; aucune perte silencieuse.

### Correction S6-3 — mesurer la traduction sur le benchmark sans fuite

- **Pré-requis** : S6-1/S6-2 validées.
- **Description** : comparer traduction principale, alternatives et fidélité sémantique avec normalisation souple ; distinguer variante acceptable et contresens. Utiliser le benchmark seulement après génération.
- **Résultat attendu** : rapport par statut/source/modèle ; liste résiduelle de contresens (`touch`, `haggard`, MWE mal traduites) et non simple taux de chaînes égales.
- **Vérifications** : échantillon manuel des accords souples ; métrique reproductible ; aucune lecture du benchmark dans les modules S6 de production.

## 7. Correction de S7 — filtre pédagogique et export final

### Correction S7-1 — ajouter une porte d'éligibilité pédagogique au niveau du sens

- **Pré-requis** : traductions S6 complètes et identités de sens stables.
- **Description** : décider garder/exclure avec traduction officielle, transparence orthographique, faux amis, fréquence/CEFR, surprise du sens et intérêt de la MWE. Une identité EN=FR n'est pas automatiquement exclue si elle masque un faux ami ou une difficulté, mais les cognats directs sans difficulté le sont.
- **Résultat attendu** : `affection=affection`, `intelligible=intelligible` et les cognats transparents comparables sont exclus ; les faux amis comme `sensible` restent évalués sur leur sens réel ; les MWE compositionnelles/triviales du benchmark sont retirées.
- **Vérifications** : les 53 identités exactes initiales sont classées avec raison ; fixtures positives/négatives ; précision de sélection pédagogique contre le benchmark au seuil Q0-1.

### Correction S7-2 — exporter une ligne cohérente par unité de sens

- **Pré-requis** : S7-1 validée.
- **Description** : exporter uniquement des unités complètes `(canon, surfaces homogènes, POS, sense_id, définition, FR, contextes)`. Conserver les catégories MWE dans un champ distinct si nécessaire. Ne jamais concaténer des surfaces de sens différents.
- **Résultat attendu** : `vocab.csv` ne contient plus `come to` mélangé ; `burn out` a des lignes distinctes par sens ; POS et IDs MWE sont renseignés ; aucune traduction officielle vide.
- **Vérifications** : contraintes de schéma automatisées ; unicité de la clé de sens ; cohérence occurrence/surface ; comparaison Q0-1.

### Correction S7-3 — rendre la file de révision exhaustive et actionnable

- **Pré-requis** : branches d'incertitude S3, S5 et S6 disponibles.
- **Description** : agréger dans `review_queue.csv` les incertitudes de détection, POS/sens, traduction et éligibilité. Ajouter raison, candidats, contexte, provenance et action attendue.
- **Résultat attendu** : aucune unité n'est silencieusement supprimée ; `latch` y figure tant qu'il n'est pas résolu ; les `pending`, `aucun_sens_adapte` et conflits de cluster sont visibles.
- **Vérifications** : invariant comptable `candidats = exportés + exclus avec raison + révision`, à granularité occurrence ; test de chaque branche.

### Correction S7-4 — gate finale de qualité et non-régression multi-livres

- **Pré-requis** : toutes les corrections précédentes validées.
- **Description** : exécuter le pipeline complet, Q0-1 et les tests hors benchmark. Fixer les seuils à partir de la qualité de `vocab_corrige.csv`, puis tester sur au moins un autre texte afin d'éviter les exceptions propres à *The Humans*.
- **Résultat attendu** : qualité globale équivalente au benchmark, écarts résiduels explicitement classés comme variantes acceptables ou révision humaine ; aucun défaut critique connu.
- **Vérifications finales obligatoires** :
  - `latch` conservé et correctement routé ;
  - aucune ligne `come to` ne mélange les cinq surfaces fautives ;
  - `let it go`, `come back to earth`, `could care less`, `at ease`, `burn out`, `put to rest`, `steer clear of`, `tighten one's belt` présents si éligibles ;
  - sens distincts de `burn out` avec IDs distincts ;
  - `affection` et `intelligible` exclus pour transparence dans ces sens ;
  - corrections POS/sens des 14 mots du benchmark ;
  - fragments `York`, `Virgin`, `ranch`, `observation`, `nursing`, `crystal` non exportés dans leurs composés ;
  - zéro `meaning_fr_official` vide dans le CSV final ;
  - zéro MWE avec POS vide ou catégorie grammaticale utilisée comme `sense_id` ;
  - invariant comptable S7-3 satisfait ;
  - tous les tests existants et nouveaux passent.

## 8. Politique de livraison de chaque correction

Chaque correction est un lot indépendant. Elle doit :

1. commencer par recalculer ou lire la baseline pertinente ;
2. ajouter les tests qui échouent avant le correctif ;
3. modifier le minimum d'étapes nécessaire, sans exception codée pour une ligne du benchmark ;
4. régénérer uniquement les artefacts requis en respectant les digests et caches ;
5. exécuter tests ciblés, tests de non-régression et comparateur Q0-1 ;
6. publier un mini-rapport avant/après avec gains, régressions et cas encore ouverts ;
7. ne passer au lot suivant que lorsque le `Résultat attendu` et toutes les `Vérifications` du lot sont satisfaits.

Une hausse du score global ne suffit jamais si elle masque une régression critique. Les gates nommés (`latch`, séparation des MWE, absence de suppression silencieuse, traduction complète) sont bloquants.
