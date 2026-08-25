# Pipeline de sélection du vocabulaire d’une œuvre

## Objectif

Le pipeline doit extraire d’un livre, d’une pièce, d’un scénario ou de sous-titres les unités réellement intéressantes pour comprendre l’anglais employé par les natifs.

Il doit répondre simultanément à quatre besoins :

1. sélectionner les mots et expressions susceptibles d’être inconnus d’un apprenant avancé ;
2. ne conserver que le ou les sens effectivement employés dans l’œuvre ;
3. reconnaître les idiomes, phrasal verbs et autres expressions lexicalisées avant de traiter leurs composants comme des mots isolés ;
4. distinguer ce qui aide à comprendre l’œuvre actuelle de ce qui sera réutilisable dans d’autres films, séries ou pièces.

Le pipeline recommandé travaille donc sur des **unités lexicales contextualisées** :

```text
(lemme ou expression, catégorie grammaticale, sens contextuel)
```

Il ne faut pas sélectionner des mots isolés puis chercher leur sens seulement à la fin. Une telle organisation éliminerait des sens rares de mots courants et laisserait passer les composants d’expressions multi-mots.

## Vue d’ensemble

```text
texte complet
→ détection des sections utiles
→ analyse linguistique et indexation des occurrences
→ détection des idiomes et phrasal verbs potentiels
→ validation de leur usage lexicalisé ou littéral
→ attribution d’un sens à chaque occurrence
→ regroupement par unité + POS + sens
→ calcul des signaux pédagogiques
→ comparaison des classements
→ export CSV + JSONL
```

## 1. Délimiter le contenu utile

Le fichier brut d’un ebook ne contient pas uniquement l’œuvre. Il peut aussi contenir des mentions légales, un sommaire, des crédits, une présentation commerciale ou une biographie de l’auteur.

Le pipeline doit conserver :

- les dialogues ;
- la narration ;
- les didascalies ;
- les prologues et épigraphes appartenant réellement à l’œuvre.

Il doit exclure :

- le copyright ;
- le sommaire ;
- les informations commerciales ;
- les crédits et la distribution ;
- les biographies et métadonnées éditoriales.

Cette étape est indispensable : dans le fichier actuel de *The Humans*, le mot `photocopying` apparaît dans les mentions légales et ne doit pas devenir une suggestion de vocabulaire.

La détection automatique des sections doit pouvoir être corrigée par des limites explicites dans la configuration, car la structure varie fortement d’un ebook à l’autre.

## 2. Analyser les occurrences avant de les regrouper

Chaque token doit conserver :

- sa forme originale ;
- son lemme ;
- sa catégorie grammaticale en contexte ;
- sa phrase, sa scène ou son chapitre ;
- ses offsets exacts dans le texte ;
- ses relations de dépendance ;
- son appartenance éventuelle à une expression multi-mot.

Les ressources lexicales doivent être interrogées avec le lemme **et** le POS lorsque celui-ci est disponible.

Le code actuel regroupe tous les niveaux CEFR d’une graphie. Cela conduit par exemple à conserver `water` employé comme nom A1 parce que `water` employé comme verbe est B2. Le filtre doit au minimum distinguer `(water, noun)` de `(water, verb)`.

Le regroupement final doit se faire sur :

```text
(forme canonique, POS, sens contextuel)
```

Si un même lemme apparaît avec deux sens distincts dans l’œuvre, les deux sens produisent deux entrées séparées.

## 3. Détecter les expressions avant les mots simples

### 3.1 Utiliser `idiomatch` comme générateur de candidats

`idiomatch` reste utile pour reconnaître des variantes morphologiques et des expressions discontinues. Le réglage retenu est :

```python
Idiomatcher.from_pretrained(n=2)
```

Il permet notamment de rapprocher :

```text
crack open
crack it open
crack the window open
```

Cependant, `idiomatch` ne doit pas décider seul. Sur le livre actuel, il produit de nombreux candidats compositionnels ou peu intéressants tels que :

```text
go to
talk about
in the room
open doors
```

Il doit donc être considéré comme un outil à haut rappel, pas comme une vérité lexicale.

### 3.2 Ajouter d’autres sources de candidats

Les candidats multi-mots doivent venir de plusieurs sources complémentaires :

- `idiomatch` ;
- les expressions présentes dans WordNet ;
- des ressources annotées comme MAGPIE et PARSEME ;
- des règles de dépendance pour les verbes à particule ;
- des règles pour les formes séparables et constructions résultatives.

Ces ressources améliorent la couverture, mais aucune n’est exhaustive. MAGPIE montre aussi qu’une expression potentiellement idiomatique peut être littérale dans certaines occurrences. La présence dans un lexique ne suffit donc pas : la décision doit être contextuelle.

### 3.3 Valider chaque occurrence

Chaque candidat doit être classé comme :

```text
idiome
phrasal_verb
construction_lexicalisée
usage_littéral
incertain
```

Une expression confirmée réserve ses composants uniquement dans le span concerné.

Exemples :

- `figure out` produit une entrée multi-mot et ne produit pas `figure` seul pour cette occurrence ;
- `open door` littéral ne bloque ni `open` ni `door` ;
- si `figure` apparaît ailleurs avec un sens autonome intéressant, cette autre occurrence reste admissible ;
- `get away with` doit primer sur le candidat imbriqué `get away`.

Cette règle satisfait le besoin « ne pas prendre le mot seul s’il fait partie d’un idiome », sans supprimer abusivement le même mot dans toutes ses autres occurrences.

## 4. Identifier le sens avant la sélection pédagogique

L’inventaire initial des sens peut venir de WordNet, complété par les définitions des expressions multi-mots. Le système doit aussi prévoir explicitement la réponse :

```text
aucun_sens_adapté
```

Cette option est nécessaire lorsque WordNet est trop fin, incomplet ou obsolète pour l’usage rencontré.

Le traitement recommandé est :

1. accepter directement les unités véritablement monosémiques ;
2. exécuter GlossBERT sur un contexte adapté au document ;
3. utiliser SemCor comme prior seulement lorsque le sens est suffisamment attesté ;
4. réserver OMSTI à l’attestation et à la couverture, pas à une estimation de fréquence réelle ;
5. demander un arbitrage au LLM local lorsque les signaux restent ambigus ;
6. regrouper ensuite les occurrences partageant réellement le même sens.

La fenêtre de contexte ne doit plus être fixée universellement à ±2 répliques. Elle doit respecter la structure disponible : répliques voisines dans une pièce, paragraphe dans un roman, ou fenêtre de sous-titres dans une vidéo.

## 5. Pourquoi un LLM local est nécessaire

### 5.1 Ce que `sense_in_context.py` apporte déjà

Le script actuel contient des idées utiles qu’il faut conserver comme baseline :

- utilisation de GlossBERT ;
- contexte élargi ;
- inventaire de sens WordNet ;
- preuve française provenant d’OMW/WOLF ;
- réduction du poids des traductions françaises peu discriminantes ;
- conservation de la marge entre les deux meilleurs sens ;
- 14 cas de régression manuellement vérifiés.

Il ne faut donc pas remplacer aveuglément ce script par un LLM.

### 5.2 Pourquoi le script ne suffit pas pour un livre entier

`sense_in_context.py` n’est pas encore un pipeline généralisable :

- les mots, POS, phrases anglaises et sens attendus sont écrits manuellement dans `TESTS` ;
- les traductions françaises ont également été rédigées manuellement ;
- le livre du dépôt ne contient pas ces traductions ;
- les heuristiques ont été ajustées sur les mêmes 14 exemples qui servent ensuite à annoncer 14/14 ;
- il n’existe pas de jeu de test indépendant ;
- le traitement vise essentiellement les mots simples ;
- il ne décide pas si une expression potentielle est idiomatique ou littérale ;
- GlossBERT reste limité à l’inventaire et aux distinctions de WordNet ;
- l’addition directe du score GlossBERT et du bonus français n’est pas calibrée statistiquement ;
- la marge entre les deux premiers résultats n’est pas encore transformée en probabilité de fiabilité ;
- la segmentation du contexte est adaptée à la mise en page d’une pièce précise ;
- WOLF et WoNeF contiennent du bruit compensé par des coefficients empiriques.

Le résultat 14/14 est un bon test de régression, mais pas encore une preuve de généralisation.

### 5.3 Ce que le LLM doit faire

Le LLM local intervient uniquement dans les situations où un raisonnement contextuel est réellement utile :

- plusieurs sens restent plausibles ;
- GlossBERT possède une marge faible ;
- les signaux disponibles se contredisent ;
- il faut distinguer un idiome d’un usage littéral ;
- la construction est discontinue ;
- aucun sens WordNet ne convient correctement ;
- une paraphrase pédagogique française est nécessaire.

Le LLM doit recevoir une tâche fermée contenant :

- le mot ou span cible ;
- la phrase et un contexte limité ;
- le POS ;
- les sens candidats avec leurs identifiants et définitions ;
- l’option `aucun_sens_adapté`.

Il doit produire un JSON validable :

```text
selected_sense
usage_type
contextual_meaning_fr
evidence
confidence
alternative_senses
```

La température doit être fixée à zéro. Le modèle doit citer l’indice textuel qui justifie son choix. Les sorties non conformes ou peu confiantes doivent aller dans une file de vérification.

### 5.4 Pourquoi une approche hybride est préférable

Les méthodes locales déterministes restent préférables pour :

- la tokenisation ;
- la lemmatisation et les POS ;
- les recherches lexicales ;
- les mots monosémiques ;
- les cas à forte marge ;
- les fréquences et statistiques ;
- la reproductibilité et le coût.

Le LLM apporte une capacité complémentaire pour :

- raisonner sur la phrase complète ;
- reconnaître le caractère compositionnel ou non d’une expression ;
- comprendre les constructions discontinues ;
- comparer des définitions très proches ;
- produire une signification française limitée au contexte ;
- refuser un inventaire de sens inadéquat.

Son gain ne doit pas être supposé. Il doit être mesuré en comparant :

1. `sense_in_context.py` ;
2. GlossBERT sans traduction française manuelle ;
3. le LLM local seul avec un inventaire fermé ;
4. le pipeline hybride.

### 5.5 Ne pas créer une fausse preuve bilingue

Si le LLM produit lui-même une traduction française, cette traduction ne doit pas être réutilisée comme une preuve indépendante en faveur de sa propre décision. Cela créerait une validation circulaire.

Une traduction française peut servir de preuve supplémentaire seulement si elle provient d’une source indépendante, par exemple une traduction humaine ou déjà publiée. Sinon, le LLM doit directement produire le sens contextuel français sans que ce texte soit recompté comme un second modèle.

## 6. Sélection pédagogique et rôle de l’AoA

### 6.1 Abandonner le seuil AoA, pas forcément la donnée

Le filtre du type :

```text
AoA ≥ 5
```

doit être supprimé.

L’AoA de Kuperman représente l’âge auquel des adultes natifs pensent avoir appris un mot. Ce n’est pas une mesure directe de difficulté pour un apprenant L2.

Dans les ressources locales, la corrélation de Spearman entre AoA et Zipf atteint environ `−0,66` sur les mots du livre présents dans les deux jeux de données. Une grande partie de l’information AoA est donc déjà liée à la fréquence.

Le seuil actuel a également un défaut conceptuel : il exclut les mots acquis très tôt par les natifs. Or les apprenants L2 peuvent précisément moins bien connaître certains mots liés à l’enfance, à la famille et à la vie quotidienne. Pour comprendre le « vrai anglais », ces écarts peuvent être plus intéressants que les mots académiques tardifs et transparents pour un francophone.

### 6.2 Donner la priorité à la connaissance L2

Le meilleur signal de difficulté disponible est un classement construit directement à partir de la reconnaissance des mots par des apprenants L2.

Chaque unité doit recevoir des composantes distinctes :

- `need_l2` : probabilité estimée de méconnaissance par un apprenant L2 ;
- `native_coverage` : proportion de natifs connaissant le mot ;
- `book_gain` : fréquence et dispersion dans l’œuvre ;
- `dialogue_reuse` : fréquence dans les sous-titres et conversations ;
- `sense_surprise` : caractère moins banal du sens rencontré ;
- `mwe_opacity` : difficulté à déduire l’expression depuis ses composants ;
- `analysis_confidence` : fiabilité de l’analyse.

La prévalence native ne doit plus être un seuil brutal fixé à `0,99`. Elle devient un signal continu permettant de distinguer :

- le vocabulaire largement partagé par les natifs ;
- le vocabulaire spécifique mais nécessaire dans l’œuvre ;
- les termes spécialisés ou mal couverts par les ressources.

Le CEFR reste un signal secondaire ou de repli, en respectant le POS et, si possible, le sens. Il ne doit pas supprimer un sens avancé d’un lemme dont le sens principal est A1/A2.

### 6.3 Tester trois variantes AoA

Les variantes suivantes doivent être comparées :

1. aucun signal AoA ;
2. AoA brut ;
3. interaction « acquis tôt par les natifs mais tardivement par les apprenants L2 ».

L’AoA ne doit être intégrée au classement principal que si elle améliore de façon stable les annotations humaines. Si les résultats sont ambigus, la variante sans AoA reste la référence.

## 7. Fréquence : difficulté ou utilité

Le classement actuel trie les mots par Zipf croissant. Cela fonctionne pour trouver des curiosités lexicales rares, mais pas nécessairement pour maximiser la compréhension future.

Pour les films, séries et pièces, une fréquence élevée dans les sous-titres ou conversations est un avantage : elle indique que l’unité sera probablement rencontrée de nouveau.

La fréquence ne doit donc pas servir simultanément :

- de proxy principal de difficulté ;
- et de pénalité contre l’apprentissage.

La difficulté doit venir principalement de la connaissance L2. La fréquence doit mesurer la réutilisabilité.

Les sources recommandées sont :

- SUBTLEX-US ou `FreqZipfUS` pour les films et séries ;
- Spoken BNC2014 pour la conversation réelle ;
- le nombre d’occurrences et leur dispersion dans l’œuvre analysée.

## 8. Comparer compréhension immédiate et réutilisabilité

Le score général doit conserver deux axes :

- gain pour comprendre l’œuvre actuelle ;
- probabilité de revoir l’unité dans d’autres dialogues natifs.

Une famille de classements peut être calculée ainsi :

```text
utility(α) =
    besoin d’apprentissage
    × validité native
    × [α × gain dans l’œuvre + (1−α) × réutilisabilité]
```

Les valeurs à comparer sont :

```text
α = 0
α = 0,25
α = 0,5
α = 0,75
α = 1
```

Le rapport doit montrer :

- les déplacements dans les top 50, 100 et 200 ;
- les unités propres à l’œuvre ;
- les unités fortement réutilisables ;
- la couverture des occurrences du livre ;
- la sensibilité du résultat au choix de `α`.

Toutes les unités valides restent exportées. Le classement `α=0,5` peut servir de vue initiale, sans masquer les autres variantes.

## 9. Sorties attendues

Le pipeline doit produire :

- un CSV facile à relire ;
- un JSONL conservant toutes les occurrences, décisions et provenances ;
- une file séparée de vérification des cas incertains ;
- un rapport comparant les variantes AoA et les pondérations.

Champs principaux :

```text
canonical_form
surface_forms
unit_type
pos
sense_id
meaning_fr
definition_en
occurrences
book_count
dispersion
need_l2
native_coverage
dialogue_reuse
book_gain
sense_surprise
mwe_opacity
confidence
alternatives
needs_review
claimed_component_spans
provenance
```

Les cas incertains restent dans la sortie exhaustive avec `needs_review=true`.

## 10. Validation

### 10.1 Échantillon humain

Sélectionner 50 unités de manière stratifiée :

- mots simples et expressions ;
- niveaux de fréquence variés ;
- AoA précoce et tardive ;
- cas monosémiques et polysémiques ;
- confiance forte et faible ;
- usages littéraux et idiomatiques.

Pour chaque unité, annoter :

```text
connu / inconnu
utile / inutile
sens correct / incorrect
expression lexicalisée / littérale
```

Avec seulement 50 unités, cette expérience permettra de comparer les variantes et de détecter les grandes erreurs, mais pas de revendiquer une validation statistique définitive.

### 10.2 Évaluation du sens

Comparer sur les mêmes occurrences :

1. le comportement actuel de `sense_in_context.py` ;
2. GlossBERT seul ;
3. le LLM local seul avec inventaire fermé ;
4. le pipeline hybride.

Mesurer :

- l’exactitude du sens ;
- la précision des expressions confirmées ;
- la couverture ;
- l’exactitude par niveau de confiance ;
- les refus `aucun_sens_adapté` ;
- le temps de traitement.

Les 14 cas existants doivent rester des tests de régression, mais ne doivent pas être comptés comme jeu d’évaluation indépendant.

### 10.3 Régressions fonctionnelles

Vérifier notamment :

- l’absence de candidats issus des mentions légales ;
- la jointure CEFR par POS ;
- la reconnaissance de `figure out`, `wing it`, `get away with` et `crack … open` ;
- le rejet de `go to`, `in the room` et `open door` lorsqu’ils sont littéraux ;
- la priorité de l’expression la plus spécifique ;
- l’absence de mot isolé dans une occurrence réclamée par une expression ;
- la conservation des occurrences autonomes du même mot ;
- la séparation de plusieurs sens du même lemme ;
- la présence des alternatives pour les cas incertains ;
- la reproductibilité des sorties structurées du LLM local.

## 11. Pipeline recommandé final

```text
ŒUVRE UTILE
    ↓
segmentation + offsets + scènes/chapitres
    ↓
lemmes + POS + dépendances
    ↓
candidats MWE à haut rappel
    ↓
validation littérale/lexicalisée
    ↓
réservation des spans MWE confirmés
    ↓
candidats mots simples restants
    ↓
inventaire des sens par occurrence
    ↓
GlossBERT + priors prudents
    ↓
LLM local pour les ambiguïtés
    ↓
regroupement par unité + POS + sens
    ↓
connaissance L2 + prévalence native
    ↓
gain dans l’œuvre + réutilisabilité dialoguée
    ↓
variantes AoA et α
    ↓
CSV + JSONL + file de vérification
```

La décision centrale est donc la suivante : **ne pas remplacer `sense_in_context.py`, mais le transformer en baseline locale au sein d’un pipeline hybride**. GlossBERT et les ressources lexicales traitent les cas structurés ; le LLM local arbitre uniquement les phénomènes que les scores lexicaux ne savent pas résoudre de manière fiable, notamment la polysémie fine et l’opposition entre expression idiomatique et usage littéral.
