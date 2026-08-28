# Test rapide des modèles spaCy pour S1-2

## Objectif

Voir rapidement si `en_core_web_lg` ou `en_core_web_trf` détecte manifestement mieux les entités et composés multi-tokens que l'actuel `en_core_web_sm` sur *The Humans*.

Ce test est exploratoire : il doit donner une tendance en moins de quelques heures de travail, pas constituer une validation scientifique.

## Périmètre

Comparer les trois modèles sur :

- `The Humans - Stephen Karam.txt` ;
- les lignes anglaises de `The Humans - Stephen Karam-TRAD.txt`.

Le bilingue reprend largement le même anglais. Il sert à vérifier que l'extraction et la segmentation ne changent pas les conclusions ; ses occurrences ne sont pas comptées comme un second corpus indépendant.

Ne lancer que spaCy et `pipeline.multi_token`. Ne lancer ni GlossBERT, ni LLM, ni les étapes aval. Ne modifier ni le modèle de production ni les artefacts courants.

## Déroulement

### 1. Petit script de comparaison

Créer un script isolé acceptant un modèle et une source. Pour chaque run, conserver le modèle et sa version, le temps total, les candidats avec leurs offsets, types et provenances, ainsi que les nombres de candidats NER et `compound`.

Écrire uniquement sous `pipeline_out/spacy_quick_compare/`.

### 2. Lancer les modèles

Exécuter `sm`, `lg` et `trf` sur le texte anglais simple, puis sur l'anglais extrait du bilingue : six runs courts au total.

Si `lg` ou `trf` n'est pas installé, demander l'autorisation avant téléchargement. Si `trf` est trop lent ou manque de mémoire, arrêter ce run et le signaler : le test doit rester court.

### 3. Comparaison automatique

Produire un rapport compact contenant :

- le résultat des six expressions `New York`, `Virgin Mary`, `ranch dip`, `observation deck`, `nursing home`, `crystal ball` ;
- les candidats trouvés uniquement par un modèle ;
- les candidats identiques ayant des bornes différentes ;
- les différences entre les deux versions du livre ;
- le temps de traitement de chaque modèle.

Tous les offsets doivent être vérifiés automatiquement contre leur source.

### 4. Audit manuel léger

Relire seulement :

- les six cas demandés ;
- au maximum 30 désaccords choisis de manière déterministe ;
- 10 accords communs comme contrôle.

Pour chaque exemple, noter `correct`, `incorrect`, `bornes incorrectes` ou `incertain`. Temps humain visé : 15 à 30 minutes.

## Décision

Le rapport conclut uniquement parmi :

- aucune différence utile visible : conserver `sm` ;
- `lg` semble meilleur : envisager un test plus sérieux avant remplacement ;
- `trf` semble meilleur : comparer le gain au surcoût avant remplacement ;
- résultats mixtes : conserver `sm` et éventuellement utiliser le meilleur modèle seulement sur les cas incertains.

Ne pas changer automatiquement le pipeline de production après ce test rapide.

## Durée visée

- script et rapport : 1 à 2 heures ;
- six runs : quelques minutes à environ une heure selon la machine et `trf` ;
- audit humain : 15 à 30 minutes.

Durée totale visée : une demi-journée maximum, souvent moins si les modèles sont déjà installés.

## Livrables

```text
pipeline_out/spacy_quick_compare/results.json
pipeline_out/spacy_quick_compare/disagreements.jsonl
pipeline_out/spacy_quick_compare/report.md
```

Le rapport doit donner quelques exemples concrets d'améliorations et de régressions sans imposer l'ouverture des fichiers intermédiaires.
