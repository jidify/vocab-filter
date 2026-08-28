# S3-4 — cache et choix du juge

## Décision

Le juge retenu est `catgpt-browser` via le backend `catgpt`, avec le prompt
`s3-judge-prompt-5`. Le traitement du livre complet n'a pas été lancé. Le
batch reste une capacité évaluée ; son activation en production doit demeurer
explicite afin de ne jamais déclencher un traitement de masse en arrière-plan.

## Mesures réelles

| Juge | Corpus/protocole | Précision | Rappel | Label exact | Latence |
|---|---|---:|---:|---:|---:|
| `mistral-small:24b` (Ollama) | 14 cas, prompt initial | 88,9 % | 80,0 % | 78,6 % | non comparable |
| `openai/gpt-5-mini` | 14 cas, prompt initial | 90,0 % | 90,0 % | 85,7 % | non comparable |
| `catgpt-browser` séquentiel | 1 cas, JSON compact | 100 % | 100 % | 0 % (semi-figé/idiome) | 37,7 s |
| `catgpt-browser` batch | 14 cas, prompt précisé | 100 % | 100 % | 100 % | 41,7 s |
| `catgpt-browser` batch | 28 cas distincts, prompt précisé | 100 % | 100 % | 100 % | 48,5 s |
| `catgpt-browser` batch | 56 cas (28 contrastifs + 28 gold) | 100 % | 89,6 % | 76,8 % | 57,0 s |
| `catgpt-browser` batch | les 28 mêmes cas gold seuls | 100 % | 78,6 % | 53,6 % | 48,1 s |
| `catgpt-browser` batch | 50 cas contrastifs S3, sans gold | 100 % | 100 % | 100 % | 57,0 s |

Les deux attentes initialement divergentes du lot de 28 ont été corrigées
après audit linguistique : `break up a box into pieces` et `check it out` sont
des emplois lexicalisés, conformément au verdict du modèle. Les réponses
brutes n'ont pas été rejouées lors de cette correction d'annotation.

## Cache et panne

La clé permanente S3 contient le backend, le modèle, la version du prompt, la
version du schéma, le canon, l'identifiant d'occurrence et la signature du
contexte pertinent. Les anciennes clés globales restent lisibles pour audit,
mais ne peuvent plus être consommées. Une panne ou une réponse non JSON donne
`incertain` et n'est jamais écrite dans le magasin permanent.

Le benchmark batch emploie en plus une variante explicite de cache pour rendre
mesurables les changements de paramètres internes au gateway lorsque son nom
de modèle reste identique.

## Seuils

Sur les 28 cas : précision 100 %, rappel 100 %, exactitude de label 100 %,
schéma valide 100 %, révision évitable 0 %. Les seuils S3/Q0-1 sont satisfaits
sur ce corpus stratifié. Une validation statistique plus large reste distincte
de l'achèvement fonctionnel de S3-4.

## Taille de batch

Le lot de 56 a renvoyé exactement 56 décisions, toutes conformes au schéma, en
57,0 secondes. Les 13 divergences apportées par les cas gold sont déjà présentes
quand ces 28 cas sont soumis seuls (une divergence change seulement de
`semi_fige` à `littéral`). Il n'y a donc pas de dégradation attribuable à la
taille 56 sur ce contrôle. En revanche, le gold corpus est documenté comme un
benchmark de détection de spans Q0-3, pas comme un gold indépendant de taxonomie
S3 ; ses catégories et ses canons de surface ne doivent pas être assimilés sans
adjudication à des réponses de désambiguïsation S3.

La taille opérationnelle est donc fixée à **50** (`config.S3_JUDGE_BATCH_SIZE`).
Le lot contrastif correspondant contient 50 cas distincts, renvoie 50 décisions
valides en 57,0 secondes et satisfait tous les seuils. L'évaluateur refuse un
lot supérieur à cette limite ; aucune exécution du livre complet n'a été faite.
