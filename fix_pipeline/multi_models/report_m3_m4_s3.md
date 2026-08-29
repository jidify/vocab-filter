# Rapport M3/M4 — tâches S3 sur slot modèle et prompt lot

## Écart avant / après

| Sujet | Avant | Après |
|---|---|---|
| `S3-judge-occurrence` | backend/modèle globaux, un appel par occurrence | slot `S3-judge-occurrence`; unitaire par défaut, ou lots de taille configurée |
| Prompt occurrence | seul prompt scalaire | prompt scalaire conservé + enveloppe batch `decisions[]` indexée par `occurrence_id` |
| Panne / réponse batch incomplète | panne unitaire → `incertain`, non stocké | même traitement; identifiant manquant ou dupliqué → `incertain` non stocké pour l'occurrence concernée |
| Cache S3 | modèle/backend mais pas le mode | métadonnées client et clé de magasin incluent tâche, modèle, mode et taille effective |
| `S3-definition-cluster` | backend/modèle globaux, un appel par cluster | slot dédié; unitaire par défaut, ou lots de clusters existants sans modifier leur définition |
| Prompt définition | objet scalaire | objet scalaire + enveloppe `decisions[]` indexée par `cluster_id` |

`mode_batch` reste strictement le regroupement de plusieurs cas dans un même
prompt. Aucun usage de `litellm.batch_completion` n'a été ajouté.

## Vérifications

- Les fixtures offline couvrent les deux chemins de `S3-judge-occurrence`, dont
  les identifiants manquants et dupliqués.
- Elles couvrent les deux chemins de `S3-definition-cluster`; le test lot
  vérifie que les clusters déjà produits restent ceux consommés.
- Le client JSON accepte maintenant le provider explicite du slot (`ollama`,
  `catgpt` ou `openai`) et le test OpenAI est mocké.
- Le défaut du registre M1 reste `mode_batch=false`, `batch_size=1` pour les
  deux tâches S3; sans variable dédiée, le comportement de prompt est donc
  unitaire.

## Gate suivant

M5 — `S5-arbitrate`: raccord au slot, deux prompts unitaire/lot et tests mock,
sans changer la politique qui décide quand l'arbitrage est sollicité.
