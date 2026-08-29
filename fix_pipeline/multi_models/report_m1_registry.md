# Lot M1 — registre et parseur de configuration LLM

## Périmètre

Implémentation limitée au registre et à la résolution de configuration. Aucun module métier S3–S6
ne consomme encore ce registre ; aucun appel réseau, prompt, cache ou parallélisme HTTP n'a changé.

## Avant / après

| Point | Avant (M0) | Après (M1) |
|---|---|---|
| Identifiants de tâche | inventaire documentaire | 9 descripteurs immuables dans `pipeline/llm_tasks.py` |
| Défauts | constantes et littéraux dispersés | modèle, `mode_batch` et `batch_size` figés par tâche |
| Override dédié | absent | `VOCAB_LLM_<TASK_ID>=provider/model;batch=...;batch_size=...` |
| Repli S3/S5 | backend global direct | repli préservé sur `VOCAB_LLM_BACKEND` et modèle global |
| Alias gateway | `PROVIDER=chatgpt` décoratif | `chatgpt` normalisé en `catgpt` |
| Validation | absente | tâche/provider/options inconnus, doublons et batch invalide refusés |
| Taille effective | implicite | `effective_batch_size(...) == 1` en unitaire |

Les chemins S6 conservent leurs défauts déclarés 40 / 40 / 20 / 10. Le plafond Stage C à 20 reste
un défaut de registre seulement : son branchement et son découpage appartiennent à M2.

## Vérifications

- Tests écrits avant le module : échec initial attendu avec `ModuleNotFoundError`.
- `python -m unittest test_llm_tasks.py` : 9 tests, succès.
- `python -m unittest test_llm_backends.py test_llm_tasks.py` : 11 tests, succès.
- Chaque tâche `batch_allowed=true` est résolue et testée en configuration unitaire et lot. Il ne
  s'agit pas encore des deux familles de prompts métier : elles restent volontairement aux lots de
  branchement M2–M5 conformément au plan.

## Gate suivant

M2 : brancher les quatre tâches S6 déjà batchées sur le registre, puis livrer pour chacune un vrai
chemin de prompt unitaire et un chemin de prompt lot, avec schémas/parsers et tests hors réseau.
