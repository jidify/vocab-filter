# TODO — schéma JSON unitaire répété dans chaque item du prompt de lot S3-judge-occurrence

**Statut : constaté, pas traité.** Observé le 2026-08-30 en vérifiant le
prompt réellement envoyé lors d'un run S1→S5 sur *The Humans* avec
`VOCAB_LLM_S3_JUDGE_OCCURRENCE="catgpt/catgpt-browser;batch=true;batch_size=50"`.

## Le constat

`pipeline/mwe_judge.py::judge_occurrences_batch` construit le texte de
chaque occurrence via `_occurrence_prompt()` — la même fonction que le
chemin unitaire (`judge_occurrence`). Cette fonction termine chaque bloc par
le rappel complet du schéma JSON **unitaire** :

```
Réponds en JSON strict avec ce schéma :
{"label": "<une des 5 catégories>", "canonical_form": "<canon>", ...}
```

`OCC_BATCH_PROMPT_TEMPLATE` concatène ensuite ces `count` blocs (un par
occurrence) et ajoute, une seule fois à la fin, le schéma **batch**
(`{"decisions":[...]}`). Résultat vérifié en lot réel : pour
`batch_size=50`, le schéma JSON unitaire est répété 50 fois dans le même
prompt avant le schéma batch final — pur bruit, puisque seul le schéma
batch de fin est pertinent pour la réponse attendue.

Vérifié par rendu direct du prompt (2 items d'exemple, mêmes fonctions que
le code de production) :

```python
from pipeline import mwe_judge as mj
p, wn = mj._occurrence_prompt(idiom, occ, segments_by_idx)
# p contient déjà "Réponds en JSON strict avec ce schéma : {...}" à la fin
```

## Pourquoi c'est peut-être pertinent

- Gonfle sensiblement la taille du prompt à `batch_size` élevé (ici 50),
  ce qui peut jouer sur le coût, la latence, et — pour un backend comme
  `catgpt/catgpt-browser` (gateway navigateur, pas d'API native) — sur la
  fiabilité/lenteur observée en pratique (un premier appel de lot 50 est
  resté bloqué plus de 45 minutes sans erreur ni progression visible avant
  d'être tué manuellement).
- N'affecte a priori PAS la justesse des décisions (le schéma batch final
  reste correct et `_normalize_occurrence_result` ne dépend pas du schéma
  unitaire répété), donc ce n'est pas un bug fonctionnel — plutôt un coût
  caché.

## Mise à jour — plan de décorrélation lot/stockage

Le point 2 ci-dessous coûtait auparavant un rejeu complet de S3 (le prompt
rendu faisait partie de la clé de cache, `pipeline_out/cache/`) : retirer le
rappel de schéma unitaire aurait changé le texte de CHAQUE prompt, donc
invalidé tout le cache d'un coup. Ce n'est plus vrai depuis le passage à
`llm_client.run_units()`/`pipeline/llm_store.py` : la clé du magasin unitaire
porte sur l'entrée SÉMANTIQUE de l'occurrence (`mwe_judge._occurrence_payload`),
jamais sur le texte du prompt rendu — retirer ce rappel ne coûterait plus
qu'un bump de `mwe_judge.S3_PROMPT_VERSION` (le vrai levier d'invalidation
volontaire désormais), pas un rejeu accidentel. Le point 3 est également
tranché : vérifié, la même redondance existe bien pour
`S3-definition-cluster` (`DEFINITION_PROMPT_TEMPLATE`) et `S5-arbitrate`
(`ARBITRATION_TEMPLATE`) — les 4 tâches S6 (prompts unitaire/lot distincts,
`build_unit_user_prompt`/`build_user_prompt`) n'ont pas ce défaut.

## Pas encore fait

1. Mesurer l'impact réel en tokens/latence du schéma répété à
   `batch_size=50` (comparer taille de prompt avec/sans le rappel de
   schéma unitaire dans chaque bloc).
2. Si l'impact est significatif : faire en sorte que `_occurrence_prompt`
   (et les équivalents `_definition_request`/`_arbitration_prompt`)
   n'émettent le rappel de schéma unitaire que pour le chemin unitaire, pas
   pour le chemin lot — par exemple via un paramètre
   `include_schema_reminder: bool = True` désactivé côté lot. Bumper la
   constante de version de la tâche concernée en même temps (voir
   ci-dessus).
3. ~~Vérifier si la même redondance existe pour d'autres tâches en lot~~ —
   fait, voir ci-dessus.
4. Décider si ça vaut la peine face au reste du plan `fix_pipeline` — pas
   un gate qualité nommé, juste un potentiel gain de coût/fiabilité.
