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

## Pas encore fait

1. Mesurer l'impact réel en tokens/latence du schéma répété à
   `batch_size=50` (comparer taille de prompt avec/sans le rappel de
   schéma unitaire dans chaque bloc).
2. Si l'impact est significatif : faire en sorte que `_occurrence_prompt`
   n'émette le rappel de schéma unitaire que pour le chemin `judge_occurrence`
   (unitaire), pas pour `judge_occurrences_batch` — par exemple via un
   paramètre `include_schema_reminder: bool = True` désactivé côté lot.
3. Vérifier si la même redondance existe pour d'autres tâches en lot du
   registre (`S3-definition-cluster`, `S5-arbitrate`, tâches S6) avant de
   ne corriger qu'un seul cas isolé.
4. Décider si ça vaut la peine face au reste du plan `fix_pipeline` — pas
   un gate qualité nommé, juste un potentiel gain de coût/fiabilité.
