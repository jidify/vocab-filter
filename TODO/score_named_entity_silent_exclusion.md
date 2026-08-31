# TODO — exclusion silencieuse des sens WordNet d'entité nommée dans aggregate_and_score

**Statut : constaté, pas traité.** Découvert le 2026-08-31 en tentant de
régénérer `pipeline_out/vocab.csv` (`uv run python -m pipeline.export`) après
un passage complet de S6-1/S6-2 en catgpt (`sense_fr_frontier.py` puis
`sense_fr_adjudicate.py --with-backtranslation --with-judge`).

## Le constat

`pipeline/score.py::aggregate_and_score`, ligne ~338 :

```python
for key, occs in grouped.items():
    lemma, wn_pos, sense_id = key
    if is_named_entity_sense(sense_id):
        # ... le sens réellement retenu par S5 est celui d'une entité
        # nommée — jamais du vocabulaire à apprendre.
        continue
```

`is_named_entity_sense(sense_id)` (score.py:126) est correct dans son
intention : il détecte un sens WordNet à `instance_hypernym` (« est une
instance de », marqueur des noms propres — personne/lieu/organisation —
plutôt que « est une sorte de »). Exemple vérifié : le sens retenu par S5
pour "Brigid" (un personnage de la pièce) est `bridget.n.01`, littéralement
**Sainte Brigide d'Irlande** dans WordNet — un vrai faux ami de
désambiguïsation, à raison exclu du vocabulaire à exporter.

Le problème n'est pas l'exclusion elle-même, c'est le `continue` nu : il
saute le groupe ENTIER (toutes les occurrences de ce `(lemma, pos,
sense_id)`), y compris celles marquées `needs_review=True` par S5, **sans
laisser aucune trace** — ni `review_queue.csv`, ni raison enregistrée, ni
compte. C'est exactement la classe de défaut que l'invariant S5-3
(`pipeline/export.py::assert_no_uncertain_occurrence_lost`, appelé par
`export.run()` avant toute écriture) est censé empêcher — mais ce chemin
d'exclusion précis ne semble jamais avoir été couvert par cet invariant
depuis son introduction : `pipeline.export.run()` n'avait apparemment
jamais été exécuté pour de vrai depuis, jusqu'à aujourd'hui.

**Vérifié pré-existant, sans rapport avec le passage catgpt de ce jour** :
rejouer le même invariant avec le magasin `data/sense_fr.jsonl` tel que
committé (`git show HEAD:data/sense_fr.jsonl`, AVANT toute retraduction)
produit exactement le même échec, aux mêmes clés et mêmes comptes près.
**Vérifié aussi que le `vocab.csv` déjà présent dans le dépôt (mesuré à
99,12 % de fidélité par S6-3, `fix_pipeline/evaluate_s6_3_translation_leakfree.py`)
manque déjà silencieusement 8 de ces occurrences** — seule `pig smash`
(voir plus bas) y figure. Ceci viole directement le seuil §0 du plan :
« disparitions silencieuses d'occurrences | 0 ».

### Occurrences concrètement perdues (mesurées ce jour)

| lemme | pos | sense_id retenu par S5 | occurrences perdues |
|---|---|---|---:|
| brigid | n | bridget.n.01 (Sainte Brigide) | 12 |
| carnegie | n | carnegie.n.02 | 2 |
| lord | n | godhead.n.01 | 4 |
| christ | n | jesus.n.01 | 2 |
| earth | n | earth.n.01 (la planète) | 1 |
| world | n | earth.n.01 (la planète) | 1 |
| massachusetts | n | massachusetts.n.01 | 1 |
| farmer | n | farmer.n.03 | 1 |

`('pig smash', None, 'mwe:pig smash:semi_fige')` apparaît aussi dans le
diagnostic de l'invariant, mais **son mécanisme semble différent** — cette
entrée existe bien dans `vocab.csv` (`needs_review=True`,
`occurrences=1`), donc pas de perte silencieuse constatée pour elle ;
`is_named_entity_sense` n'a rien à voir ici (clé re-typée par
`data/manual_corrections.jsonl` vers une clé `mwe:...`, `pos=None` non
renseigné dans la correction — probablement un mismatch `None`/`""` entre
la clé attendue et celle réellement exportée). **Pas encore diagnostiqué en
détail, à vérifier séparément** avant de le considérer couvert par le même
correctif que ci-dessus.

## Correction retenue

Option « plus conforme à l'esprit du plan » (S5-3 « bifurcation, jamais
suppression » ; S7-3 « candidats = exportés + exclus avec raison +
révision ») plutôt que la version minimale (apprendre juste à l'invariant à
accepter cette exclusion sans rien changer à l'agrégation) : faire que
`aggregate_and_score` **enregistre** ces exclusions "entité nommée" quelque
part d'auditable (raison + occurrences), au lieu du `continue` muet
actuel — pour qu'elles restent comptables et relisibles, cohérent avec le
reste du dispositif (`review_queue.csv`, invariant comptable de S7-3).

## Pas encore fait

1. Décider où loger ces exclusions auditées : nouveau champ/liste renvoyé
   par `aggregate_and_score` (à côté de `units`), ou ligne dédiée dans
   `review_queue.csv`/un nouvel artefact `pipeline_out/named_entity_exclusions.csv`.
2. Modifier le `continue` de score.py:338 pour construire cette trace
   (lemma, pos, sense_id, nombre d'occurrences, raison `"named_entity_sense"`)
   avant de sauter le groupe.
3. Mettre à jour `assert_no_uncertain_occurrence_lost` (ou son appelant)
   pour compter ces exclusions tracées comme couvertes, au lieu de lever
   `RuntimeError` — l'invariant doit rester strict sur toute perte NON
   tracée, mais ne plus bloquer sur une exclusion volontaire désormais
   auditable.
4. Étendre les tests de non-régression (`test_q0_2_regression.py` ou
   dédié) avec `brigid`/`bridget.n.01` comme cas positif : occurrence
   `needs_review=True` sur un sens d'entité nommée, doit apparaître dans la
   trace d'exclusion, jamais dans `vocab.csv`, jamais silencieusement nulle
   part.
5. Diagnostiquer séparément le cas `pig smash` (clé `mwe:...` issue d'une
   correction manuelle, `pos=None`) — probablement un bug de comparaison de
   clé distinct, pas la même cause racine.
6. Une fois corrigé, relancer `pipeline.export` sur le magasin déjà
   retraduit en catgpt (`data/sense_fr.jsonl` du 2026-08-31) pour produire
   enfin un `vocab.csv` à jour, puis Q0-1/S6-3 pour comparer au benchmark.
