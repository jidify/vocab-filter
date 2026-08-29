# S5-4 — fragments de composés et d'entités

## Baseline

Le comparateur Q0-1 reproduit les **9 canons mot** présents uniquement dans
la sortie historique : `crystal`, `ease`, `forth`, `nursing`, `observation`,
`ranch`, `tighten`, `virgin`, `york`. La métrique multiensemble reste séparée
(16 lignes mot, homonymes compris).

Les six défauts directement visés par S5-4 assignaient encore un synset simple
à un token couvert : `New York`, `Virgin Mary`, `ranch dip`,
`observation deck`, `nursing home`, `crystal ball`. Les trois autres canons
relèvent de MWE confirmées et de la réservation S4-2.

## Correction

- S5 reconnaît les candidats couvrants structurellement confirmés (NER >= 0,90
  ou dépendance `nominal_compound` >= 0,80) avant toute ouverture WordNet.
- L'occurrence reçoit une exclusion auditée
  `covered_by_confirmed_multi_token` dans `senses.jsonl`; aucun faux synset
  simple n'est assigné ni exporté.
- La décision est portée par l'`occurrence_id`. Une occurrence autonome du même
  lemme, sans span couvrant confirmé, continue vers la résolution normale.
- Le même garde précède le repli `top_k`; une limitation de calcul ne peut donc
  pas réintroduire le faux sens dominant.
- Une hypothèse faible ou de simple frontière ne possède aucun pouvoir de
  suppression.

## Vérification

Les tests ciblés couvrent les six contextes, l'occurrence autonome, l'hypothèse
faible, le filtre d'export occurrence-scoped, S1-2, S4-2, S5-2/S5-3 et la mesure
Q0-1 des neuf canons. Résultat : **37/37 tests réussis**.

Le benchmark n'est jamais lu par le pipeline de production ; il sert seulement
à la mesure en lecture seule dans `fix_pipeline/evaluate_fix_quality.py`.
