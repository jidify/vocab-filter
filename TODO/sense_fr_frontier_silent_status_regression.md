# TODO — sense_fr_frontier.run() peut écraser silencieusement un statut déjà verrouillé par un pire résultat

**Statut : constaté, pas traité.** Découvert le 2026-08-31 en comparant
`data/sense_fr.jsonl` avant/après le passage complet en catgpt
(`fix_pipeline`, commit "Retranslate S6 sense store via catgpt...") : de
nombreuses entrées auparavant traduites avec confiance sont repassées
`pending` — vide dans `vocab.csv` — sans qu'aucun avertissement n'ait été
émis pendant le run.

## Le constat

`pipeline/sense_fr_frontier.py::run()` recalcule `resolved`/`unresolved`
depuis `senses.jsonl`/`selected_mwe.jsonl` à CHAQUE exécution, puis exclut
seulement les entrées `PROTECTED_STATUSES = {"validated", "auto_joint"}`
(`is_protected()`, ligne ~452) avant de construire les lots à traduire.
Pour TOUT le reste — y compris `auto_strong`/`auto_corroborated`/
`auto_judged`, des statuts pourtant considérés "verrouillés" ailleurs dans
le dispositif (voir `pipeline/verify_fr_lock.py`) — la boucle finale fait :

```python
for target, _occs, _candidates in items:
    translation = translations_by_key.get(target["key"])
    entry = build_entry(target, translation, model=model)
    store[entry["key"]] = entry          # écrasement inconditionnel
```

Aucune comparaison entre l'ancienne et la nouvelle entrée. Un nouvel essai
qui échoue (erreur de schéma, item absent/dupliqué du lot — voir le commit
ci-dessus) ou qui aboutit à un jugement moins confiant écrase quand même
l'ancienne réponse, potentiellement bien meilleure.

### Mesuré précisément ce jour (diff avant/après le run catgpt complet)

**193 entrées** avaient un statut verrouillé AVANT ce run et sont `pending`
APRÈS — aucune n'était `validated` (seul statut réellement protégé qui a
tenu), confirmant que la faille touche spécifiquement les 3 statuts
"confiants mais non protégés" :

| Statut avant | Nombre régressé |
|---|---:|
| `auto_strong` | 137 |
| `auto_corroborated` | 41 |
| `auto_judged` | 15 |

Deux causes distinctes, à ne pas confondre — mesurées séparément via
`agreement` de la nouvelle entrée `pending` :

- **Cause mécanique (46 cas, `frontier_sans_reponse`)** : le lot a
  purement et simplement échoué (item absent/dupliqué de la réponse du
  gateway catgpt, ou lot au complet non parsable) — l'ancienne traduction,
  potentiellement parfaite, est perdue pour une raison n'ayant RIEN à voir
  avec sa qualité.
- **Cause de jugement (137 cas, `sense_id_douteux` + `sense_id_suspect`)** :
  catgpt signale `sense_fit="doubtful"`/`"mismatch"` sur des sens que
  openai/gpt-5-mini avait jugés `"ok"` sans réserve lors du run précédent
  (ex. `absolutely.r.01`, `admit.v.01`, `apprehensible.s.01`...). Ce n'est
  PAS forcément une régression réelle — catgpt a peut-être raison d'être
  plus prudent — mais rien dans le dispositif actuel ne permet de le
  savoir : aucun échantillon audité comparable à celui de S6-3
  (`fix_pipeline/evaluate_s6_3_translation_leakfree.py`) n'existe pour les
  désaccords de `sense_fit` entre deux passages.
- Reste (10 cas) : `frontier_sans_ressource`, `frontier_desaccord`,
  `frontier_explicitation`, `frontier_reformulation` — mêmes portes de
  blocage normales de `sense_fr.blocks_auto_lock`, déclenchées cette
  fois-ci alors qu'elles ne l'étaient pas avant.

Exemples concrets vérifiés dans `vocab.csv` : `touch` (« toucher » →
vide), `haggard` (« émacié » → vide), `calm` (traduit → vide sur 3 des 4
sens), `facility` (« installations » → vide).

## Pourquoi c'est grave

- `data/sense_fr.jsonl` est le magasin **permanent**, réutilisé d'un livre
  à l'autre (voir la docstring de `pipeline/sense_fr.py`) — une régression
  ici ne dégrade pas seulement le run courant, elle dégrade silencieusement
  le dictionnaire partagé pour tout futur livre qui réutiliserait ces
  `sense_id`.
- Ça casse la comparabilité de toute mesure Q0-1/S6-3 dans le temps : si un
  run "d'amélioration" peut faire régresser des centaines d'entrées déjà
  bonnes sans le signaler, un chiffre de fidélité avant/après ne veut plus
  rien dire sans aussi differ le magasin brut — ce qui n'est fait
  actuellement par AUCUN outil du dépôt.
- Même principe que le plan applique déjà partout ailleurs (S5-3 pour les
  occurrences, S6-2 pour le routage des `pending`) : « aucune perte
  silencieuse » — ici c'est le même défaut, mais au niveau du STATUT plutôt
  que de l'occurrence, et rien ne le couvre encore.

## Correction proposée

1. **Retry avant abandon** : sur l'échec d'un lot (`ValidationError`
   pydantic, item absent/dupliqué — voir `on_failure` dans `run_units()`),
   retenter le sous-ensemble concerné à une granularité plus fine (lot
   coupé en deux, ou repli en mode unitaire pour les seuls items en échec)
   avant de le classer `pending`. Élimine la part purement mécanique
   (46/193 mesurés ici).
2. **Ne plus écraser silencieusement un statut confiant par un pire
   résultat** : avant `store[entry["key"]] = entry`, comparer l'ancien
   `status` au nouveau. Si l'ancien est dans
   `{"auto_strong", "auto_corroborated", "auto_judged"}` (pas seulement
   `validated`/`auto_joint`) et que le nouveau est `pending` (ou toute
   régression de confiance) :
   - au minimum, compter et signaler bruyamment (`n_regressed` dans les
     logs + artefact dédié `pipeline_out/sense_fr_regressions.csv` avec
     clé, ancien statut/fr, nouvelle raison) ;
   - envisager de GARDER l'ancienne entrée par défaut plutôt que de
     l'écraser, sauf demande explicite de reclassement (un futur drapeau
     du type `--allow-relock`, symétrique de `--retry-pending`).
3. **Gate d'exception sur régression massive** : si un run fait régresser
   plus qu'un seuil (ex. 5 % des entrées déjà verrouillées avant ce run)
   sans intervention explicite, lever une erreur plutôt que d'écrire le
   magasin silencieusement — même logique que
   `pipeline/export.py::assert_no_uncertain_occurrence_lost`, appliquée
   aux statuts du magasin plutôt qu'aux occurrences.
4. **Ne pas traiter les deux causes de la même façon** : la cause
   mécanique (1) se résout par un retry ; la cause de jugement (137 cas
   `sense_id_douteux`/`sense_id_suspect`) demande un audit humain, pas une
   nouvelle tentative automatique — réutiliser le motif de S6-3
   (échantillon auditable, contresens résiduels) pour les désaccords de
   `sense_fit` entre deux passages/modèles, afin de savoir si catgpt a
   raison d'être plus prudent ou s'il est juste moins bien calibré.

## Pas encore fait

1. Implémenter le retry de lot (point 1) dans
   `pipeline/sense_fr_frontier.py::_translate_units`/`run_units`.
2. Ajouter la comparaison avant/après et l'artefact
   `sense_fr_regressions.csv` (point 2).
3. Ajouter le gate d'exception sur régression massive (point 3), avec un
   seuil à valider (proposition : 5 %, à recaler sur un run réel une fois
   1-2 en place).
4. Construire l'échantillon audité des 137 désaccords `sense_fit`
   catgpt/openai (point 4) avant de décider s'il faut les restaurer, les
   garder en révision, ou faire confiance au nouveau jugement.
5. Une fois ces garde-fous posés, ce sera le bon moment pour refaire tourner
   `sense_fr_frontier.py`/`sense_fr_adjudicate.py` en catgpt sur le
   magasin ACTUEL (déjà partiellement régressé par ce run) et vérifier que
   la régression ne se reproduit plus.
