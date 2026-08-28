# Probe hors plan — fix de tokenizer (`?—`/`!—` collés)

Ne fait pas partie de la numérotation Phase 0-6 de
`plan_detection_benchmark_funnel.md` : question posée après la Phase 6,
sur l'opportunité de nettoyer/corriger le texte avant analyse. Réponse
mesurée ici, chiffres à l'appui, avant toute décision de l'adopter en
production.

## Ce qui a été testé

`fix_pipeline/detection_benchmark/tokenizer_boundary_fix.py` (tracké,
réutilisable) : deux motifs d'infixe **ajoutés** (jamais retirés) au
tokenizer spaCy — un tiret cadratin/demi-cadratin collé sans espace à une
ponctuation fermante (`?!.,;:'")]` , guillemets courbes) se scinde
maintenant correctement, dans les deux sens. Constat de départ : la règle
d'infixe par défaut de spaCy n'exige un tiret qu'ENTRE deux caractères
alphabétiques — `"around—we"` se scinde bien, `"around?—we"` reste collé
en un seul token.

- **Scan diagnostic sur les 2535 segments du livre** (générique — détecte
  la signature du bug, pas une liste de caractères figée, voir
  `scan_suspect_tokens`) : **30 tokens suspects sur 30327 avant** le
  patch, **1 seul restant après** (le résidu, `"that—'member"`, une
  élision par apostrophe, volontairement pas chassé — trop ambigu à
  généraliser sans risquer de casser des contractions légitimes
  ailleurs). Zéro régression mesurée sur les traits d'union légitimes
  (`ground-floor`, `e-mail`, `and/or`, `smart-ass`, `triple-A`...).
- **Aucun fichier de production modifié** : le patch s'applique en
  mémoire sur le `nlp` retourné par `pipeline.analyze.get_nlp()`, pour la
  durée du script probe seulement (`phase_tokenizer_fix_probe.py`).
- Rejoue Baseline 2 (Phase 2) et `rules_plus` (Phase 3, version déjà
  itérée avec les deux affinements de rappel) à l'identique, tokenizer
  patché, sur les mêmes 99 segments gold.

## Résultat — impact mesuré sur les 99 segments gold : quasi nul sur le rappel, léger gain de précision

| Indicateur | Sans le fix | Avec le fix (patché) | Δ |
|---|---:|---:|---:|
| Baseline 2 — rappel global exact | 46,3% | 46,3% | **0** |
| Baseline 2 — `hard_negative` capturés | 10/27 | **9/27** | **-1** |
| `rules_plus` — rappel global exact | 70,7% | 70,7% | **0** |
| `rules_plus` — rappel MWE exact | 67,6% | 67,6% | **0** |
| `rules_plus` — phrasal verbs séparables exact | 85,7% | 85,7% | **0** |
| `rules_plus` — `hard_negative` capturés | 13/27 | **12/27** | **-1** |

Diff exact des candidats baseline2 (avant → après) :

- **Perdu** : `(seg102, [57:74])` = `"stomps around?—we"` — c'était
  exactement le piège `hard_negative` idx102 du corpus gold
  (`edge_type=dialogue_dash`), capturé à tort par `multi_token` avant le
  fix (le token-poubelle `'around?—we'` était mal étiqueté PROPN et
  absorbé dans un span `compound`). Le fix supprime cette capture — un
  vrai gain de précision, pas de rappel.
- **Gagné** : `(seg1277, [40:52])` = `"Mary statue?"` — un nouveau
  candidat `multi_token`/`nominal_compound`, mais qui ne correspond à
  AUCUN span gold (ni positif, ni piège) : neutre pour le score.

## Pourquoi le rappel ne bouge pas ici, alors que le fix est réel

Les deux seuls segments gold touchés par ce motif de tokenizer
(`seg102`, `seg1277`) n'ont **aucun span positif MWE bloqué par la
tokenisation elle-même** :

- `seg102` : le span positif attendu est `"stomps around"`
  (`phrasal_verb_inseparable`). Une fois correctement tokenisé, `stomp`
  redevient un verbe normal et `around` une particule normale — mais ni
  PARSEME (86 entrées TRAIN) ni WordNet n'attestent `stomp`+`around`
  comme couple verbe-particule : le scanner `rules_plus` ne génère
  toujours aucun candidat, pas parce que le texte est mal tokenisé, mais
  parce que le **lexique** ne couvre pas ce couple informel. C'est
  exactement la limite "longue traîne du lexique" déjà documentée dans
  `phase3_rules_plus_report.md` (ex. `stomp around`, jamais recensé nulle
  part) — un problème différent, pas résolu par ce fix.
- `seg1277` : le seul span positif du segment (`"bringing up"`,
  `[11:22]`) est bien AVANT le `?—` cassé (`[36:58]`) — jamais affecté
  par le bug, déjà correctement détecté avant comme après.

Le fix est donc réel et vérifié (29/30 sur le livre entier, zéro
régression), mais son bénéfice ne se voit quasiment pas sur ce corpus de
99 segments précisément parce que les 2 segments concernés n'ont pas de
MWE positive coincée derrière ce motif — la coïncidence joue en
sa défaveur ici, pas contre son principe. Sur les 2436 segments restants
du livre (hors gold, jamais mesurés faute de corpus annoté dessus), rien
n'exclut que d'autres occurrences du même motif bloquent une vraie MWE —
juste invérifiable sans agrandir le corpus gold, ce que le plan interdit
explicitement à ce stade (Phase 0 : "Ne pas l'agrandir maintenant").

## Recommandation

**Fix à part entière, à adopter pour ce qu'il est — une correction de
bug de tokenizer, pas un levier de rappel démontré.** Argumentaire :

- Coût nul : aucune réécriture de texte, aucun impact sur les offsets
  caractère (contrairement à un nettoyage du texte source), patch
  analogue à `EMAIL_SPECIAL_CASES` déjà en production.
- Gain de précision mesuré et net : -1 `hard_negative` capturé à tort,
  sur les deux baselines.
- Gain de rappel non démontré sur ce corpus (0 sur les 99 segments gold)
  — ne pas le présenter comme tel dans une décision d'architecture.
- Généralise au-delà de *The Humans* par construction (motif de
  ponctuation, pas liste de mots) — contrairement à un lexique custom.

**Pas encore appliqué à `pipeline/analyze.py`** (hors périmètre de ce
probe, comme demandé) — l'intégrer en production reste un chantier
séparé à décider, avec le même statut que le reste de `rules_plus` :
utile, à coût nul, mais à ne pas confondre avec les gains de rappel de
Phase 3 (ceux-là, eux, sont mesurés et réels).
