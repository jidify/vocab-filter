# Bug — `idiomatch` sur-généralise ses slots pronominaux (candidats S2 bruités)

## Statut

**Corrigé** — voir `report_s2_gates_idiomatch_overmatch.md` (portes A/C/D de
`pipeline/mwe_gates.py`) pour l'implémentation, la mesure avant/après et les
limites restantes. `know someone` retombe de 111 à 3 occurrences ; deux
défauts additionnels du même détecteur (`I do`, `wing it`) trouvés et
corrigés dans le même lot. La famille de sur-fusion par remplissage lexical
(`come to`/`go to`, Correction S2-1) reste ouverte, hors périmètre de ce
correctif.

Repéré, documenté initialement sans correctif. Découvert le 2026-08-29 en relançant S3
(`mwe_judge.run()`) pour de vrai sur *The Humans*, en observant les
décisions renvoyées par le modèle sur des candidats manifestement absurdes.
Rattaché à **Correction S2-1** de `fix_pipeline/plan_action_fix_pipeline.md`
§2 (« empêcher les canons génériques de capturer des constructions
différentes ») — même famille de défaut que le cas déjà cité par le plan
(`come to` réunissant six surfaces distinctes), mais un cas concret
supplémentaire, chiffré, non encore traité par ce correctif.

## Symptôme

Le type candidat `"know someone"` (source : `idiomatch`, voir
`pipeline/mwe.py`) réunit **111 occurrences** dans
`pipeline_out/mwe_candidates.jsonl` — pour comparaison, le livre entier ne
totalise que 1580 occurrences MWE candidates au global sur 542 types :
`"know someone"` représente à lui seul **7 % de tout le volume S3** de ce
run. La quasi-totalité de ces 111 occurrences n'ont *aucun rapport* avec
l'idiome transitif « connaître quelqu'un » — `idiomatch` matche n'importe
quel `know`/`knew`/`knows` intransitif ou négatif dès qu'un pronom apparaît
à proximité, comme si ce pronom saturait le slot « someone » de l'entrée
lexicale.

Échantillon réel de surfaces matchées sous ce même type candidat (relevées
dans `pipeline_out/mwe_candidates.jsonl`, occurrence_ids entre parenthèses) :

| occurrence_id | surface matchée | ce qui se passe réellement |
|---|---|---|
| `m:916:50:68` | « I can go, **you know** » | `you know` = marqueur discursif, pas « connaître quelqu'un » |
| `m:957:83:99` | « She doesn't **know** » | négation intransitive simple |
| `m:960:20:26` | « **I know** » | acquiescement, intransitif |
| `m:961:0:6` | « **I know** » | idem |
| `m:962:0:14` | « **I know, I know** » | idem, répété |
| — | « it is not **known** » | passif, sans objet |
| — | « She's [who the hell **knows** » | intransitif |
| — | « **knew you** » | ordre inversé, toujours pas « connaître qqn » au sens visé |
| — | « you guys don't even **know** » | intransitif |

Autres exemples vus dans le même relevé : `knows she`, `We know`, `I do
know` — aucun n'est l'idiome « know someone » au sens qui justifierait une
entrée lexicale dédiée.

## Pourquoi c'est un problème, pas juste du bruit inoffensif

- **Coût direct** : chacune de ces 111 occurrences déclenche une décision
  S3 individuelle (S3-1 juge occurrence par occurrence, jamais le type
  global — voir `mwe_judge.py::judge_occurrence`/`judge_occurrences_batch`)
  → 111 verdicts LLM payés (temps + coût réel selon le modèle configuré)
  pour un candidat qui n'aurait jamais dû atteindre S3.
- **S3 rattrape le coup, mais seulement en partie** : les décisions
  observées sont `incertain`/`littéral`/`semi_fige`, jamais `idiome` ni
  `phrasal_verb` — donc aucun faux positif final n'est exporté grâce au
  garde-fou S3-1. Mais `I know`/`I know, I know` a été jugé `semi_fige`
  avec confiance 0.9-0.99 (voir capture d'écran de la session, confidence
  0.99 sur `m:960:20:26` et `m:961:0:6`) — un jugement *défendable* pour
  « I know » pris isolément comme marqueur figé, mais qui n'a stricto
  sensu rien à voir avec l'idiome candidat « know someone » qui l'a fait
  atteindre S3 en premier lieu. Le système fonctionne malgré la source, pas
  grâce à elle.
- **Signal perdu pour l'audit** : `count`/`occurrences` de `know someone`
  dans les rapports Q0 laisse croire à un idiome fréquent dans le livre,
  alors qu'il s'agit d'un artefact de sur-appariement.

## Cause probable (à confirmer avant correctif)

`idiomatch` est une bibliothèque tierce (`from idiomatch import
Idiomatcher`, voir `pipeline/mwe.py:21` et la docstring du module,
« générateur de candidats à haut rappel »). Le fichier de ressources
`idioms.yml`/Wiktionary définit `know someone` avec un slot ouvert
(`someone`) — l'hypothèse la plus probable, à vérifier dans le moteur de
correspondance d'`idiomatch` (`idiomatch/idiomatcher.py`, méthode de
matching des patterns à slots), est que ce slot accepte **n'importe quel
pronom OU aucun complément du tout** (matchant même l'intransitif), plutôt
que d'exiger un objet nominal/pronominal qui suit réellement `know` dans
une position d'argument transitif. `pipeline/mwe.py:137-` fait tourner
idiomatch segment par segment et consomme ses matches quasiment tels
quels (voir la boucle autour de la ligne 173, `"source": "idiomatch"`) —
rien dans ce dépôt ne re-valide aujourd'hui qu'un slot ouvert a bien été
saturé par un complément syntaxiquement plausible.

## Ampleur possible au-delà de `know someone`

Un relevé heuristique rapide (candidats dont l'idiome fait ≥2 mots mais
dont des occurrences font ≤2 tokens de surface) remonte **354 types
candidats** sur 1580 — **ce chiffre est un signal grossier, pas une liste
de bugs confirmés** : il inclut beaucoup de vrais positifs attendus (un
idiome à 2 mots comme `go to`/`come back`/`thank you` produit légitimement
des surfaces courtes). `know someone` est le seul cas dont j'ai
personnellement vérifié, occurrence par occurrence, qu'il s'agit d'un
sur-appariement réel — il ne faut pas traiter les 353 autres comme
confirmés sans le même travail de vérification manuelle.

## Prochaines étapes suggérées (non commencées)

1. Vérifier occurrence par occurrence un échantillon des 354 candidats du
   relevé heuristique ci-dessus, pour distinguer vrais positifs (idiome
   court légitime) et sur-appariements (comme `know someone`) — c'est le
   travail que **Correction S2-1** du plan demande déjà :
   « modifier l'alignement `idiomatch` pour distinguer slots autorisés et
   mots lexicaux interposés... la surface complète et une signature
   syntaxique sont obligatoires ».
2. Lire `idiomatch/idiomatcher.py` (dépendance `.venv`) pour confirmer
   précisément comment le slot `someone` est actuellement satisfait, et
   si une contrainte de dépendance syntaxique (objet direct de `know`,
   pas seulement un pronom à proximité) est disponible dans l'API de la
   bibliothèque ou doit être ajoutée en post-filtrage côté
   `pipeline/mwe.py`.
3. Une fois la règle resserrée, rejouer `pipeline_out/mwe_candidates.jsonl`
   et vérifier que `know someone` retombe à un nombre d'occurrences
   plausible (l'idiome transitif réel, s'il existe dans le livre).
4. Ajouter un test négatif dédié (assertion que `know someone` ne capture
   plus `you know`/`I know` intransitifs) dans le corpus de régression
   Q0-2, à côté du cas déjà prévu `come to`.

## Contexte de découverte (pour mémoire)

Repéré en relançant `mwe_judge.run()` (S3) sur le livre complet avec
`catgpt/catgpt-browser` (variante de prompt `s3-occurrence-tags`, Lot U4
du chantier d'unification LLM — voir
`fix_pipeline/multi_models/report_u_unified_client.md`), lui-même relancé
pour débloquer S4 (`select.py`) en vue de retester S6-1
(`sense_fr_frontier.py`) sur un échantillon. Ce run S3 a été **stoppé**
avant d'aller au bout (~3-6 lots sur 32 traités) pour documenter ce bug
avant de continuer — voir la conversation de session pour le fil complet.
`pipeline_out/mwe_decisions.jsonl` et `pipeline_out/mwe_confirmed_spans.jsonl`
restent donc dans un état intermédiaire/périmé au moment de la rédaction
de ce rapport.
