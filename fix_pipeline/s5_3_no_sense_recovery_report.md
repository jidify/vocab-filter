# S5-3 — `aucun_sens_adapte` sans suppression

## Baseline

`pipeline.score.build_records()` contenait deux sorties destructrices : une
ligne dont `best_sense == "aucun_sens_adapte"` était ignorée, puis toute ligne
dont le sens retenu n'était pas présent dans `candidates` était également
ignorée. Les trois occurrences réelles de `latch` (segments 1625, 1630 et
1646) existent dans `senses.jsonl`, mais étaient absentes de `vocab.csv` et de
`review_queue.csv`.

## Politique livrée

L'ordre de récupération est explicite et conservé dans `recovery.attempts` :

1. inventaires alternatifs lemme/POS ouverts par la résolution conjointe S5-1 ;
2. candidat MWE/composé couvrant, routé en révision sans exporter le fragment
   comme sens WordNet certain ;
3. sens custom stable uniquement avec définition anglaise, indice textuel et
   confiance d'arbitre d'au moins 0,85 ;
4. identité `unresolved.*` stable et file de révision.

Les anciens artefacts restent lisibles : `build_records()` matérialise une
identité de révision pour leurs lignes `aucun_sens_adapte`, sans les modifier.
La review queue publie route, raison, candidats et action attendue.

## Gates

- `latch` conserve `latch.n.01` et `latch.n.02`, le contexte et les trois
  occurrences à réviser ;
- l'invariant d'export vérifie chaque clé incertaine et son nombre
  d'occurrences avant toute écriture ;
- le test d'intégration écrit puis relit `vocab.jsonl`, `vocab.csv` et
  `review_queue.csv` ;
- 38 tests S4/S5/S6 ciblés passent ; les 32 tests S5/Q0-2 passent aussi
  (un mode réel est ignoré et onze défauts Q0-2 antérieurs restent marqués
  comme échecs attendus).

Le rejeu des artefacts de production n'a pas été forcé : le garde-fou a détecté
que `senses.inventory.sha256` ne correspond pas à l'inventaire S4 courant. Il
faut reprendre depuis `select`/`senses` pour publier un nouvel export réel sans
mélanger deux générations.
