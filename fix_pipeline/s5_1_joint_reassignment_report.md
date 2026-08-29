# S5-1 — réassignation conjointe lemme/POS/sens

## Avant

- S5 ne consultait que le couple lemme/POS principal de S1.
- `sense_fr_reassign` restait hors du pipeline standard et refusait les changements de POS.
- Une reprise pouvait réutiliser `senses.jsonl` après un changement de politique S5 si le digest S4 n'avait pas changé.

## Après

- S5 ouvre l'analyse principale puis toutes les alternatives S1 ayant un inventaire WordNet réel, score leurs sens dans le même contexte et n'accepte qu'un ID présent dans l'inventaire ouvert.
- Chaque ligne conserve l'analyse initiale, l'analyse retenue, sa source, la raison, le lemme résolu et les IDs autorisés.
- La surface exacte participe désormais au digest S4 ; une version de résolution participe au digest de reprise de chaque ligne S5.
- `sense_fr_reassign` appartient au chemin standard avant `export`, vérifie le digest consommé et conserve la provenance de ses réaffectations tardives.

## Vérifications

- Fixtures couvertes : `frosting`, `creeping`, `facilities`, `stressing`, `bitch`.
- Tests ciblés : 38/38 réussis avec les tests de tranches lors du premier passage ; après ajout du test d'orchestration, les 24 tests S5/S1/réassignation sans répertoire temporaire réussissent.
- Compilation Python et `git diff --check` réussis.
- Suite globale : 159 tests exécutés, 11 échecs attendus et 1 test ignoré ; quatre erreurs d'import préexistantes liées à l'encodage console/idiomatch. Les relances suivantes des tests utilisant `TemporaryDirectory` ont été bloquées par les ACL du bac à sable Windows, pas par une assertion du pipeline.

## Régressions et cas ouverts

- Aucune régression observée dans les tests ciblés.
- Aucun artefact de livre n'a été régénéré : cette livraison modifie le code, les contrats de digest et les tests, sans lancer un S5 complet coûteux.
