# Prompts autosuffisants pour corriger le pipeline après `/clear`

## Mode d'emploi

Exécuter **un seul prompt à la fois**, dans l'ordre. Après validation et commit éventuel du lot, faire `/clear`, puis utiliser le prompt suivant. Chaque prompt impose de lire d'abord le plan global :

`./fix_pipeline/plan_action_fix_pipeline.md`

Règles communes à tous les prompts :

- travailler dans `C:\DOCS\_perso\vocab-filter` ;
- respecter les changements déjà présents et ne pas annuler le travail utilisateur ;
- considérer `pipeline_out/vocab_corrige.csv` comme benchmark en lecture seule, jamais comme entrée de production ; ce benchmark corrige le périmètre de `vocab.csv` initial et n'est pas un inventaire exhaustif du livre ;
- ne jamais considérer automatiquement une unité absente du benchmark comme un faux positif : conserver tout véritable idiome, phrasal verb ou mot simple lexicalement, contextuellement et pédagogiquement validé ; classer ces unités comme améliorations hors périmètre, avec `latch` comme cas témoin obligatoire ;
- ne pas coder d'exception spécifique à *The Humans* ou à une ligne du benchmark ;
- commencer par des tests reproduisant le défaut, puis implémenter et mesurer avant/après ;
- ne pas annoncer la correction terminée si les résultats/vérifications de la section du plan ne sont pas tous satisfaits ;
- mettre à jour les artefacts, tests et documentation directement nécessaires au lot ;
- terminer par les commandes exécutées, métriques avant/après, régressions éventuelles et prochain gate débloqué.

---

## Prompt Q0-1 — évaluateur de référence

Lis intégralement `./fix_pipeline/plan_action_fix_pipeline.md`, en particulier **§0, la Politique d'évaluation des écarts hors benchmark, Correction Q0-1** et **§8**. Implémente l'évaluateur reproductible comparant `pipeline_out/vocab.csv` à `pipeline_out/vocab_corrige.csv`. Il doit gérer les homonymes et produire `pipeline_out/fix_quality_metrics.json` et `pipeline_out/fix_quality_report.md` avec métriques par dimension et cas nommés. Le benchmark ayant été construit depuis le périmètre initial de `vocab.csv`, classe séparément correspondances, variantes acceptables, améliorations hors périmètre, révisions et vrais faux positifs ; ne pénalise jamais automatiquement une absence du benchmark. `latch` doit être reconnu comme récupération attendue. Le benchmark doit rester strictement en lecture seule et ne doit être importé par aucun module de production. Ajoute des tests du comparateur, exécute-le, explique tout écart avec la baseline documentée et ne poursuis pas vers Q0-2.

## Prompt Q0-2 — corpus de régression stratifié

Lis le plan global, surtout **§0, la Politique d'évaluation des écarts hors benchmark, Correction Q0-2**, la baseline Q0-1 et **§8**. Construis le corpus/tests de régression stratifié couvrant MWE fusionnées/manquées/polysémiques, POS/lemme, sens WordNet, composés/entités, `aucun_sens_adapte`, transparence, pending et améliorations valides hors périmètre. Inclue `latch` et au moins un véritable idiome/phrasal verb supplémentaire. Les tests LLM ordinaires doivent être déterministes hors réseau ; sépare clairement l'évaluation réelle. Prouve que chaque défaut connu échoue actuellement pour la raison attendue, sans implémenter encore les corrections S1.

## Prompt S1-1 — alternatives de lemme et POS

Lis le plan global, surtout **§1, Correction S1-1**, Q0 et **§8**. Fais évoluer S1 et le schéma d'occurrence pour préserver analyse principale et alternatives morphosyntaxiques. Couvre au minimum `frosting`, `creeping`, `facilities`, `stressing`, `bitch`. Assure la compatibilité/migration des consommateurs et digests. Valide offsets, schéma et tests ; mesure avec Q0-1, sans modifier encore la sélection de sens S5.

## Prompt S1-2 — composés et entités

Lis le plan global, surtout **§1, Correction S1-2**, S1-1 et **§8**. Ajoute un artefact ou des champs de candidats multi-tokens pour `New York`, `Virgin Mary`, `ranch dip`, `observation deck`, `nursing home`, `crystal ball`, avec spans, score et provenance. Ne réserve ni ne supprime encore les tokens. Ajoute tests positifs/négatifs, vérifie les offsets et montre que les consommateurs aval peuvent lire ces hypothèses.

## Prompt S1-3 — candidats VPC enrichis

Lis le plan global, surtout **§1, Correction S1-3**, le contrat VPC existant et **§8**. Enrichis les candidats VPC et leurs traits contextuels sans traiter l'absence PARSEME comme rejet sémantique. Valide les quatre occurrences de `burn out` et le routage des MWE manquantes mentionnées. Préserve provenance et spans exacts. Exécute `test_vpc_contract.py` et les nouvelles régressions ; ne juge pas encore les sens.

## Prompt S2-1 — fin des canons MWE sur-génériques

Lis le plan global, surtout **§2, Correction S2-1**, les sorties S1 et **§8**. Corrige le matching/alignement afin que les mots lexicaux interposés ne soient pas avalés par des slots génériques. Ajoute les assertions négatives `come to`, `look to`, `go for`, `get better`, `back up` et les assertions positives exactes. Régénère S2, vérifie `mwe_candidates.jsonl` et quantifie précision/rappel avec Q0-1.

## Prompt S2-2 — rappel des MWE

Lis le plan global, surtout **§0 Politique d'évaluation des écarts hors benchmark**, **§2, Correction S2-2**, S2-1 et **§8**. Augmente le rappel par sources complémentaires et patrons généralisables. Fais apparaître comme candidats exacts les familles `let it go`, `come back to earth`, `get worked up`, `at ease`, `burn out`, `put to rest`, `steer clear of`, `could care less`, `tighten one's belt`. Chaque ajout doit avoir provenance, offsets, test positif et contre-exemple. Autorise et conserve les véritables MWE supplémentaires absentes du benchmark ; rapporte-les comme améliorations hors périmètre après validation indépendante. Mesure le gain de rappel et la variation de précision avant de conclure.

## Prompt S2-3 — schéma d'hypothèses MWE

Lis le plan global, surtout **§2, Correction S2-3**, les schémas S1/S2 et **§8**. Sépare forme candidate, surface, spans membres/complet, signature syntaxique et sources. Permets plusieurs hypothèses concurrentes sur le même span sans écrasement. Migre les consommateurs et ajoute un test explicite sur `let it go`. Ne décide pas encore du sens.

## Prompt S3-1 — jugement par occurrence

Lis le plan global, surtout **§3, Correction S3-1**, Q0-2, S2-3 et **§8**. Remplace le jugement global sur trois exemples par le protocole occurrence-first prévu. Traite toutes les sources, pas seulement les VPC directionnels. Produis lexicalité, canon proposé, POS, paraphrase et signaux de confiance. Valide `let it go`/`let him go`/`let's go`, `burn out`, `could care less` et les faux MWE. Compare modèle local et fixtures déterministes ; ne regroupe pas encore les sens.

## Prompt S3-2 — clustering et identité de sens MWE

Lis le plan global, surtout **§3, Correction S3-2**, S3-1 et **§8**. Implémente le regroupement après jugement sur canon+POS+sens compatible, avec ID DBnary/WordNet exact ou ID custom stable. Sépare les sens de `burn out`, `let ... go` et `come back to earth`. Interdis les catégories `idiome/phrasal_verb/semi_fige` comme `sense_id`. Ajoute invariants d'identité, tests et métriques sur les 98 POS/126 IDs divergents.

## Prompt S3-3 — définition contextuelle des MWE

Lis le plan global, surtout **§3, Correction S3-3**, S3-2 et **§8**. Supprime le choix automatique du premier sens d'`idioms.yml`. Sélectionne une définition exacte par cluster ou crée une définition custom justifiée. Couvre les exemples `break up`, `bring up`, `check in`, `get a grip`, `give out`, `keep up`, `look after`, `turn off`, `work out`. Ajoute un contrôle de contradiction définition/contexte et mesure les 38 divergences du benchmark.

## Prompt S3-4 — cache et choix du juge

Lis le plan global, surtout **§3, Correction S3-4**, Q0-2 et **§8**. Versionne les clés de cache par protocole/modèle/schéma/contexte, migre ou invalide proprement les anciennes décisions incompatibles et empêche le cache de figer une panne. Évalue le modèle local et un modèle frontière autorisé sur les mêmes strates ; définis l'escalade du résidu difficile selon les mesures, pas selon l'intuition. Vérifie explicitement que l'ancien verdict de `could care less` n'est pas réutilisé.

## Prompt S4-1 — inventaire unifié par sens

Lis le plan global, surtout **§4, Correction S4-1**, les sorties S3 et **§8**. Unifie mots/MWE autour de `(canon, POS, sense_id)` et agrège seulement les surfaces de ce sens. Migre `selected_mwe.jsonl`, `lexical_inventory.jsonl`, digests et consommateurs. Valide les clés distinctes de `burn out` et l'absence de regroupement `come to`. Exécute les tests de tranches/inventaire concernés.

## Prompt S4-2 — réservation exacte et réversible

Lis le plan global, surtout **§4, Correction S4-2**, S4-1 et **§8**. Garantit que seuls les spans d'occurrences confirmées sont réservés, que les MWE rejetées rendent leurs tokens et que les hypothèses incertaines ne suppriment rien. Teste spans discontinus, hypothèses concurrentes, `latch`, ainsi que les composés `New York` et `Virgin Mary`. Fournis un bilan comptable avant/après réservation.

## Prompt S4-3 — différer le filtre pédagogique

Lis le plan global, surtout **§4, Correction S4-3**, la porte S4 actuelle et **§8**. Transforme les signaux CEFR/prévalence en métadonnées lorsque la décision dépend du sens ; conserve uniquement les exclusions certaines à S4. Ajoute un test lemme banal/sens rare et vérifie qu'aucun sens potentiellement difficile n'est perdu avant S5.

## Prompt S5-1 — réassignation conjointe dans le pipeline standard

Lis le plan global, surtout **§5, Correction S5-1**, S1-1, l'actuel `sense_fr_reassign.py`, `run_pipeline.py` et **§8**. Intègre une résolution conjointe lemme/POS/sens au chemin standard avec inventaire ouvert contrôlé. Corrige les fixtures `frosting`, `creeping`, `facilities`, `stressing`, `bitch`, conserve provenance et interdit les IDs inventés. Mets à jour l'orchestration, les digests et tests de reprise.

## Prompt S5-2 — arbitrage de sens calibré

Lis le plan global, surtout **§5, Correction S5-2**, Q0-2 et **§8**. Remplace le seuil de marge 0,15 comme critère principal par une politique évaluée. Couvre localisation, POS, bilingue, entropie, entités/composés et désaccords. Valide `facility`, `plow`, `haggard`, `spa`, `barely`, `poke`, `roll`, `touch`, publie matrice de confusion et calibration, et conserve `verify_senses_regression.py` vert.

## Prompt S5-3 — aucun_sens_adapte sans suppression

Lis le plan global, surtout **§5, Correction S5-3**, les branches de récupération et **§8**. Supprime le `continue` destructeur de l'export et implémente l'ordre autre POS/lemme → MWE/composé → custom justifié → révision. Utilise `latch` comme gate bloquant. Ajoute l'invariant qu'aucune occurrence incertaine ne disparaît et vérifie tous les artefacts aval jusqu'à la review queue.

## Prompt S5-4 — fragments d'entités/composés

Lis le plan global, surtout **§5, Correction S5-4**, S1-2/S4-2 et **§8**. Branche les spans de composés/entités dans la résolution afin d'empêcher les faux sens simples. Corrige les contextes `New York`, `Virgin Mary`, `ranch dip`, `observation deck`, `nursing home`, `crystal ball` sans supprimer une occurrence autonome éventuelle. Ajoute tests occurrence-scoped et mesure les 9 mots retirés par le benchmark.

## Prompt S6-1 — traduction d'un sens verrouillé

Lis le plan global, surtout **§6, Correction S6-1**, les schémas S3/S5 et **§8**. Fais consommer à S6 une identité complète et des occurrences homogènes. Ajoute un contrôle bloquant de cohérence sens–définition–FR. Valide `look after`, `give out`, `turn off`, `work out` et les autres définitions corrigées. Ne verrouille jamais automatiquement un mismatch/doubtful.

## Prompt S6-2 — résidu pending

Lis le plan global, surtout **§6, Correction S6-2**, les statuts du magasin FR et **§8**. Résous par preuves indépendantes ou route en révision toutes les traductions pending. Le CSV final ne doit jamais contenir de traduction officielle vide. Couvre `fit`, `watering`, `flush`, `overtone`, `settle`, `sound`; mesure les 99 blancs initiaux et démontre leur destination exacte.

## Prompt S6-3 — métrique de traduction sans fuite

Lis le plan global, surtout **§6, Correction S6-3**, Q0-1 et **§8**. Ajoute l'évaluation souple des traductions après génération, en distinguant synonymie acceptable et contresens. Produis résultats par statut/source/modèle, avec échantillon auditable. Vérifie que le benchmark n'est importé par aucun module de production.

## Prompt S7-1 — filtre pédagogique par sens

Lis le plan global, surtout **§0 Politique d'évaluation des écarts hors benchmark**, **§7, Correction S7-1**, les métriques Q0 et **§8**. Implémente une porte finale combinant transparence, faux amis, fréquence/CEFR, surprise du sens et intérêt MWE. Utilise `affection`, `intelligible`, `sensible`, `latch` et les 53 identités exactes comme fixtures contrastives. L'absence du benchmark ne doit jouer aucun rôle dans la décision de production : conserve les ajouts authentiques et rapporte-les comme améliorations hors périmètre. Chaque exclusion doit avoir une raison ; mesure précision/rappel pédagogique sur le périmètre couvert et précision auditée des ajouts.

## Prompt S7-2 — export cohérent

Lis le plan global, surtout **§7, Correction S7-2**, le schéma S4 et **§8**. Fais respecter une ligne complète par unité de sens, sans concaténation inter-sens, POS MWE vide, catégorie en guise d'ID ni FR vide. Valide `come to`, les sens de `burn out`, `let it go` et les contraintes de schéma. Exécute l'export complet et Q0-1.

## Prompt S7-3 — file de révision exhaustive

Lis le plan global, surtout **§7, Correction S7-3**, toutes les branches d'incertitude et **§8**. Unifie les révisions détection/POS/sens/traduction/éligibilité avec raison, candidats, contexte et action. Implémente et teste l'invariant `candidats = exportés + exclus avec raison + révision`. Vérifie `latch`, pending, `aucun_sens_adapte` et conflits de cluster.

## Prompt S7-4 — validation finale

Lis intégralement le plan global, surtout **§0 Politique d'évaluation des écarts hors benchmark**, **§7, Correction S7-4**, la liste complète des gates et **§8**. Exécute le pipeline de bout en bout, tous les tests, l'évaluateur Q0-1 et une non-régression sur un autre texte. Ne masque aucun écart par une exception livre-spécifique et ne force pas le résultat à rester dans le périmètre incomplet du benchmark. Fournis le rapport final : métriques sur le périmètre de `vocab_corrige.csv`, liste et précision auditée des améliorations hors périmètre (idiomes, phrasal verbs et mots simples comme `latch`), validation de chaque gate nommé, variantes acceptables, résidu humain et preuve qu'il n'existe plus de défaut critique connu.
