# TODO — intégrer le fix de tokenizer "tiret collé après ponctuation fermante"

**Statut : intégré côté POC (`POC/`), PAS en production (`pipeline/`).**
Voir le plan "Stage 0 du pipeline POC — configuration unique du tokenizer" :
`POC/poc_pipeline/tokenizer_setup.py::configure_tokenizer` applique
`patch_dash_after_punctuation` aux trois tokenizers du POC (analyze.get_nlp,
extract_word_contexts.py, et — nouveau — `matcher.nlp` d'idiomatch), avant
tout parsing. Le blocage n°2 ci-dessous (accès au `nlp` interne d'idiomatch)
est donc levé côté POC : `matcher.nlp` était déjà exposé en clair dans
`poc_pipeline/mwe.py`.

`pipeline/analyze.py` (PRODUCTION) n'a toujours PAS été touché — les risques
1 (inventaire figé) et 3 (re-run S1→S6) ci-dessous restent entiers pour la
prod, ils ne s'appliquent pas au POC (pas d'inventaire figé, pas de
`data/sense_fr.jsonl` à préserver). Prototypé et mesuré dans
`fix_pipeline/detection_benchmark/tokenizer_boundary_fix.py` +
`phase_tokenizer_fix_probe.py`/`_report.md` (probe hors plan, après Phase 6
de `fix_pipeline/plan_detection_benchmark_funnel.md`) — voir "Pourquoi pas
maintenant" ci-dessous pour la production.

## La cause

`en_core_web_sm` (règle d'infixe par défaut de spaCy) ne scinde un tiret
cadratin/demi-cadratin qu'ENTRE deux caractères alphabétiques. Un tiret
collé sans espace à une ponctuation fermante précédente (`?!.,;:'")]`,
guillemets courbes) n'est jamais scindé :

```
'around—we'   -> ['around', '—', 'we']     (correct)
'around?—we'  -> ['around?—we']            (un seul token-poubelle)
```

Scan sur les 2535 segments de *The Humans* (script réutilisable,
`tokenizer_boundary_fix.scan_suspect_tokens` — détecte la SIGNATURE du bug,
pas une liste de mots figée) : **30 tokens suspects sur 30327** avant
patch, tous cette même famille. Deux motifs d'infixe ajoutés (jamais
retirés) corrigent **29/30** sans aucune régression mesurée sur les traits
d'union légitimes (`ground-floor`, `e-mail`, `and/or`, `smart-ass`...).
Détail complet, chiffres, et effet mesuré sur le benchmark de détection
(Phase 2/3, rejoués avec ce fix) : `fix_pipeline/detection_benchmark/
phase_tokenizer_fix_probe_report.md`.

Conclusion du probe : le fix est réel et sans coût en soi, mais son effet
mesuré sur le rappel MWE (99 segments gold) est nul — seul un gain de
précision (-1 `hard_negative` capturé à tort) a été observé. Ce n'est donc
PAS un levier de rappel démontré, juste une correction de bug de tokenizer
qui vaut la peine pour elle-même.

## Pourquoi pas maintenant (risques identifiés, non mesurés)

1. **Invalidation de l'inventaire figé.** `pipeline/inventory.py` (
   `INVENTORY_HASH_PATH`/`LEXICAL_INVENTORY_PATH`) verrouille S5/S6
   (`senses.jsonl`, `data/sense_fr.jsonl`, exports) contre tout changement
   de S1-S4 (`verify_consumer`). Changer la tokenisation change
   `occurrences.jsonl` (S1) — pas un patch "à côté", ça invalide la chaîne
   aval et exige un re-run complet S1→S6, pas mesuré ici (combien
   d'entrées `sense_fr.jsonl` seraient concernées, combien de temps/coût
   LLM pour les regénérer).
2. **Deux tokenizers spaCy actifs, un seul patché.** `pipeline/mwe.py::
   get_matcher()` construit son PROPRE `nlp` interne via
   `Idiomatcher.from_pretrained()`, indépendant de
   `pipeline.analyze.get_nlp()`. Patcher seulement `analyze.get_nlp()`
   laisserait idiomatch tokeniser le même texte différemment de
   `analyze.py`/`pipeline.vpc` — une incohérence architecturale à
   trancher (patcher aussi le `nlp` interne d'idiomatch ? Vérifier que
   `mwe_alignment.py` reste correct dans les deux cas ?), pas juste un
   copier-coller du patch.
3. **Effet de reparsing en cascade, pas isolé au tiret.** Corriger la
   tokenisation d'un token change l'ANALYSE DE DÉPENDANCES DE TOUTE LA
   PHRASE autour (vu concrètement sur seg1277 : "Mary statue" devient un
   composé nominal correctement détecté une fois la phrase reparsée — un
   effet positif ici, mais les ~30 phrases touchées n'ont pas été
   relues une par une pour vérifier qu'aucune n'introduit une régression
   NER/compound ailleurs).
4. **Généralité non validée au-delà d'un seul livre.** Le mécanisme
   (classes de ponctuation, pas liste de mots) est générique par
   construction, mais n'a été vérifié QUE sur *The Humans* — à revalider
   (au moins relancer `scan_suspect_tokens`) sur le prochain livre traité
   avant de supposer qu'il suffit.

## Plan d'intégration à étudier (pas encore fait, PRODUCTION seulement)

1. ~~Décider du sort du `nlp` interne d'idiomatch (risque 2)~~ — résolu côté
   POC : `Idiomatcher.from_pretrained().nlp` est accessible directement
   (`poc_pipeline/mwe.py::get_matcher`, déjà utilisé en clair ailleurs dans
   ce module) et reçoit désormais le patch. Reste à faire le même geste côté
   `pipeline/mwe.py` si ce TODO est un jour porté en production.
2. Ajouter `tokenizer_boundary_fix.patch_dash_after_punctuation` (ou son
   équivalent promu depuis `fix_pipeline/` vers `pipeline/`) dans
   `pipeline/analyze.py::get_nlp()`, à côté d'`EMAIL_SPECIAL_CASES`.
3. Re-run S1 complet, diff `occurrences.jsonl`/`multi_token_candidates.
   jsonl`/`vpc_candidates.jsonl` contre les artefacts de production
   actuels — cataloguer explicitement chaque phrase changée (~30 attendues,
   à confirmer) et relire chacune (risque 3).
4. Chiffrer le coût réel du re-run S1→S6 (risque 1) : combien d'entrées
   `data/sense_fr.jsonl` sont réellement affectées (probablement une toute
   petite fraction, les offsets/lemmes des ~2500 segments non touchés ne
   changent pas) avant de décider si un re-run complet ou une migration
   ciblée est plus raisonnable.
5. Relancer `fix_pipeline/detection_benchmark/phase2_run_baselines.py` et
   `phase3_run_rules_plus.py` (pas juste le probe) pour confirmer que rien
   ne régresse une fois le patch réellement en place, avant de le
   considérer terminé.
