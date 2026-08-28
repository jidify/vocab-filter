# Point d'entrée des prompts de correction

Le catalogue complet et maintenu des prompts se trouve dans :

`./fix_pipeline/prompts_fixes_pipeline.md`

Après chaque `/clear`, ouvrir ce catalogue, sélectionner le prochain prompt dans l'ordre Q0 → S7, puis lire intégralement :

`./fix_pipeline/plan_action_fix_pipeline.md`

Règle de benchmark obligatoire : `pipeline_out/vocab_corrige.csv` a été corrigé à partir du périmètre de l'ancien `vocab.csv`. Il constitue une référence de qualité sur ce périmètre, pas un inventaire exhaustif du livre. Une unité absente du benchmark n'est donc pas automatiquement un faux positif. Tout véritable idiome, phrasal verb ou mot simple — notamment `latch` — doit être conservé s'il est attesté, correctement délimité, sémantiquement cohérent, bien traduit et pédagogiquement pertinent. Il doit être rapporté comme **amélioration hors périmètre**.

Le pipeline de production ne doit jamais lire le benchmark. L'évaluation doit distinguer :

1. correspondance au benchmark ;
2. variante acceptable ;
3. amélioration hors périmètre ;
4. révision nécessaire ;
5. véritable régression ou faux positif.

Le prompt sélectionné dans le catalogue complet est autosuffisant pour la correction concernée et ses critères de validation.
