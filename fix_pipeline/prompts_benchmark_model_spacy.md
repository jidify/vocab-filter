# Prompts — comparaison rapide des modèles spaCy

Deux prompts suffisent. Le second n'est lancé qu'après la courte revue humaine produite par le premier.

## Prompt 1 — exécuter et comparer

Lis `fix_pipeline/plan_benchmark_model_spacy.md`. Réalise le test rapide complet sans modifier le modèle ni les artefacts de production. Crée un exécuteur isolé, puis compare `en_core_web_sm`, `en_core_web_lg` et `en_core_web_trf` sur `The Humans - Stephen Karam.txt` et sur l'anglais extrait de `The Humans - Stephen Karam-TRAD.txt`. Ne lance que spaCy et `pipeline.multi_token`, jamais GlossBERT ou un LLM. Si un modèle manque, demande l'autorisation avant de l'installer. Vérifie tous les offsets, mesure les temps et écris uniquement sous `pipeline_out/spacy_quick_compare/`. Produis `results.json`, `disagreements.jsonl` et un `report.md` donnant le résultat des six expressions, les candidats propres à chaque modèle, les différences de bornes et les différences entre sources. Prépare une liste compacte de 30 désaccords maximum et 10 accords à relire manuellement. Arrête un run `trf` manifestement trop lent ou en manque de mémoire au lieu de transformer ce test en chantier.

## Prompt 2 — conclure après revue rapide

Lis le plan, le rapport du Prompt 1 et les réponses humaines ajoutées aux 30 désaccords maximum et aux 10 contrôles. Calcule seulement les comptes utiles : corrects, incorrects, bornes incorrectes et incertains par modèle, séparés entre NER et `compound`. Ajoute au rapport les améliorations/régressions concrètes, les temps et une conclusion prudente : conserver `sm`, approfondir `lg`, approfondir `trf` ou envisager une cascade. Ne fais ni bootstrap, ni interface d'annotation, ni changement du pipeline de production.
