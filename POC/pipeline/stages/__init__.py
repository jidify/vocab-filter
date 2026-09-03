"""Scripts d'étape du pipeline POC — extract/translate mots, extract/translate
MWE, merge, localize. Invoqués par build_vocabulary_to_learn_pipeline.py
chacun dans son propre sous-processus (jamais importés directement, voir sa
docstring) — ce fichier ne fait que marquer le répertoire comme module
Python, au cas où l'un de ces scripts serait un jour importé plutôt que
lancé en sous-processus.
"""
