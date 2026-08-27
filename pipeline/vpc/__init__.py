"""Détection de verb-particle constructions (VPC / phrasal verbs) — Lot 2.

Ce sous-paquet est vendoré depuis `book-lexical-analyzer`
(https://github.com/jdantan — dépôt local `C:\\DOCS\\_perso\\book-lexical-analyzer`,
commit `5bd3f5fce03d4b467f0c8431d0aa91c154cc2841`, module `booklex`, version `0.1.0`),
même auteur, projet sœur. Fichiers portés tels quels (adaptations mineures documentées
en tête de chaque module) :

- `detectors/phrasal_verbs.py` (booklex/detectors/phrasal_verbs.py)
- `domain/phrasal_verbs.py`    (booklex/domain/phrasal_verbs.py)
- `domain/vpc_frames.py`       (booklex/domain/vpc_frames.py)
- `domain/resources.py`        (booklex/domain/resources.py)
- `resources/vpc_reference.py` (booklex/resources/vpc_reference.py)

Volontairement NON portés (voir le plan, Partie 4 "Lot 2") : `domain/models.py`
(le type `Document` — remplacé ici par `pipeline/vpc/adapter.py`, qui projette un
`Doc` spaCy directement, sans passer par le pipeline d'ingestion de booklex),
`analysis/` et `evaluation/` (hors périmètre : agrégation/rapport, pas détection).

Les ressources gelées associées (`data/vpc/*.json`) proviennent de
`resources/vpc/*.json` du même projet et portent l'annotation VMWE PARSEME
(licence CC-BY-4.0 pour les annotations ; voir `data/vpc/manifest.json` et, côté
projet source, `resources/manifests/vpc-phase2.8.json` /
`resources/manifests/parseme-en-1.3.json` pour l'attribution complète).

`domain/ports.py` de ce sous-paquet n'est PAS un portage : ce sont des Protocol
minimaux réécrits ici pour éviter de tirer `domain/models.py`/`domain/vmwe.py`
(non nécessaires à la détection seule). `resources/wordnet_nltk.py` n'est pas non
plus un portage : c'est un fournisseur WordNet basé sur `nltk` (déjà une
dépendance du projet), remplaçant le fournisseur `wn`/WNDB du projet source
(voir le plan, point 11 — pas de nouvel index à maintenir).
"""

VENDORED_SOURCE_VERSION = "0.1.0"
