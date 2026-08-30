# TODO — la phase GlossBERT de S5 doit tourner sur une machine avec GPU

**Statut : constaté, pas traité.** Observé le 2026-08-30 en suivant en
direct un run S1→S5 sur *The Humans* (petit livre, une seule pièce de
théâtre).

## Le constat

`pipeline/senses.py::run()` désambiguïse chaque occurrence de mot simple
via GlossBERT (`get_gloss_model()` — BERT, `pipeline/senses.py:64-85`).
Sur la machine où ce run a tourné (pas de GPU utilisé, calcul CPU pur) :

- **6869 occurrences** de mots simples à traiter pour *The Humans* (un
  livre court, une pièce de théâtre en un acte).
- Débit observé, stable sur toute la durée du run : **~0.4-0.5
  occurrence/seconde**.
- Temps total projeté pour cette seule phase : **environ 4h30** (272 min),
  avant même la sélection/traduction/export en aval.

Un commentaire déjà présent dans le code (`senses.py:64-72`) documente le
même problème : l'auto-attention de BERT est en O(L²), et sans le patch de
troncature à 192 tokens (au lieu des 512 par défaut de la librairie
`glossbert`), le run était encore ~7x plus lent (ETA ~235 min mesurée sur
2922 occurrences, un périmètre plus petit que celui-ci). Le patch déjà en
place aide, mais le calcul reste fondamentalement CPU-bound.

## Pourquoi c'est bloquant à terme

Le plan (`fix_pipeline/plan_action_fix_pipeline.md`, section "Contraintes
de conception confirmées") affirme que **la machine de production dédiée
dispose d'une RTX 5090** et que le GPU est "une ressource normale de
production, pas une hypothèse optionnelle". Mais ce run n'en a visiblement
pas profité pour GlossBERT — sur un livre déjà court, la phase a pris
plusieurs heures. Sur un livre nettement plus long (roman complet, etc.),
ce temps grandirait proportionnellement au nombre d'occurrences de mots
simples, ce qui rendrait un cycle correction→run complet du pipeline
impraticable pour itérer (le plan §8 "Politique de livraison" demande
justement de régénérer et comparer à chaque lot).

## Précision importante — la librairie gère déjà le GPU

Vérifié directement : `glossbert.GlossBERT.__init__` a la signature
`(self, model='kanishka/GlossBERT', cuda=True)` — le paramètre existe et
**vaut déjà `True` par défaut**. `pipeline/senses.py::get_gloss_model()`
appelle `GlossBERT()` sans argument (`senses.py:78`), donc rien côté code
applicatif n'empêche le GPU d'être utilisé.

Le run observé (4h30 projetées, débit ~0.4-0.5/s constant) tournait donc
soit sans GPU CUDA visible sur la machine, soit sur une machine sans GPU
du tout — deux tentatives de vérification pendant ce run (`nvidia-smi`,
absent du PATH ; compteur de performance Windows `\GPU Engine(*)`,
indisponible) n'ont trouvé aucune trace de GPU actif sur CETTE machine.
Cela concorde avec l'hypothèse que cette session ne tournait pas sur "la
machine de production dédiée" à RTX 5090 mentionnée dans le plan
(`fix_pipeline/plan_action_fix_pipeline.md`) mais sur une autre machine
(poste de développement sans GPU dédié, ou GPU non exposé/driver absent).

## Pas encore fait

1. Confirmer sur la VRAIE machine de production (RTX 5090) que
   `torch.cuda.is_available()` renvoie `True` et que `GlossBERT(cuda=True)`
   utilise effectivement le GPU (device du modèle chargé, utilisation
   GPU mesurée pendant un run réel — `nvidia-smi` ou équivalent).
2. Si le GPU n'est PAS détecté même sur la machine de production (driver
   manquant, mauvaise build torch installée sans support CUDA...),
   diagnostiquer et corriger l'environnement plutôt que le code — le
   paramètre applicatif est déjà correct.
3. Mesurer le gain réel GPU vs CPU sur un sous-ensemble comparable une
   fois confirmé que le GPU est bien sollicité, pour objectiver le facteur
   d'accélération attendu (le patch de troncature de contexte à 192
   tokens, lui, avait déjà été mesuré — voir le commentaire cité
   plus haut — la question GPU ne l'a pas encore été).
4. Documenter dans le README/plan que lancer S5 sur une machine sans GPU
   CUDA disponible est explicitement déconseillé/à éviter pour tout livre
   au-delà d'un format très court, une fois la mesure faite.
