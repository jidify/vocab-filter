"""Script one-shot — élague les clés MWE HÉRITÉES de `data/sense_fr.jsonl`
(plan §6, Correction S6-1 : "imposer une entrée cohérente au traducteur").

Contexte : avant S3-2 (regroupement par sens, pas par catégorie), une MWE
était clée `mwe:<canon>:<phrasal_verb|idiome|semi_fige>` — la CATÉGORIE
grammaticale-lexicale servait de `sense_id`, jamais un vrai sens. Depuis
S3-2/S4-1, `pipeline.score.build_mwe_units()`/`inventory.make_unit_key`
produisent `mwe:<canon>:<pos>:<sense_id>` — une clé par SENS distinct d'un
même canon (`mwe:burn out:verb:...` pour "s'épuiser" ≠ celle pour
"s'éteindre"). Recoupement mesuré entre les 174 clés héritées du magasin et
les ~497 unités actuelles de `pipeline_out/selected_mwe.jsonl` : 0 — aucune
clé héritée n'a de correspondance dans le nouveau format. Certaines de ces
clés héritées portent une `definition_en` PÉRIMÉE (mesurée AVANT ce script :
`give out`/`turn off`/`bring up`/`check in`/`get a grip`/`keep up` — voir
git log "Block automatic locking..."), donc les CONSERVER ne fait que
perpétuer une identité que S6 ne consomme plus correctement.

Deux issues, JAMAIS une suppression aveugle (plan §7-3 : "aucune disparition
silencieuse") :

- **ÉLISION** : une clé héritée dont le `canonical_form` a AU MOINS UNE
  correspondance dans `pipeline_out/selected_mwe.jsonl` (au format actuel)
  est supprimée du magasin. `pipeline.sense_fr_frontier.collect_frontier_targets()`
  la retraduira au prochain run, sous sa/ses nouvelle(s) clé(s) — y compris
  une éventuelle décision `validated` humaine héritée (ex. `give out`,
  `look after`, `turn off`), qui devient alors le TÉMOIN direct du nouveau
  contrôle bloquant sens-définition-FR (pipeline/verify_sense_coherence.py) :
  si la porte fonctionne, le modèle frontière retrouve seul un triplet
  cohérent, sans l'aide de la correction humaine.
- **CONSERVATION** : une clé héritée SANS correspondance actuelle (ex.
  "beat", "pig smash", "smart ass", "lame", "hail mary" — toutes issues
  d'une correction manuelle ponctuelle sur une occurrence MOT, voir
  data/manual_corrections.jsonl, jamais d'une détection MWE S2/S3/S4)
  n'a AUCUN chemin de régénération automatique : "beat"/"pig smash" sont
  d'ailleurs bloquées `pending`/`sense_id_non_resolu` avant le correctif de
  `pipeline.sense_fr.collect_targets()` qui les classait à tort `kind:
  "synset"`. Les supprimer serait une perte de travail humain IRRÉVERSIBLE
  (statut `validated` pour 3 d'entre elles) sans aucun moyen de les
  reconstruire — laissées intactes.

`data/sense_fr.lock.json` est mis à jour dans le MÊME geste : seules les
clés ÉLIDÉES en sont retirées (jamais recalculé en entier — un recalcul
prendrait l'état COURANT comme nouvelle référence, ce qui annule la
détection de régression pour tout le reste du magasin, cf.
pipeline/verify_fr_lock.py).

Usage :
    uv run python -m tools.migrate_sense_fr_mwe_keys --dry-run
    uv run python -m tools.migrate_sense_fr_mwe_keys
"""

from __future__ import annotations

import argparse
import json

from pipeline import config, sense_fr, verify_fr_lock

LEGACY_CATEGORIES = {"phrasal_verb", "idiome", "semi_fige"}


def is_legacy_mwe_key(key: str) -> bool:
    """`mwe:<canon>:<catégorie>` — exactement 3 segments deux-points, le
    dernier une des 3 catégories héritées. Le format actuel a 4 segments
    (`mwe:<canon>:<pos>:<sense_id>`) et ne matche donc jamais ceci."""
    if not key.startswith("mwe:"):
        return False
    parts = key.split(":")
    return len(parts) == 3 and parts[2] in LEGACY_CATEGORIES


def current_mwe_canonical_forms() -> set[str]:
    """Canons (casefold) couverts par le format ACTUEL — ce que
    `pipeline.sense_fr.collect_targets()` regénérera au prochain run."""
    if not config.SELECTED_MWE_PATH.exists():
        return set()
    forms = set()
    with config.SELECTED_MWE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            forms.add(json.loads(line)["canonical_form"].casefold())
    return forms


def plan_migration(store: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Renvoie (a_elider, a_conserver) — clés héritées triées."""
    regenerable = current_mwe_canonical_forms()
    elide, keep = [], []
    for key, entry in store.items():
        if not is_legacy_mwe_key(key):
            continue
        canon = (entry.get("lemmas_en") or [None])[0]
        if canon is not None and canon.casefold() in regenerable:
            elide.append(key)
        else:
            keep.append(key)
    return sorted(elide), sorted(keep)


def run(dry_run: bool = False) -> int:
    store = sense_fr.load_store()
    elide, keep = plan_migration(store)

    print(f"{len(elide) + len(keep)} clé(s) MWE héritée(s) trouvée(s) dans {config.SENSE_FR_STORE_PATH.name}.")
    print(f"  {len(elide)} à ÉLIDER (régénérables — un unit_key actuel existe pour ce canon) :")
    for key in elide:
        entry = store[key]
        print(f"    - {key} (status={entry.get('status')}, fr={entry.get('fr')!r})")
    print(f"  {len(keep)} à CONSERVER (aucune régénération automatique possible) :")
    for key in keep:
        entry = store[key]
        print(f"    - {key} (status={entry.get('status')}, fr={entry.get('fr')!r})")

    if dry_run:
        print("--dry-run : rien n'est écrit.")
        return 0

    if not elide:
        print("Rien à élider.")
        return 0

    for key in elide:
        del store[key]
    sense_fr.write_store(store)

    lock = verify_fr_lock.load_lock() or {}
    n_lock_removed = sum(1 for key in elide if key in lock)
    for key in elide:
        lock.pop(key, None)
    if n_lock_removed:
        verify_fr_lock.write_lock(lock)

    print(f"{len(elide)} clé(s) héritée(s) supprimée(s) du magasin "
          f"(dont {n_lock_removed} retirée(s) du verrou). {len(keep)} clé(s) conservée(s) intactes.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Affiche le plan d'élision/conservation sans rien écrire.")
    args = parser.parse_args()
    raise SystemExit(run(dry_run=args.dry_run))
