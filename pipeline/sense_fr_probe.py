"""Sonde jetable — Étape 0 du plan de traduction FR par sense_id
(voir le plan "Traduction française de référence, indexée par
sense_id"). Lecture seule : n'écrit rien dans le magasin, ne fait
aucun appel LLM. Réutilise les fonctions de collecte de
pipeline/sense_fr.py — ne pas dupliquer la logique de recherche
omw-fr/WoNeF ici.

Interroge omw-fr (WOLF) et WoNeF séparément pour chaque sense_id
réellement retenu dans pipeline_out/vocab.jsonl, et dénombre les
niveaux d'accord entre les deux ressources. A servi à fixer, sur des
chiffres réels (voir la sortie archivée), les seuils utilisés par
sense_fr.py::classify_synset_key.

Usage :
    uv run python -m pipeline.sense_fr_probe
"""

from __future__ import annotations

from collections import Counter

from nltk.corpus import wordnet as nwn
from nltk.corpus.reader.wordnet import WordNetError

from pipeline import sense_fr, senses


def run() -> int:
    print("Chargement de WoNeF (f-score, seule variante présente)...")
    wonef_by_id = sense_fr.load_wonef_fscore()
    print(f"  {len(wonef_by_id)} synsets indexés.\n")

    targets = sense_fr.collect_targets()
    sense_entries = [t for t in targets.values() if t["kind"] == "synset"]
    mwe_entries = [t for t in targets.values() if t["kind"] == "mwe"]

    print(f"{len(sense_entries)} sense_id distincts (unités \"word\") dans vocab.jsonl.")
    print(f"{len(mwe_entries)} unités MWE (sans synset — hors périmètre de cette sonde ; "
          f"dépendront entièrement du LLM + relecture dans sense_fr.py).\n")

    categories = Counter()
    unresolved = []
    examples = {"none": [], "disagree": []}

    for entry in sorted(sense_entries, key=lambda e: -e["occurrences"]):
        sense_id = entry["key"]
        try:
            synset = nwn.synset(sense_id)
        except (WordNetError, ValueError) as exc:
            unresolved.append((sense_id, str(exc)))
            continue

        offset = f"{synset.offset():08d}"
        pos = synset.pos()
        english_lemmas = [l.name().replace("_", " ") for l in synset.lemmas()]

        omw = sense_fr.fr_candidates_omw(offset, pos, english_lemmas)
        wonef = sense_fr.fr_candidates_wonef(offset, pos, english_lemmas)

        omw_stems = {senses.fr_stem(c) for c in omw}
        wonef_stems = {senses.fr_stem(c) for c in wonef}

        if not omw and not wonef:
            categories["aucune_source"] += 1
            if len(examples["none"]) < 10:
                examples["none"].append((sense_id, entry["lemmas_en"]))
        elif omw and not wonef:
            categories["omw_seul"] += 1
        elif wonef and not omw:
            categories["wonef_seul"] += 1
        elif omw_stems & wonef_stems:
            categories["concordantes"] += 1
        else:
            categories["divergentes"] += 1
            if len(examples["disagree"]) < 15:
                examples["disagree"].append((sense_id, entry["lemmas_en"], omw, wonef))

    total = sum(categories.values())
    print("=" * 70)
    print("Couverture par source (sur sense_id résolus par NLTK) :")
    print("=" * 70)
    for label, key in [
        ("Aucune ressource", "aucune_source"),
        ("omw-fr seul", "omw_seul"),
        ("WoNeF seul", "wonef_seul"),
        ("Concordantes (omw-fr ∩ WoNeF non vide)", "concordantes"),
        ("Divergentes (omw-fr et WoNeF, sans recouper)", "divergentes"),
    ]:
        n = categories[key]
        pct = 100 * n / total if total else 0
        print(f"  {label:<45} {n:>4} ({pct:5.1f}%)")
    print(f"  {'TOTAL':<45} {total:>4}")

    if unresolved:
        print(f"\n{len(unresolved)} sense_id NON résolus par NLTK (à investiguer) :")
        for sid, err in unresolved[:20]:
            print(f"  - {sid} : {err}")

    print("\n" + "=" * 70)
    print("Exemples — divergentes (omw-fr et WoNeF proposent des choses différentes)")
    print("=" * 70)
    for sid, lemmas, omw, wonef in examples["disagree"]:
        print(f"  {sid} ({'/'.join(lemmas)})")
        print(f"    omw-fr : {omw}")
        print(f"    WoNeF  : {wonef}")

    print("\n" + "=" * 70)
    print("Exemples — aucune ressource (dépendront entièrement du LLM + relecture)")
    print("=" * 70)
    for sid, lemmas in examples["none"]:
        print(f"  {sid} ({'/'.join(lemmas)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
