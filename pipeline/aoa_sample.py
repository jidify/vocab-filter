"""Génère la liste des mots à annoter connu/inconnu pour trancher le
signe de l'AoA (voir le plan, "La question AoA" et item 7 de la
Verification). Ne dépend QUE de S4 (Zipf, AoA résiduel) — pas du sens
choisi en S5, donc peut tourner avant que le batch de désambiguïsation
soit terminé.

Principe : les deux conventions de signe (AOA_SIGN=+1 vs -1) ne
produisent des classements différents QUE pour les mots à fort
|aoa_resid| — ce sont les seuls où la question a un enjeu réel. On
prend donc les ~60 mots au résidu AoA le plus extrême, dans les deux
sens (précoce-mais-rare ET tardif), plutôt qu'un tirage aléatoire."""

from __future__ import annotations

import csv
import json

from pipeline import config, lexicon
from pipeline.score import aoa_residual


def run(n: int = 60) -> int:
    config.ensure_out_dir()

    with config.SELECTED_TYPES_PATH.open(encoding="utf-8") as f:
        types = [json.loads(l) for l in f]

    rows = []
    for t in types:
        zipf = t.get("zipf")
        if zipf is None:
            continue
        surface = t["surface_forms"][0] if t["surface_forms"] else t["lemma"]
        resid = aoa_residual(surface, t["lemma"], zipf)
        if resid is None:
            continue
        rows.append({
            "lemma": t["lemma"], "pos": t["wn_pos"], "zipf": zipf,
            "aoa_resid": resid, "book_count": t["book_count"],
        })

    rows.sort(key=lambda r: r["aoa_resid"])
    early_rare = rows[:n // 2]          # résidu très négatif : acquis tôt par les
                                         # natifs mais rare -> "précoce-mais-rare" à tester
    late = rows[-(n // 2):]             # résidu très positif : acquis tard par les
                                         # natifs (souvent latinat/transparent FR) -> "tardif" à tester

    sample = early_rare + late
    path = config.OUT_DIR / "aoa_annotation_sample.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "pos", "zipf", "aoa_resid", "book_count",
                          "hypothese", "connu_inconnu (à remplir)"])
        for r in early_rare:
            writer.writerow([r["lemma"], r["pos"], round(r["zipf"], 2),
                              round(r["aoa_resid"], 2), r["book_count"],
                              "précoce-mais-rare (devrait être une vraie lacune L2)", ""])
        for r in late:
            writer.writerow([r["lemma"], r["pos"], round(r["zipf"], 2),
                              round(r["aoa_resid"], 2), r["book_count"],
                              "tardif (devrait être transparent/cognat FR)", ""])

    print(f"{len(sample)} mots -> {path}")
    print("Remplissez la colonne 'connu_inconnu' avec 'connu' ou 'inconnu' "
          "(votre connaissance, avant de lire le livre).")
    print("Interprétation : si les 'précoce-mais-rare' sont majoritairement "
          "'inconnu' ET les 'tardif' majoritairement 'connu', le signe "
          "retenu (AOA_SIGN=-1 dans score.py) est confirmé. Si c'est "
          "l'inverse, il faut le retourner (AOA_SIGN=+1). Si aucun motif "
          "net ne se dégage, retirer l'AoA (AOA_SIGN=0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
