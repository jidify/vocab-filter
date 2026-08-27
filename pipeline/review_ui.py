"""Génère une page HTML locale, autonome, pour relire pipeline_out/sense_fr_review.csv
sans jamais avoir à taper un code WordNet à la main.

Constat qui motive ce module : `reassigner_vers` (voir pipeline/sense_fr.py::
REVIEW_FIELDS et pipeline/sense_fr_commit.py) attend un sense_id WordNet exact
("e-mail.v.01") ou une clé "mwe:<expression>:<label>" — techniquement fiable
(pipeline/score.py ne fait confiance qu'à ce que ce module a réellement écrit dans le
magasin, jamais à un texte deviné), mais personne ne connaît ces codes par cœur. Cette
page les affiche à choisir dans une liste déroulante, avec leur définition anglaise en
clair : le sense_id qui part dans le CSV téléchargé est TOUJOURS celui affiché par
WordNet, jamais retapé par l'utilisateur.

Pas de serveur, pas de dépendance externe (pas de CDN) : les données (lignes du CSV +
inventaire WordNet par mot, calculé une fois ici en Python) sont embarquées telles
quelles dans la page — elle s'ouvre en local, hors ligne, en double-cliquant dessus.

Un navigateur ne peut pas écrire directement dans pipeline_out/ (sécurité) : le bouton
"Télécharger" produit un fichier sense_fr_review.csv à déplacer par-dessus l'existant
avant de lancer `pipeline.sense_fr_commit` — étape manuelle assumée (voir le plan du
2026-08-27 "Une page HTML locale pour choisir le sens WordNet dans une liste").

Usage :
    uv run python -m pipeline.review_ui
    (puis ouvrir pipeline_out/review_ui.html dans un navigateur)
"""

from __future__ import annotations

import csv
import json

from pipeline import config, sense_fr
from pipeline.sense_fr_reassign import open_inventory


def load_rows() -> list[dict]:
    if not config.SENSE_FR_REVIEW_PATH.exists():
        return []
    with config.SENSE_FR_REVIEW_PATH.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_data(rows: list[dict]) -> list[dict]:
    """Enrichit chaque ligne avec son inventaire WordNet (toutes catégories
    grammaticales confondues, comme pipeline/sense_fr_reassign.py — le POS
    affiché par S5 n'est pas plus fiable que le sense_id). Vide pour les
    lignes `kind == "mwe"` : rien à re-clé, une expression figée ne
    redevient jamais un sense_id WordNet."""

    inventory_cache: dict[str, list[dict]] = {}
    data = []
    for row in rows:
        row = dict(row)
        if row.get("kind") == "synset":
            lemma = (row.get("lemmas_en") or "").split("/")[0].strip()
            if lemma not in inventory_cache:
                inventory_cache[lemma] = open_inventory(lemma) if lemma else []
            row["wordnet_inventory"] = inventory_cache[lemma]
        else:
            row["wordnet_inventory"] = []
        data.append(row)
    return data


def render_html(data: list[dict]) -> str:
    # Empêche un contexte_en contenant littéralement "</script>" de casser
    # le bloc embarqué (rare mais le livre contient des didascalies avec
    # toutes sortes de ponctuation).
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

    n_synset = sum(1 for r in data if r.get("kind") == "synset")
    n_no_candidate = sum(
        1 for r in data if r.get("kind") == "synset" and not r["wordnet_inventory"]
    )

    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Relecture sense_fr — {len(data)} en attente</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: system-ui, sans-serif; background: #f4f4f6; color: #1a1a1a;
          margin: 0; padding: 0 0 4rem; }}
  header {{ position: sticky; top: 0; background: #1a1a2e; color: #fff; padding: 1rem 1.5rem;
            display: flex; align-items: center; gap: 1rem; z-index: 10; flex-wrap: wrap; }}
  header h1 {{ font-size: 1.1rem; margin: 0; font-weight: 600; }}
  header .stats {{ font-size: 0.85rem; opacity: 0.8; }}
  header button {{ margin-left: auto; background: #4f8cff; color: #fff; border: none;
                    padding: 0.6rem 1.2rem; border-radius: 6px; font-size: 0.95rem;
                    cursor: pointer; font-weight: 600; }}
  header button:hover {{ background: #3a75e0; }}
  main {{ max-width: 980px; margin: 1.5rem auto; padding: 0 1rem; display: flex;
          flex-direction: column; gap: 1rem; }}
  .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; }}
  .card.touched {{ border-color: #4f8cff; box-shadow: 0 0 0 2px #4f8cff22; }}
  .row-top {{ display: flex; justify-content: space-between; align-items: baseline;
              gap: 1rem; flex-wrap: wrap; }}
  .key {{ font-family: ui-monospace, monospace; font-weight: 700; font-size: 1rem; }}
  .badge {{ display: inline-block; background: #eee; border-radius: 4px; padding: 0.1rem 0.5rem;
            font-size: 0.75rem; margin-left: 0.4rem; color: #555; }}
  .meta {{ font-size: 0.82rem; color: #666; margin-top: 0.15rem; }}
  .context {{ background: #f8f8fa; border: 1px solid #eee; border-radius: 6px;
              padding: 0.6rem 0.8rem; margin: 0.6rem 0; font-size: 0.9rem; line-height: 1.5;
              white-space: pre-wrap; max-height: 12rem; overflow-y: auto; }}
  .suggested {{ font-size: 0.85rem; color: #555; margin-bottom: 0.5rem; }}
  .suggested b {{ color: #222; }}
  label {{ display: block; font-size: 0.8rem; color: #444; margin: 0.5rem 0 0.15rem; font-weight: 600; }}
  select, input[type=text], textarea {{ width: 100%; box-sizing: border-box; padding: 0.4rem 0.5rem;
                                          border: 1px solid #ccc; border-radius: 5px; font-size: 0.9rem;
                                          font-family: inherit; }}
  textarea {{ resize: vertical; min-height: 2.5rem; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 1rem; }}
  .mwe-fallback {{ margin-top: 0.6rem; padding: 0.6rem 0.8rem; background: #fff8ec;
                    border: 1px solid #f0d9a8; border-radius: 6px; display: none; }}
  .mwe-fallback.visible {{ display: block; }}
  .no-candidate {{ font-size: 0.82rem; color: #a05a00; margin-top: 0.3rem; }}
  .decision-row {{ display: flex; gap: 1rem; align-items: flex-end; margin-top: 0.6rem; flex-wrap: wrap; }}
  .decision-row > div {{ flex: 1; min-width: 140px; }}
  footer {{ text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<header>
  <h1>Relecture sense_fr</h1>
  <span class="stats">{len(data)} en attente · {n_synset} mot(s) simple(s)
    ({n_no_candidate} sans aucun candidat WordNet) · {len(data) - n_synset} expression(s) figée(s)</span>
  <button onclick="downloadCsv()">Télécharger le CSV corrigé</button>
</header>
<main id="main"></main>
<footer>
  Généré par <code>pipeline/review_ui.py</code> — après téléchargement, déplacer le
  fichier par-dessus <code>pipeline_out/sense_fr_review.csv</code> puis lancer
  <code>uv run python -m pipeline.sense_fr_commit</code>.
</footer>

<script>
const ROWS = {payload};
const REVIEW_FIELDS = {json.dumps(sense_fr.REVIEW_FIELDS)};
const MWE_LABELS = ["idiome", "phrasal_verb", "semi_fige"];

function el(tag, attrs, children) {{
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {{}})) {{
    if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }}
  (children || []).forEach(c => c && e.appendChild(c));
  return e;
}}

function buildCard(row, idx) {{
  const card = el("div", {{ class: "card", id: "card-" + idx }});
  const isSynset = row.kind === "synset";

  card.appendChild(el("div", {{ class: "row-top" }}, [
    el("span", {{ class: "key", text: row.key }}),
    el("span", {{ class: "badge", text: row.kind }}),
    el("span", {{ class: "badge", text: (row.pos || "?") }}),
    el("span", {{ class: "badge", text: (row.occurrences || "0") + " occ." }}),
    el("span", {{ class: "badge", text: row.agreement || "" }}),
  ]));
  card.appendChild(el("div", {{ class: "meta", text: "lemmes : " + row.lemmas_en +
    "  —  définition actuelle : " + (row.definition_en || "(aucune)") }}));
  if (row.contexte_en) {{
    card.appendChild(el("div", {{ class: "context", text: row.contexte_en }}));
  }}
  if (row.suggested_fr) {{
    card.appendChild(el("div", {{ class: "suggested", html:
      "suggestion actuelle : <b>" + escapeHtml(row.suggested_fr) + "</b>" +
      (row.suggested_fr_alt ? " (alt. : " + escapeHtml(row.suggested_fr_alt) + ")" : "") +
      (row.sense_fit ? "  ·  sense_fit=" + row.sense_fit : "") }}));
  }}

  let selectEl = null, mweBox = null, exprInput = null, labelSelect = null, defInput = null;

  if (isSynset) {{
    label("Sens réel de cette occurrence :", card);
    selectEl = el("select", {{}});
    selectEl.appendChild(el("option", {{ value: "" }}, [document.createTextNode(
      "— garder tel quel (pas de re-clé, juste une traduction à confirmer) —")]));
    (row.wordnet_inventory || []).forEach(cand => {{
      const opt = el("option", {{ value: cand.sense_id }});
      opt.textContent = cand.pos + " · " + cand.sense_id + " · " + cand.definition;
      selectEl.appendChild(opt);
    }});
    selectEl.appendChild(el("option", {{ value: "__mwe__" }}, [document.createTextNode(
      "— aucun ne correspond, saisir une expression composée —")]));
    card.appendChild(selectEl);

    if (!row.wordnet_inventory || row.wordnet_inventory.length === 0) {{
      card.appendChild(el("div", {{ class: "no-candidate",
        text: "Aucun sens WordNet trouvé pour ce lemme — bascule directement sur la saisie d'expression." }}));
      selectEl.value = "__mwe__";
    }}

    mweBox = el("div", {{ class: "mwe-fallback" }});
    mweBox.appendChild(label_only("Expression complète (ex. \\"smart-ass\\")"));
    exprInput = el("input", {{ type: "text", placeholder: "smart-ass" }});
    mweBox.appendChild(exprInput);
    mweBox.appendChild(label_only("Type d'expression"));
    labelSelect = el("select", {{}});
    MWE_LABELS.forEach(l => labelSelect.appendChild(el("option", {{ value: l, text: l }})));
    mweBox.appendChild(labelSelect);
    mweBox.appendChild(label_only("Définition en anglais (optionnel)"));
    defInput = el("textarea", {{ placeholder: "A person who makes clever, sarcastic remarks." }});
    mweBox.appendChild(defInput);
    card.appendChild(mweBox);

    const syncMwe = () => {{
      mweBox.classList.toggle("visible", selectEl.value === "__mwe__");
    }};
    selectEl.addEventListener("change", syncMwe);
    syncMwe();
  }}

  const decisionRow = el("div", {{ class: "decision-row" }});
  const frFinal = el("input", {{ type: "text", value: row.suggested_fr || "" }});
  const frAltFinal = el("input", {{ type: "text", value: row.suggested_fr_alt || "",
    placeholder: "alt1; alt2" }});
  const noteInput = el("input", {{ type: "text", placeholder: "note pour l'audit (optionnel)" }});
  const decisionSelect = el("select", {{}});
  [["", "(laisser en attente)"], ["ok", "ok — valider"], ["no", "no — suggestion fausse"],
   ["none", "none — aucun équivalent FR"]].forEach(([v, t]) =>
    decisionSelect.appendChild(el("option", {{ value: v, text: t }})));

  decisionRow.appendChild(wrapField("Traduction française", frFinal));
  decisionRow.appendChild(wrapField("Alternatives (;)", frAltFinal));
  decisionRow.appendChild(wrapField("Décision", decisionSelect));
  decisionRow.appendChild(wrapField("Note", noteInput));
  card.appendChild(decisionRow);

  const markTouched = () => {{
    card.classList.add("touched");
    if (decisionSelect.value === "" && frFinal.value.trim()) decisionSelect.value = "ok";
  }};
  [frFinal, frAltFinal, noteInput].forEach(i => i.addEventListener("input", markTouched));
  if (selectEl) selectEl.addEventListener("change", markTouched);
  if (exprInput) exprInput.addEventListener("input", markTouched);

  row._widgets = {{ selectEl, exprInput, labelSelect, defInput, frFinal, frAltFinal, decisionSelect, noteInput }};
  return card;
}}

function label(text, parent) {{ parent.appendChild(label_only(text)); }}
function label_only(text) {{ return el("label", {{ text }}); }}
function wrapField(labelText, inputEl) {{
  const div = el("div", {{}});
  div.appendChild(el("label", {{ text: labelText }}));
  div.appendChild(inputEl);
  return div;
}}
function escapeHtml(s) {{
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}}

const main = document.getElementById("main");
ROWS.forEach((row, idx) => main.appendChild(buildCard(row, idx)));

function csvField(v) {{
  v = (v == null) ? "" : String(v);
  if (/[",\\n]/.test(v)) return '"' + v.replace(/"/g, '""') + '"';
  return v;
}}

function downloadCsv() {{
  const lines = [REVIEW_FIELDS.map(csvField).join(",")];
  ROWS.forEach(row => {{
    const w = row._widgets;
    let reassignerVers = "";
    let definitionPerso = "";
    if (w && w.selectEl) {{
      if (w.selectEl.value === "__mwe__") {{
        const expr = (w.exprInput.value || "").trim().toLowerCase()
          .replace(/-/g, " ").replace(/\\s+/g, " ");
        if (expr) {{
          reassignerVers = "mwe:" + expr + ":" + w.labelSelect.value;
          definitionPerso = (w.defInput.value || "").trim();
        }}
      }} else if (w.selectEl.value) {{
        reassignerVers = w.selectEl.value;
      }}
    }}
    const out = {{}};
    REVIEW_FIELDS.forEach(f => out[f] = row[f] || "");
    out["reassigner_vers"] = reassignerVers;
    out["definition_en_perso"] = definitionPerso;
    if (w) {{
      out["fr_final"] = w.frFinal.value || "";
      out["fr_alt_final"] = w.frAltFinal.value || "";
      out["decision"] = w.decisionSelect.value || "";
      out["note"] = w.noteInput.value || "";
    }}
    lines.push(REVIEW_FIELDS.map(f => csvField(out[f])).join(","));
  }});
  const blob = new Blob(["\\ufeff" + lines.join("\\r\\n")], {{ type: "text/csv;charset=utf-8" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sense_fr_review.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
}}
</script>
</body>
</html>
"""


def run() -> int:
    config.ensure_out_dir()
    rows = load_rows()
    if not rows:
        print(f"Aucune ligne en attente dans {config.SENSE_FR_REVIEW_PATH} — rien à générer.")
        return 0

    data = build_data(rows)
    html = render_html(data)
    config.REVIEW_UI_PATH.write_text(html, encoding="utf-8")

    n_synset = sum(1 for r in data if r["kind"] == "synset")
    n_no_candidate = sum(1 for r in data if r["kind"] == "synset" and not r["wordnet_inventory"])
    print(f"{len(data)} ligne(s) ({n_synset} mot(s), {n_no_candidate} sans candidat WordNet) "
          f"-> {config.REVIEW_UI_PATH}")
    print("Ouvrir ce fichier dans un navigateur, remplir, télécharger le CSV corrigé, "
          "le déplacer sur pipeline_out/sense_fr_review.csv, puis lancer "
          "`uv run python -m pipeline.sense_fr_commit`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
