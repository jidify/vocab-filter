"""Petit serveur HTTP local (stdlib seule, `127.0.0.1` uniquement) pour
relire les entrées `pending` de data/sense_fr.jsonl sans jamais taper un
sense_id ou une clé `mwe:` à la main, et sans passer par un cycle
télécharger -> déplacer -> `sense_fr_commit`.

Refonte du 2026-08-27 (voir le plan "IHM de correction manuelle :
plusieurs workflows, lexique piloté par les données") : la V1 était une
page HTML statique embarquant les données du run courant, qui ne savait
proposer qu'un autre SENS DU MÊME LEMME (`sense_fr_reassign.open_inventory`
appliqué au seul lemme affiché). Insuffisant dès qu'il faut changer de
MOT (ex. "mail" -> "e-mail"), réutiliser une traduction déjà validée pour
un autre livre, ou créer une expression qui n'existe encore nulle part —
ce dernier cas exigeait jusqu'ici d'éditer à la main
pipeline/mwe.py::CUSTOM_IDIOMS et pipeline/analyze.py::EMAIL_SPECIAL_CASES.

Cinq situations, une seule page :
    A. le sens affiché est le bon           -> traduction + décision
    B. mauvais sens, MÊME mot                -> liste des sens WordNet du lemme
    C. mauvais MOT                           -> recherche WordNet libre
    D. la cible existe déjà dans le magasin  -> 1 clic, reprend sa traduction
    E. la cible n'existe encore nulle part   -> formulaire de création
       (écrit dans data/custom_lexicon.jsonl, lu au runtime par mwe.py/
       analyze.py — voir pipeline/custom_lexicon.py)

Toute décision passe par pipeline.sense_fr_commit.apply_decision — LE
MÊME chemin que le commit par lot (`uv run python -m
pipeline.sense_fr_commit`, toujours disponible pour une relecture hors
ligne via CSV) — c'est cette unicité de chemin que verify_fr_lock.py
protège.

Usage :
    uv run python -m pipeline.review_ui
    uv run python -m pipeline.review_ui --port 8888 --no-browser
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from pipeline import config, custom_lexicon, purge_unit, sense_fr, senses, verify_fr_lock
from pipeline.sense_fr_commit import apply_decision
from pipeline.sense_fr_reassign import open_inventory, search as wordnet_search

# ============================================================
# Données — chargées à la demande, magasin toujours relu à chaque requête
# (fichier de 900 entrées, quelques ms — voir pipeline/sense_fr.py) pour
# refléter tout de suite une décision qui vient d'être écrite.
# ============================================================

_OCC_CACHE: dict[str, list[dict]] | None = None

# ThreadingHTTPServer donne un thread par requête, mais le lecteur WordNet
# de nltk (fichiers zip) n'est PAS thread-safe : deux requêtes simultanées
# touchant WordNet (ex. deux recherches dans le bloc de redirection, ou une
# recherche pendant qu'une autre requête construit l'inventaire d'un lemme)
# font planter la lecture du zip avec une AssertionError côté nltk (observé
# dans pipeline_out/review_ui.log). Plutôt que de traquer chaque site
# d'appel nltk à travers sense_fr_reassign.py/score.py/sense_fr_commit.py,
# on sérialise tout le traitement applicatif d'une requête : un serveur
# local à un seul utilisateur n'a rien à gagner à traiter deux requêtes en
# parallèle (chaque appel prend quelques ms), et ça élimine la classe
# entière de courses plutôt qu'un cas précis.
_REQUEST_LOCK = threading.Lock()


def get_occurrences_by_sense() -> dict[str, list[dict]]:
    """Phrases du livre courant, indexées par sens — pipeline_out/senses.jsonl
    ne change pas pendant une séance de relecture (seul un nouveau run S5 le
    régénère), mise en cache pour ne le lire qu'une fois par lancement du
    serveur, pas à chaque requête (~0,2s mesuré, voir sense_fr.load_store()
    rechargé lui à chaque requête car quasi instantané)."""
    global _OCC_CACHE
    if _OCC_CACHE is None:
        _OCC_CACHE = senses.load_occurrences_by_sense()
    return _OCC_CACHE


def _exact(candidate: dict, lemma: str) -> bool:
    lemma_norm = lemma.strip().lower().replace(" ", "_")
    return any(l.lower() == lemma_norm for l in candidate.get("lemmas", []))


def _attach_wordnet_inventory(rows: list[dict]) -> list[dict]:
    inventory_cache: dict[str, list[dict]] = {}
    for row in rows:
        if row["kind"] != "synset":
            row["wordnet_inventory"] = []
            continue
        lemma = (row.get("lemmas_en") or "").split("/")[0].strip()
        if lemma not in inventory_cache:
            inventory_cache[lemma] = open_inventory(lemma) if lemma else []
        candidates = inventory_cache[lemma]
        for c in candidates:
            c["exact"] = _exact(c, lemma)
            c["current"] = c["sense_id"] == row["key"]
        row["wordnet_inventory"] = candidates
    return rows


def build_pending_payload() -> list[dict]:
    store = sense_fr.load_store()
    rows = sense_fr.pending_review_rows(store, get_occurrences_by_sense())
    return _attach_wordnet_inventory(rows)


def build_flagged_payload() -> list[dict]:
    """Entrées DÉJÀ verrouillées (voir verify_fr_lock.LOCKED_STATUSES)
    mais dont le pipeline lui-même a noté `sense_fit == "mismatch"` —
    ex. beat.n.08 verrouillé sur "brève pause" alors que sa définition
    WordNet est "a regular rate of repetition" (S6b a réécrit la
    traduction pour coller au contexte du livre, un stage de "juge"
    ultérieur a validé automatiquement malgré le drapeau). Invisibles
    dans build_pending_payload (status != "pending"), donc jusqu'ici
    impossibles à corriger depuis cette page — voir le plan du
    2026-08-27 "Étendre l'IHM aux entrées verrouillées incohérentes".

    `decided_by != "human"` exclut toute entrée déjà retraitée par un
    relecteur (apply_decision y écrit systématiquement "human") : une
    fois corrigée elle ne revient plus ici, même si `sense_fit` (annotation
    historique de la passe automatique) n'est jamais effacé."""
    store = sense_fr.load_store()
    flagged = [
        e for e in store.values()
        if e.get("status") in verify_fr_lock.LOCKED_STATUSES
        and e.get("sense_fit") == "mismatch"
        and e.get("decided_by") != "human"
    ]
    flagged.sort(key=lambda e: -e.get("occurrences", 0))
    rows = [sense_fr.build_review_row(e, get_occurrences_by_sense()) for e in flagged]
    return _attach_wordnet_inventory(rows)


def search_store(query: str, limit: int = 30) -> list[dict]:
    q = query.strip().lower()
    if len(q) < 2:
        return []
    store = sense_fr.load_store()
    hits = []
    for e in store.values():
        haystack = " ".join([
            e["key"], " ".join(e.get("lemmas_en") or []),
            e.get("definition_en") or "", e.get("fr") or "",
        ]).lower()
        if q in haystack:
            hits.append({
                "key": e["key"], "kind": e["kind"],
                "lemmas_en": e.get("lemmas_en") or [], "pos": e.get("pos"),
                "definition_en": e.get("definition_en") or "",
                "fr": e.get("fr"), "fr_alt": e.get("fr_alt") or [],
                "status": e.get("status"),
            })
    # Correspondance exacte de clé/lemme d'abord, puis alphabétique.
    hits.sort(key=lambda h: (q not in h["key"].lower(), h["key"]))
    return hits[:limit]


# ============================================================
# Décision
# ============================================================

DECISION_FIELDS = [
    "key", "decision", "reassigner_vers", "definition_en_perso",
    "fr_final", "fr_alt_final", "note",
]


def handle_decision(body: dict) -> tuple[int, dict]:
    row = {f: (body.get(f) or "") for f in DECISION_FIELDS}
    if not row["key"]:
        return 400, {"error": "champ 'key' manquant"}

    store = sense_fr.load_store()
    result = apply_decision(store, row, reviewer="human")

    if result["status"] == "error":
        return 400, {"error": result["message"], "key": result["key"]}
    if result["status"] == "skipped":
        return 400, {"error": "ni decision ni reassigner_vers renseignés", "key": row["key"]}

    lexicon_added = []
    if result["status"] == "reassigned":
        new_entry = store[result["key"]]
        if body.get("add_to_lexicon") and result["key"].startswith("mwe:"):
            custom_lexicon.add_idiom(new_entry["lemmas_en"][0], new_entry.get("definition_en") or "")
            lexicon_added.append(f"expression « {new_entry['lemmas_en'][0]} »")
        surfaces = [s.strip() for s in (body.get("tokenizer_surfaces") or "").split(",") if s.strip()]
        if surfaces:
            custom_lexicon.add_tokenizer_surfaces(
                surfaces, reason=f"ajouté depuis review_ui pour {result['key']}",
            )
            lexicon_added.append(f"cas de tokenisation {surfaces}")

    sense_fr.write_store(store)
    remaining = sense_fr.write_review_csv(store, get_occurrences_by_sense())

    return 200, {
        "status": result["status"], "key": result["key"], "message": result["message"],
        "lexicon_added": lexicon_added, "remaining": remaining,
    }


def handle_purge(body: dict) -> tuple[int, dict]:
    """POST /api/purge — supprime DÉFINITIVEMENT une entrée (voir
    pipeline/purge_unit.py) : ni un mauvais sens ni un mauvais mot, mais
    une entrée qui n'a rien à faire dans le vocabulaire (ex. "york" pour
    des occurrences de "New York" — entité nommée que les deux gardes
    automatiques ne détectent pas, voir la docstring de purge_unit).
    Distinct du bloc A (decision="no", qui laisse l'entrée `rejected`
    dans le magasin) : ici la clé disparaît de data/sense_fr.jsonl et de
    pipeline_out/ tout entier."""
    key = (body.get("key") or "").strip()
    if not key:
        return 400, {"error": "champ 'key' manquant"}
    reason = (body.get("reason") or "").strip()

    try:
        result = purge_unit.purge(key, reason=reason)
    except ValueError as exc:
        return 400, {"error": str(exc), "key": key}

    # senses.jsonl vient d'être réécrit par purge_unit.purge() — le cache
    # de la séance (voir get_occurrences_by_sense ci-dessus) est périmé.
    global _OCC_CACHE
    _OCC_CACHE = None

    return 200, {"status": "purged", "key": result["key"], "removed": result["removed"]}


# ============================================================
# Serveur HTTP
# ============================================================

class Handler(BaseHTTPRequestHandler):
    server_version = "ReviewUI/2"

    def log_message(self, fmt, *args):  # moins bavard que le défaut
        pass

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        q = (qs.get("q") or [""])[0]

        if parsed.path == "/":
            self._html(PAGE_HTML)  # statique, aucun accès magasin/WordNet — pas besoin du verrou
            return

        with _REQUEST_LOCK:
            if parsed.path == "/api/pending":
                self._json(200, build_pending_payload())
            elif parsed.path == "/api/flagged":
                self._json(200, build_flagged_payload())
            elif parsed.path == "/api/wordnet":
                self._json(200, wordnet_search(q))
            elif parsed.path == "/api/store":
                self._json(200, search_store(q))
            else:
                self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        handler = {"/api/decision": handle_decision, "/api/purge": handle_purge}.get(parsed.path)
        if handler is None:
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"error": "JSON invalide"})
            return
        with _REQUEST_LOCK:
            code, payload = handler(body)
        self._json(code, payload)


# ============================================================
# Page (coquille HTML autonome ; les données arrivent par fetch())
# ============================================================

PAGE_HTML = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Relecture sense_fr</title>
<style>
  :root { color-scheme: light; }
  body { font-family: system-ui, sans-serif; background: #f4f4f6; color: #1a1a1a;
         margin: 0; padding: 0 0 4rem; }
  header { position: sticky; top: 0; background: #1a1a2e; color: #fff; padding: 1rem 1.5rem;
           display: flex; align-items: center; gap: 1rem; z-index: 10; flex-wrap: wrap; }
  header h1 { font-size: 1.1rem; margin: 0; font-weight: 600; }
  header .stats { font-size: 0.85rem; opacity: 0.85; }
  header .reload-btn { margin-left: auto; background: #2f2f4f; color: #fff; border: 1px solid #4a4a70;
                        border-radius: 6px; padding: 0.4rem 0.8rem; font-size: 0.85rem; cursor: pointer; }
  header .reload-btn:hover { background: #3a3a5f; }
  .intro { max-width: 980px; margin: 1rem auto 0; padding: 0.8rem 1.25rem; background: #eef2ff;
           border: 1px solid #c7d2fe; border-radius: 8px; font-size: 0.88rem; line-height: 1.5; }
  .intro b { color: #3730a3; }
  main { max-width: 980px; margin: 1.5rem auto; padding: 0 1rem; display: flex;
         flex-direction: column; gap: 1rem; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; }
  .card.touched { border-color: #4f8cff; box-shadow: 0 0 0 2px #4f8cff22; }
  .row-top { display: flex; justify-content: space-between; align-items: baseline;
             gap: 1rem; flex-wrap: wrap; }
  .key { font-family: ui-monospace, monospace; font-weight: 700; font-size: 1rem; }
  .badge { display: inline-block; background: #eee; border-radius: 4px; padding: 0.1rem 0.5rem;
           font-size: 0.75rem; margin-left: 0.4rem; color: #555; }
  .meta { font-size: 0.82rem; color: #666; margin-top: 0.15rem; }
  .surface-badge { display: block; margin: 0.6rem 0; padding: 0.5rem 0.8rem; background: #fef08a;
                    border: 2px solid #eab308; border-radius: 6px; font-size: 1rem; color: #713f12; }
  .surface-badge .lbl { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em;
                          font-weight: 600; opacity: 0.85; display: block; margin-bottom: 0.1rem; }
  .surface-badge b { font-size: 1.15rem; }
  .context { background: #f8f8fa; border: 1px solid #eee; border-radius: 6px;
             padding: 0.6rem 0.8rem; margin: 0.6rem 0; font-size: 0.9rem; line-height: 1.5;
             white-space: pre-wrap; max-height: 12rem; overflow-y: auto; }
  .context mark.surface-hit { background: #fef08a; padding: 0 0.15rem; border-radius: 3px;
                                font-weight: 700; color: #713f12; }
  .suggested { font-size: 0.85rem; color: #555; margin-bottom: 0.5rem; }
  .suggested b { color: #222; }
  label { display: block; font-size: 0.8rem; color: #444; margin: 0.5rem 0 0.15rem; font-weight: 600; }
  select, input[type=text], textarea { width: 100%; box-sizing: border-box; padding: 0.4rem 0.5rem;
                                        border: 1px solid #ccc; border-radius: 5px; font-size: 0.9rem;
                                        font-family: inherit; }
  textarea { resize: vertical; min-height: 2.5rem; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0 1rem; }
  .decision-row { display: flex; gap: 1rem; align-items: flex-end; margin-top: 0.6rem; flex-wrap: wrap; }
  .decision-row > div { flex: 1; min-width: 140px; }
  footer { text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; }

  .action-box { margin-top: 1rem; padding: 0.9rem 1rem; border: 1px solid #e2e2e2; border-radius: 8px;
                background: #fcfcfd; }
  .action-title { font-size: 0.9rem; font-weight: 700; color: #333; margin-bottom: 0.5rem; }
  .link-note { font-size: 0.8rem; color: #888; margin-top: 0.8rem; font-style: italic; }
  .redirect-panel { margin-top: 0.7rem; padding: 0.8rem; background: #fafafa;
                     border: 1px solid #e2e2e2; border-radius: 6px; }
  .cand-group-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em;
                       color: #888; margin: 0.7rem 0 0.3rem; }
  .cand-group-title:first-child { margin-top: 0; }
  .cand { display: flex; justify-content: space-between; align-items: center; gap: 0.6rem;
          padding: 0.4rem 0.5rem; border-radius: 5px; cursor: pointer; font-size: 0.86rem; }
  .cand:hover { background: #eef2ff; }
  .cand.selected { background: #dbeafe; }
  .cand .txt { flex: 1; }
  .cand .txt code { font-weight: 600; }
  .cand .current-tag { color: #a05a00; font-size: 0.78rem; }
  .cand .fr-tag { color: #166534; font-weight: 600; }
  .search-row { display: flex; gap: 0.5rem; margin-bottom: 0.4rem; }
  .search-row input { flex: 1; }
  .search-hint { font-size: 0.78rem; color: #888; margin: 0.2rem 0 0.5rem; }
  .empty-note { font-size: 0.82rem; color: #888; padding: 0.3rem 0.5rem; }
  .create-toggle { margin-top: 0.5rem; background: none; border: none; color: #3730a3;
                    font-size: 0.85rem; cursor: pointer; text-decoration: underline; padding: 0; }
  .create-form { display: none; margin-top: 0.6rem; padding: 0.7rem; background: #fff8ec;
                 border: 1px solid #f0d9a8; border-radius: 6px; }
  .create-form.open { display: block; }
  .create-form .checkline { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.5rem; flex-wrap: wrap; }
  .create-form .checkline label { margin: 0; font-weight: 400; }
  .create-form .checkline input[type=checkbox] { width: auto; }
  .create-form .checkline input[type=checkbox]:disabled { opacity: 0.4; cursor: not-allowed; }
  .create-form .checkline .clear { margin-left: 0.4rem; cursor: pointer; color: #a05a00;
                                     text-decoration: underline; font-size: 0.82rem; }
  .lexicon-warning { margin-top: 0.6rem; padding: 0.5rem 0.7rem; background: #fde3e3;
                      border: 1px solid #e59a9a; border-radius: 6px; font-size: 0.8rem;
                      color: #7a1f1f; line-height: 1.4; }
  .create-form button { margin-top: 0.6rem; }
  .armed-banner { margin-top: 0.6rem; padding: 0.55rem 0.75rem; background: #fff3d6;
                  border: 1px solid #f0c96b; border-radius: 6px; font-size: 0.85rem; color: #6b4e00; }
  .armed-banner code { font-weight: 700; }
  .armed-banner .clear { float: right; cursor: pointer; color: #6b4e00; text-decoration: underline; }
  .tokenizer-field { margin-top: 0.5rem; }
  .save-row { margin-top: 0.9rem; display: flex; align-items: center; gap: 0.8rem; }
  .save-row button { background: #4f8cff; color: #fff; border: none; padding: 0.5rem 1.1rem;
                      border-radius: 6px; font-size: 0.88rem; cursor: pointer; font-weight: 600; }
  .save-row button:hover { background: #3a75e0; }
  .save-row button:disabled { background: #aac2ee; cursor: default; }
  .save-msg { font-size: 0.85rem; }
  .save-msg.error { color: #b91c1c; }
  .save-msg.ok { color: #166534; }
  .btn-mini { background: none; border: 1px solid #ccc; border-radius: 5px; padding: 0.2rem 0.6rem;
              font-size: 0.78rem; cursor: pointer; color: #444; }
  .btn-mini:hover { background: #eef2ff; }

  .action-box.danger { border-color: #e59a9a; background: #fff7f7; }
  .action-box.danger .action-title { color: #7a1f1f; }
  .danger-note { font-size: 0.82rem; color: #7a1f1f; line-height: 1.4; margin-bottom: 0.5rem; }
  .purge-row { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; }
  .purge-row input[type=text] { flex: 1; min-width: 200px; }
  .btn-danger { background: #c0392b; color: #fff; border: none; padding: 0.5rem 1.1rem;
                border-radius: 6px; font-size: 0.88rem; cursor: pointer; font-weight: 600; }
  .btn-danger:hover { background: #a5311f; }
  .btn-danger:disabled { background: #e0a6a0; cursor: default; }
  .purge-cancel { font-size: 0.82rem; color: #666; text-decoration: underline; cursor: pointer; }
  .purge-detail { font-size: 0.78rem; color: #666; margin-top: 0.4rem; }

  .section-title { max-width: 980px; margin: 1.6rem auto 0.4rem; padding: 0 1rem; font-size: 1rem;
                    font-weight: 700; color: #333; }
  .section-desc { max-width: 980px; margin: 0 auto 0.6rem; padding: 0 1rem; font-size: 0.85rem; color: #666; }
  .card.flagged { border-left: 5px solid #dc7a12; }
  .locked-banner { margin-bottom: 0.6rem; padding: 0.6rem 0.8rem; background: #fdecd3;
                    border: 1px solid #dc7a12; border-radius: 6px; font-size: 0.85rem; color: #7c3d02; }
  .locked-banner b { color: #5a2c00; }
</style>
</head>
<body>
<header>
  <h1>Relecture sense_fr</h1>
  <span class="stats" id="stats">chargement…</span>
  <button class="reload-btn" type="button" onclick="loadAll()">🔄 Recharger</button>
</header>
<div class="intro" id="intro">
  Chaque carte propose <b>trois actions indépendantes</b>, chacune avec son propre bouton :
  corriger la traduction du sens affiché (bloc du haut), rediriger ces occurrences vers un
  autre sens/une autre expression quand le mot affiché n'est pas le bon (bloc du milieu), et
  supprimer l'entrée quand elle n'a simplement rien à faire dans le vocabulaire (bloc du bas,
  en rouge) — plusieurs peuvent être faites l'une après l'autre sur la même carte tant que la
  suppression n'a pas eu lieu. Ex. « ass » alors que le texte dit « smart-ass » (rediriger),
  ou « york » alors que le texte dit « New York » (supprimer — ce n'est pas un mauvais sens
  ni un mauvais mot, ce n'est simplement pas du vocabulaire). Rediriger ne modifie jamais la
  clé d'origine (elle reste dans le magasin permanent, potentiellement correcte pour un autre
  livre) — pensez au bloc du haut si elle a aussi besoin d'être corrigée. Supprimer, en
  revanche, retire la clé et toute trace la concernant dans <code>pipeline_out/</code> ;
  c'est ponctuel, pas mémorisé — un futur run complet depuis S1 peut la recréer. Tout
  s'enregistre directement dans <code>data/sense_fr.jsonl</code>, aucun fichier à déplacer.
</div>
<main id="main"></main>
<footer>Servi localement par <code>pipeline/review_ui.py</code> — Ctrl+C dans le terminal pour arrêter.</footer>

<script>
const MWE_LABELS = ["idiome", "phrasal_verb", "semi_fige"];
let PENDING = [];
let FLAGGED = [];

function el(tag, attrs, children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else e.setAttribute(k, v);
  }
  (children || []).forEach(c => c && e.appendChild(c));
  return e;
}
function escapeHtml(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function highlightSurfaceForms(text, forms) {
  const escaped = escapeHtml(text);
  if (!forms || !forms.length) return escaped;
  const pattern = forms.map(escapeRegExp).sort((a, b) => b.length - a.length).join("|");
  const re = new RegExp("\\b(" + pattern + ")\\b", "gi");
  return escaped.replace(re, '<mark class="surface-hit">$1</mark>');
}
function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function loadAll() {
  const [pendingRes, flaggedRes] = await Promise.all([
    fetch("/api/pending"), fetch("/api/flagged"),
  ]);
  PENDING = await pendingRes.json();
  FLAGGED = await flaggedRes.json();
  render();
}

function updateStats() {
  // Compte statique au moment du chargement — une carte peut légitimement
  // rester affichée après un enregistrement réussi (ex. une redirection
  // seule ne change jamais le statut de la clé d'origine, voir
  // sense_fr_commit.py). "🔄 Recharger" donne le compte à jour.
  document.getElementById("stats").textContent =
    `${PENDING.length} en attente` +
    (FLAGGED.length ? `  ·  ${FLAGGED.length} verrouillée(s) incohérente(s)` : "");
}

function render() {
  const main = document.getElementById("main");
  main.innerHTML = "";

  if (FLAGGED.length) {
    main.appendChild(el("div", { class: "section-title",
      text: "⚠️ Déjà verrouillées, mais incohérentes avec leur propre définition" }));
    main.appendChild(el("div", { class: "section-desc",
      html: "Le pipeline a lui-même noté <code>sense_fit = mismatch</code> pour ces entrées, " +
        "puis les a quand même verrouillées automatiquement. Elles sont réutilisées telles " +
        "quelles pour les prochains livres tant qu'elles ne sont pas corrigées ici." }));
    FLAGGED.forEach(row => main.appendChild(buildCard(row, { flagged: true })));
  }

  main.appendChild(el("div", { class: "section-title", text: "À relire" }));
  if (PENDING.length === 0) {
    main.appendChild(el("div", { class: "empty-note",
      text: "Aucune entrée en attente. 🎉" }));
  }
  PENDING.forEach(row => main.appendChild(buildCard(row, {})));
  updateStats();
}

function buildSaveControls(label) {
  const saveRow = el("div", { class: "save-row" });
  const saveBtn = el("button", { type: "button", text: label });
  const saveMsg = el("span", { class: "save-msg" });
  saveRow.appendChild(saveBtn);
  saveRow.appendChild(saveMsg);
  return { saveRow, saveBtn, saveMsg };
}

async function doSave(payload, saveBtn, saveMsg) {
  saveMsg.textContent = "";
  saveMsg.className = "save-msg";
  saveBtn.disabled = true;
  try {
    const res = await fetch("/api/decision", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      saveMsg.textContent = "✗ " + data.error;
      saveMsg.className = "save-msg error";
      saveBtn.disabled = false;
      return false;
    }
    let msg = "✓ enregistré";
    if (data.status === "reassigned") msg += " → " + data.key;
    if (data.lexicon_added && data.lexicon_added.length) {
      msg += " (lexique : " + data.lexicon_added.join(", ") + ")";
    }
    saveMsg.textContent = msg;
    saveMsg.className = "save-msg ok";
    saveBtn.disabled = false;
    return true;
  } catch (e) {
    saveMsg.textContent = "✗ erreur réseau : " + e;
    saveMsg.className = "save-msg error";
    saveBtn.disabled = false;
    return false;
  }
}

function buildCard(row, opts) {
  opts = opts || {};
  const card = el("div", { class: "card" + (opts.flagged ? " flagged" : ""),
    id: "card-" + row.key.replace(/[^a-zA-Z0-9]/g, "_") });
  const isSynset = row.kind === "synset";

  card.appendChild(el("div", { class: "row-top" }, [
    el("span", { class: "key", text: row.key }),
    el("span", { class: "badge", text: row.kind }),
    el("span", { class: "badge", text: (row.pos || "?") }),
    el("span", { class: "badge", text: (row.occurrences || "0") + " occ." }),
    el("span", { class: "badge", text: row.agreement || "" }),
  ]));
  if (opts.flagged) {
    card.appendChild(el("div", { class: "locked-banner", html:
      "🔒 Déjà verrouillée (statut <b>" + escapeHtml(row.status || "") + "</b>) sur la traduction " +
      "ci-dessous, réutilisée telle quelle pour les prochains livres. Le pipeline note lui-même : « " +
      escapeHtml(row.sense_fit_note || "incohérence non détaillée") + " »" }));
  }
  card.appendChild(el("div", { class: "meta", text: "autres lemmes de ce sens WordNet : " + row.lemmas_en +
    "  —  définition actuelle : " + (row.definition_en || "(aucune)") }));
  if (row.surface_forms && row.surface_forms.length) {
    card.appendChild(el("div", { class: "surface-badge", html:
      '<span class="lbl">tel qu\'il apparaît dans le livre</span>' +
      row.surface_forms.map(f => "<b>" + escapeHtml(f) + "</b>").join(" · ") }));
  }
  if (row.contexte_en) {
    card.appendChild(el("div", { class: "context",
      html: highlightSurfaceForms(row.contexte_en, row.surface_forms) }));
  }
  if (row.suggested_fr) {
    card.appendChild(el("div", { class: "suggested", html:
      "suggestion actuelle : <b>" + escapeHtml(row.suggested_fr) + "</b>" +
      (row.suggested_fr_alt ? " (alt. : " + escapeHtml(row.suggested_fr_alt) + ")" : "") +
      (row.sense_fit ? "  ·  sense_fit=" + row.sense_fit : "") }));
  }

  const markTouched = () => card.classList.add("touched");

  // ============================================================
  // Bloc A — corriger la traduction de `row.key` lui-même. Totalement
  // indépendant du bloc B ci-dessous : ne touche jamais reassigner_vers.
  // ============================================================
  const boxA = el("div", { class: "action-box" });
  boxA.appendChild(el("div", { class: "action-title",
    text: "✏️ Traduire ce sens WordNet (" + row.key + ")" }));

  const frFinalA = el("input", { type: "text", value: row.suggested_fr || "" });
  const frAltA = el("input", { type: "text", value: row.suggested_fr_alt || "", placeholder: "alt1; alt2" });
  const noteA = el("input", { type: "text", placeholder: "note pour l'audit (optionnel)" });
  const decisionA = el("select", {});
  [["", "(laisser en attente)"], ["ok", "ok — valider"], ["no", "no — suggestion fausse"],
   ["none", "none — aucun équivalent FR"]].forEach(([v, t]) =>
    decisionA.appendChild(el("option", { value: v, text: t })));

  const rowA = el("div", { class: "decision-row" });
  rowA.appendChild(wrapField("Traduction française", frFinalA));
  rowA.appendChild(wrapField("Alternatives (;)", frAltA));
  rowA.appendChild(wrapField("Décision", decisionA));
  rowA.appendChild(wrapField("Note", noteA));
  boxA.appendChild(rowA);
  [frFinalA, frAltA, noteA].forEach(i => i.addEventListener("input", markTouched));
  decisionA.addEventListener("change", markTouched);

  const ctrlsA = buildSaveControls("Enregistrer cette traduction");
  boxA.appendChild(ctrlsA.saveRow);
  card.appendChild(boxA);

  ctrlsA.saveBtn.addEventListener("click", () => doSave({
    key: row.key, decision: decisionA.value, reassigner_vers: "", definition_en_perso: "",
    fr_final: frFinalA.value, fr_alt_final: frAltA.value, note: noteA.value,
    add_to_lexicon: false, tokenizer_surfaces: "",
  }, ctrlsA.saveBtn, ctrlsA.saveMsg));

  // ============================================================
  // Bloc B — rediriger CES occurrences vers un autre sens/une expression.
  // Toujours visible (jamais masqué par un bouton "déplier" : c'est cette
  // dissimulation qui rendait l'action invisible une fois le bloc A fait,
  // et réciproquement). N'écrit jamais dans `row.key` — voir le bandeau
  // d'aide ci-dessous.
  // ============================================================
  if (isSynset) {
    card.appendChild(el("div", { class: "link-note",
      text: "La redirection ci-dessous ne modifie pas " + row.key + " lui-même (il reste dans le " +
        "magasin permanent) — pensez aussi au bloc du haut s'il a besoin d'être corrigé ou rejeté." }));

    const boxB = el("div", { class: "action-box" });
    boxB.appendChild(el("div", { class: "action-title",
      text: "🔀 Rediriger ces occurrences vers un autre sens / une expression" }));

    const frFinalB = el("input", { type: "text", value: row.suggested_fr || "" });
    const frAltB = el("input", { type: "text", value: row.suggested_fr_alt || "", placeholder: "alt1; alt2" });
    const noteB = el("input", { type: "text", placeholder: "note pour l'audit (optionnel)" });
    const rowB = el("div", { class: "decision-row" });
    rowB.appendChild(wrapField("Traduction française (de la cible)", frFinalB));
    rowB.appendChild(wrapField("Alternatives (;)", frAltB));
    rowB.appendChild(wrapField("Note", noteB));
    boxB.appendChild(rowB);
    [frFinalB, frAltB, noteB].forEach(i => i.addEventListener("input", markTouched));

    const state = { redirect: null };  // {targetKey, definitionEnPerso, addToLexicon, sourceLabel}
    const redirectPanel = buildRedirectPanel(row, state, () => refreshArmedUi(), frFinalB, frAltB);
    boxB.appendChild(redirectPanel);

    const armedBanner = el("div", { class: "armed-banner" });
    armedBanner.style.display = "none";
    boxB.appendChild(armedBanner);

    const tokenizerField = el("div", { class: "tokenizer-field" }, [
      el("label", { text: "Cas de tokenisation à ajouter (optionnel, séparés par des virgules — "
        + "ex. e-mail, e-mails, e-mailing)" }),
      el("input", { type: "text", placeholder: "e-mail, e-mails, e-mailing, e-mailed" }),
    ]);
    tokenizerField.style.display = "none";
    boxB.appendChild(tokenizerField);

    function refreshArmedUi() {
      if (!state.redirect) {
        armedBanner.style.display = "none";
        tokenizerField.style.display = "none";
        return;
      }
      armedBanner.style.display = "block";
      armedBanner.innerHTML = "";
      armedBanner.appendChild(document.createTextNode("⚠ Redirection armée vers "));
      armedBanner.appendChild(el("code", { text: state.redirect.targetKey }));
      armedBanner.appendChild(document.createTextNode(
        " — la traduction du bloc ci-dessus (celui-ci) sera enregistrée sous cette NOUVELLE "
        + "clé, jamais sous " + row.key + ". "));
      const clearBtn = el("span", { class: "clear", text: "annuler" });
      clearBtn.addEventListener("click", () => { state.redirect = null; refreshArmedUi(); });
      armedBanner.appendChild(clearBtn);
      tokenizerField.style.display = "block";
      markTouched();
    }

    const ctrlsB = buildSaveControls("Enregistrer cette redirection");
    boxB.appendChild(ctrlsB.saveRow);
    card.appendChild(boxB);

    ctrlsB.saveBtn.addEventListener("click", async () => {
      if (!state.redirect) {
        ctrlsB.saveMsg.textContent = "✗ choisir d'abord une cible (sens WordNet, recherche, ou créer une expression)";
        ctrlsB.saveMsg.className = "save-msg error";
        return;
      }
      if (!frFinalB.value.trim()) {
        ctrlsB.saveMsg.textContent = "✗ traduction requise pour la cible";
        ctrlsB.saveMsg.className = "save-msg error";
        return;
      }
      const ok = await doSave({
        key: row.key, decision: "ok", reassigner_vers: state.redirect.targetKey,
        definition_en_perso: state.redirect.definitionEnPerso || "",
        fr_final: frFinalB.value, fr_alt_final: frAltB.value, note: noteB.value,
        add_to_lexicon: !!state.redirect.addToLexicon,
        tokenizer_surfaces: tokenizerField.querySelector("input").value,
      }, ctrlsB.saveBtn, ctrlsB.saveMsg);
      if (ok) { state.redirect = null; refreshArmedUi(); }
    });
  }

  // ============================================================
  // Bloc C — suppression définitive. Ni un mauvais sens (bloc A) ni un
  // mauvais mot (bloc B) : l'entrée n'a simplement rien à faire dans le
  // vocabulaire (ex. "york" pour des occurrences de "New York"). Retire
  // la clé et toute trace la concernant dans pipeline_out/ — voir
  // pipeline/purge_unit.py. Toujours présent, pending comme flagged.
  // Confirmation en deux temps DANS la page (pas de confirm() natif,
  // qu'aucun autre bloc de cette page n'utilise).
  // ============================================================
  const boxC = el("div", { class: "action-box danger" });
  boxC.appendChild(el("div", { class: "action-title",
    text: "🗑️ Supprimer définitivement " + row.key }));
  boxC.appendChild(el("div", { class: "danger-note", text:
    "À réserver aux entrées qui n'ont rien à faire dans le vocabulaire (bruit d'entité "
    + "nommée qu'aucune garde automatique n'a filtré — pas une mauvaise traduction, pas un "
    + "mauvais sens : pour ça, les blocs ci-dessus). Retire la clé de data/sense_fr.jsonl et "
    + "toute trace la concernant dans pipeline_out/, régénère vocab.csv/jsonl. Suppression "
    + "PONCTUELLE, rien n'est mémorisé : un futur run complet depuis S1 peut recréer la clé."
    + (isSynset ? "" : " Limite connue pour une clé mwe: l'expression reste dans "
       + "selected_mwe.jsonl et peut réapparaître dans vocab.csv sans traduction officielle.")
  }));

  const reasonInput = el("input", { type: "text",
    placeholder: "raison (optionnel, pour l'audit) — ex. \"New York, pas du vocabulaire\"" });
  boxC.appendChild(reasonInput);

  const purgeRow = el("div", { class: "purge-row" });
  const purgeBtn = el("button", { class: "btn-danger", type: "button", text: "Supprimer…" });
  const purgeMsg = el("span", { class: "save-msg" });
  purgeRow.appendChild(purgeBtn);
  purgeRow.appendChild(purgeMsg);
  boxC.appendChild(purgeRow);
  card.appendChild(boxC);

  let purgeArmed = false;
  purgeBtn.addEventListener("click", async () => {
    if (!purgeArmed) {
      purgeArmed = true;
      purgeBtn.textContent = "⚠️ Confirmer la suppression";
      purgeMsg.textContent = "";
      const cancel = el("span", { class: "purge-cancel", text: "annuler" });
      cancel.addEventListener("click", () => {
        purgeArmed = false;
        purgeBtn.textContent = "Supprimer…";
        cancel.remove();
      });
      purgeRow.appendChild(cancel);
      return;
    }
    purgeBtn.disabled = true;
    purgeMsg.textContent = "";
    purgeMsg.className = "save-msg";
    try {
      const res = await fetch("/api/purge", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: row.key, reason: reasonInput.value }),
      });
      const data = await res.json();
      if (!res.ok) {
        purgeMsg.textContent = "✗ " + data.error;
        purgeMsg.className = "save-msg error";
        purgeBtn.disabled = false;
        purgeArmed = false;
        purgeBtn.textContent = "Supprimer…";
        return;
      }
      PENDING = PENDING.filter(r => r.key !== row.key);
      FLAGGED = FLAGGED.filter(r => r.key !== row.key);
      updateStats();
      const detail = Object.entries(data.removed)
        .filter(([, n]) => n).map(([f, n]) => f + ":" + n).join(", ");
      card.innerHTML = "";
      card.className = "card";
      card.appendChild(el("div", { class: "purge-detail",
        text: "✓ " + row.key + " supprimé" + (detail ? " (" + detail + ")" : "") }));
    } catch (e) {
      purgeMsg.textContent = "✗ erreur réseau : " + e;
      purgeMsg.className = "save-msg error";
      purgeBtn.disabled = false;
      purgeArmed = false;
      purgeBtn.textContent = "Supprimer…";
    }
  });

  return card;
}

function wrapField(labelText, inputEl) {
  const div = el("div", {});
  div.appendChild(el("label", { text: labelText }));
  div.appendChild(inputEl);
  return div;
}

function candidateRow(labelHtml, tagHtml, onClick) {
  const row = el("div", { class: "cand" });
  row.appendChild(el("span", { class: "txt", html: labelHtml }));
  if (tagHtml) row.appendChild(el("span", { html: tagHtml }));
  row.addEventListener("click", onClick);
  return row;
}

function buildRedirectPanel(row, state, onArmed, frFinal, frAltFinal) {
  const panel = el("div", { class: "redirect-panel" });
  const resultsBox = el("div", {});
  panel.appendChild(resultsBox);

  function selectCandidate(el_, targetKey, opts) {
    resultsBox.querySelectorAll(".cand.selected").forEach(n => n.classList.remove("selected"));
    createBox.querySelectorAll(".cand.selected").forEach(n => n.classList.remove("selected"));
    el_.classList.add("selected");
    state.redirect = Object.assign({ targetKey, definitionEnPerso: "", addToLexicon: false }, opts || {});
    onArmed();
  }

  function renderOwnInventory() {
    resultsBox.innerHTML = "";
    const inv = row.wordnet_inventory || [];
    if (!inv.length) {
      resultsBox.appendChild(el("div", { class: "empty-note",
        text: "Aucun sens WordNet pour ce lemme — chercher un autre mot ci-dessous, "
          + "ou créer une expression." }));
      return;
    }
    const exact = inv.filter(c => c.exact);
    const approx = inv.filter(c => !c.exact);
    if (exact.length) {
      resultsBox.appendChild(el("div", { class: "cand-group-title", text: "Sens de « " + row.lemmas_en.split("/")[0] + " »" }));
      exact.forEach(c => resultsBox.appendChild(wordnetCandRow(c)));
    }
    if (approx.length) {
      resultsBox.appendChild(el("div", { class: "cand-group-title",
        text: "Correspondances approximatives (même racine, probablement pas le bon mot)" }));
      approx.forEach(c => resultsBox.appendChild(wordnetCandRow(c)));
    }
  }

  function wordnetCandRow(c) {
    const label = "<code>" + escapeHtml(c.sense_id) + "</code> (" + c.pos + ") — " + escapeHtml(c.definition);
    const tag = c.current ? '<span class="current-tag">← sens actuellement assigné</span>' : "";
    const r = candidateRow(label, tag, (ev) => selectCandidate(r, c.sense_id, {}));
    return r;
  }

  const searchRow = el("div", { class: "search-row" });
  const searchInput = el("input", { type: "text",
    placeholder: "chercher un autre mot ou une expression déjà traduite (ex. e-mail, smart)" });
  searchRow.appendChild(searchInput);
  panel.appendChild(el("div", { class: "search-hint",
    text: "Recherche dans WordNet et dans les traductions déjà validées." }));
  panel.insertBefore(searchRow, resultsBox);

  const searchResults = el("div", {});
  panel.appendChild(searchResults);

  const doSearch = debounce(async (q) => {
    if (!q || q.trim().length < 2) { searchResults.innerHTML = ""; renderOwnInventory(); return; }
    const [wnRes, storeRes] = await Promise.all([
      fetch("/api/wordnet?q=" + encodeURIComponent(q)).then(r => r.json()),
      fetch("/api/store?q=" + encodeURIComponent(q)).then(r => r.json()),
    ]);
    resultsBox.innerHTML = "";
    searchResults.innerHTML = "";
    if (storeRes.length) {
      searchResults.appendChild(el("div", { class: "cand-group-title", text: "Déjà dans le magasin (1 clic = reprend la traduction)" }));
      storeRes.forEach(h => {
        const label = "<code>" + escapeHtml(h.key) + "</code> — " +
          "<span class=\"fr-tag\">" + escapeHtml(h.fr || "(pas de traduction)") + "</span>" +
          " · " + escapeHtml(h.status || "");
        const r = candidateRow(label, "", (ev) => {
          // Workflow D : 1 clic arme la redirection ET reprend la
          // traduction existante telle quelle (rien à retaper).
          selectCandidate(r, h.key, { sourceLabel: "store" });
          frFinal.value = h.fr || "";
          frAltFinal.value = (h.fr_alt || []).join("; ");
        });
        searchResults.appendChild(r);
      });
    }
    if (wnRes.length) {
      searchResults.appendChild(el("div", { class: "cand-group-title", text: "Sens WordNet trouvés" }));
      wnRes.forEach(c => {
        const label = "<code>" + escapeHtml(c.sense_id) + "</code> (" + c.pos + ") — " +
          escapeHtml(c.definition) + "  <i>[" + escapeHtml(c.lemmas.join(", ")) + "]</i>";
        const r = candidateRow(label, "", (ev) => selectCandidate(r, c.sense_id, {}));
        searchResults.appendChild(r);
      });
    }
    if (!storeRes.length && !wnRes.length) {
      searchResults.appendChild(el("div", { class: "empty-note",
        text: "Rien trouvé — créer une expression ci-dessous." }));
    }
  }, 300);
  searchInput.addEventListener("input", () => doSearch(searchInput.value));

  const createToggle = el("button", { class: "create-toggle", type: "button",
    text: "✗ Aucun sens/mot ne convient — créer une expression composée" });
  const createBox = el("div", { class: "create-form" });
  const exprInput = el("input", { type: "text", placeholder: "smart-ass" });
  const labelSelect = el("select", {});
  MWE_LABELS.forEach(l => labelSelect.appendChild(el("option", { value: l, text: l })));
  const defInput = el("textarea", { placeholder: "A person who makes clever, sarcastic remarks." });
  // Décochée ET désactivée par défaut : cocher cette case ajoute l'expression
  // à data/custom_lexicon.jsonl, lue par mwe.py::get_matcher() (idiomatch) —
  // qui matche sur la seule SÉQUENCE DE LEMMES, sans distinguer le sens
  // (vérifié dans idiomatch/idiomatcher.py::add_idioms/build). Pour un mot
  // simple et courant (ex. "beat"), ça détournerait aussi son usage normal
  // (verbe "battre", etc.) dans TOUS les prochains livres — voir le plan du
  // 2026-08-27 "beat.n.08 / small beat / long beat". Le déverrouillage est
  // une action délibérée séparée, pour qu'elle ne puisse jamais être cochée
  // par inadvertance.
  const lexCheck = el("input", { type: "checkbox", disabled: "disabled" });
  const useBtn = el("button", { class: "btn-mini", type: "button", text: "Utiliser cette expression" });

  createBox.appendChild(el("label", { text: "Expression complète" }));
  createBox.appendChild(exprInput);
  createBox.appendChild(el("label", { text: "Type d'expression" }));
  createBox.appendChild(labelSelect);
  createBox.appendChild(el("label", { text: "Définition en anglais" }));
  createBox.appendChild(defInput);
  createBox.appendChild(el("div", { class: "lexicon-warning", html:
    "⚠️ <b>Attention</b> : ajouter au lexique fait rechercher cette expression AUTOMATIQUEMENT " +
    "dans TOUS les prochains livres, sur la seule séquence de mots — sans distinguer le sens. " +
    "Pour un mot simple et courant (ex. « beat »), ça détournerait aussi son usage ordinaire " +
    "ailleurs. À réserver aux véritables expressions multi-mots (ex. « smart ass »)." }));
  const checkline = el("div", { class: "checkline" });
  checkline.appendChild(lexCheck);
  checkline.appendChild(el("label", { text: "ajouter au lexique d'expressions (profite aux prochains livres)" }));
  const lexUnlock = el("span", { class: "clear", text: "🔓 déverrouiller (je sais ce que je fais)" });
  lexUnlock.addEventListener("click", () => {
    lexCheck.removeAttribute("disabled");
    lexUnlock.style.display = "none";
  });
  checkline.appendChild(lexUnlock);
  createBox.appendChild(checkline);
  createBox.appendChild(useBtn);

  createToggle.addEventListener("click", () => createBox.classList.toggle("open"));
  useBtn.addEventListener("click", () => {
    const expr = (exprInput.value || "").trim().toLowerCase()
      .replace(/-/g, " ").replace(/\s+/g, " ");
    if (!expr) { exprInput.focus(); return; }
    const targetKey = "mwe:" + expr + ":" + labelSelect.value;
    createBox.querySelectorAll(".cand.selected").forEach(n => n.classList.remove("selected"));
    resultsBox.querySelectorAll(".cand.selected").forEach(n => n.classList.remove("selected"));
    state.redirect = {
      targetKey, definitionEnPerso: defInput.value || "",
      addToLexicon: lexCheck.checked, sourceLabel: "new",
    };
    onArmed();
  });

  panel.appendChild(createToggle);
  panel.appendChild(createBox);

  renderOwnInventory();
  return panel;
}

loadAll();
</script>
</body>
</html>
"""


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=config.REVIEW_UI_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not config.SENSE_FR_STORE_PATH.exists():
        print(f"Aucun magasin trouvé : {config.SENSE_FR_STORE_PATH}")
        return 1

    # Préchauffe WordNet (chargement paresseux NLTK, ~6s la première fois)
    # pour que la première recherche depuis le navigateur ne soit pas lente.
    from nltk.corpus import wordnet as nwn
    nwn.synsets("test")

    store = sense_fr.load_store()
    n_pending = sum(1 for e in store.values() if e["status"] == "pending")
    n_flagged = sum(
        1 for e in store.values()
        if e.get("status") in verify_fr_lock.LOCKED_STATUSES
        and e.get("sense_fit") == "mismatch" and e.get("decided_by") != "human"
    )
    if n_pending == 0 and n_flagged == 0:
        print("Aucune entrée en attente ni incohérence verrouillée dans data/sense_fr.jsonl — rien à relire.")
        return 0

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{n_pending} en attente + {n_flagged} verrouillée(s) incohérente(s) -> {url}  (Ctrl+C pour arrêter)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
