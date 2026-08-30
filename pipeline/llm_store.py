"""Magasin de résultats d'appel LLM, UNITAIRE — sous les magasins métier, jamais
mélangé avec eux (plan "décorréler l'appel en lot du stockage unitaire").

Les magasins métier (`data/mwe_occurrence_decisions.jsonl`, `data/sense_fr.jsonl`,
voir `pipeline/mwe_stores.py`/`pipeline/sense_fr.py`) restent la couche consultée
en premier et modifiable à la main (statuts `validated`/`auto_joint`) : ce module
n'y touche pas et n'en connaît rien.

Ce module est la couche EN DESSOUS : une réponse LLM brute par unité (occurrence,
sens, cluster...), jamais par lot. `pipeline/llm_client.py::cache_path_for` cache
un appel réseau entier — si ce lot mélange 50 unités dans un seul prompt, une
seule unité changée invalide les 49 autres. Ici, la clé ne porte jamais sur le
texte du prompt rendu ni sur la taille de lot : elle porte sur des valeurs
MÉTIER lisibles (`task_id`, `model`, `protocol`, `unit_id`) plus le hash d'une
charge sémantique (`payload_sig`) — indépendante de la composition du lot, de
l'ordre de présentation dans le prompt, et de sa mise en forme. Voir
`pipeline/llm_client.py::run_units` pour l'orchestrateur qui appelle en lot et
stocke ici en unitaire.

Colonnes hors clé (`batch_size`, `mode_batch`, `cost_usd`, `source`) : pure
observabilité, jamais lues pour décider d'un hit/miss — répondre à "ce résultat
vient d'un lot de combien ?" sans jamais influencer la lecture.

`payload` conserve le JSON canonique qui a produit `payload_sig` : c'est ce qui
rend un miss diagnosticable après coup (comparer l'ancien payload au nouveau pour
voir quel champ a changé), alors qu'un digest seul ne dit rien.

SQLite (stdlib, aucune dépendance nouvelle) : `data/llm_results.sqlite3`,
permanent comme le reste de `data/`, mais gitignored (binaire, pas de diff
lisible en revue — voir .gitignore)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from pipeline import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_results (
    task_id     TEXT NOT NULL,
    model       TEXT NOT NULL,
    protocol    TEXT NOT NULL,
    unit_id     TEXT NOT NULL,
    payload_sig TEXT NOT NULL,
    payload     TEXT NOT NULL,
    result      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    batch_size  INTEGER,
    mode_batch  INTEGER,
    cost_usd    REAL,
    source      TEXT,
    PRIMARY KEY (task_id, model, protocol, unit_id, payload_sig)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_llm_results_unit ON llm_results(unit_id);
CREATE INDEX IF NOT EXISTS idx_llm_results_task_model ON llm_results(task_id, model);
"""


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_sig(payload: dict) -> str:
    """Hash de l'entrée SÉMANTIQUE d'une unité — jamais du prompt rendu. Deux
    appelants qui construisent le même payload (même mots-clés, même valeurs)
    obtiennent la même signature quel que soit l'ordre de présentation choisi
    pour le prompt (voir `presentation_order`, dérivé de `unit_id`, jamais du
    payload)."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def connect() -> sqlite3.Connection:
    """Connexion neuve à chaque appel (pas de connexion globale mise en cache) :
    cohérent avec `pipeline/config.py` où chaque chemin peut être patché par les
    tests (voir `test_llm_client.py::LlmClientCacheIsolatedTests` pour le même
    principe appliqué à `CACHE_DIR`). `sqlite3.connect()` est bon marché ; ce
    module n'est jamais appelé par unité, seulement par lot, donc l'overhead
    d'une connexion par appel est négligeable."""
    config.ensure_data_dir()
    conn = sqlite3.connect(config.LLM_RESULTS_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


@dataclass
class ResultRow:
    """Une décision LLM UNITAIRE prête à être stockée. `payload` est la charge
    sémantique complète (pas déjà hashée) ; `put_many` calcule `payload_sig`."""
    task_id: str
    model: str
    protocol: str
    unit_id: str
    payload: dict
    result: dict
    batch_size: int | None = None
    mode_batch: bool | None = None
    cost_usd: float | None = None
    source: str = "live"


def get_many(
    *, task_id: str, model: str, protocol: str, wanted: list[tuple[str, str]],
) -> dict[str, dict]:
    """``wanted`` : liste de ``(unit_id, payload_sig)``. Renvoie
    ``{unit_id: result}`` pour les seules entrées dont la signature stockée
    correspond EXACTEMENT à celle demandée — une entrée avec le même
    ``unit_id`` mais un ``payload_sig`` différent (la phrase source a changé,
    par exemple) est un miss, pas un hit périmé silencieux."""
    if not wanted:
        return {}
    sig_by_unit = dict(wanted)
    unit_ids = list(sig_by_unit)
    conn = connect()
    try:
        placeholders = ",".join("?" * len(unit_ids))
        rows = conn.execute(
            f"SELECT unit_id, payload_sig, result FROM llm_results "
            f"WHERE task_id=? AND model=? AND protocol=? AND unit_id IN ({placeholders})",
            (task_id, model, protocol, *unit_ids),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict] = {}
    for unit_id, sig, result_json in rows:
        if sig_by_unit.get(unit_id) == sig:
            out[unit_id] = json.loads(result_json)
    return out


def put_many(rows: list[ResultRow]) -> int:
    """Une seule transaction pour tout le lot — y compris quand seule une
    partie des unités d'un appel LLM a réussi (l'appelant ne construit une
    ``ResultRow`` que pour les unités effectivement décidées ; les échecs
    n'entrent jamais ici, voir `pipeline/llm_client.py::run_units`)."""
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    values = []
    for r in rows:
        payload_json = _canonical_json(r.payload)
        sig = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        values.append((
            r.task_id, r.model, r.protocol, r.unit_id, sig, payload_json,
            json.dumps(r.result, ensure_ascii=False), now,
            r.batch_size, None if r.mode_batch is None else int(bool(r.mode_batch)),
            r.cost_usd, r.source,
        ))
    conn = connect()
    try:
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO llm_results "
                "(task_id, model, protocol, unit_id, payload_sig, payload, result, "
                "created_at, batch_size, mode_batch, cost_usd, source) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
    finally:
        conn.close()
    return len(values)


def stats(*, task_id: str | None = None, model: str | None = None) -> list[dict]:
    """Comptage par ``(task_id, model)``, pour l'inspection en CLI — jamais
    utilisé pour décider d'un hit/miss."""
    clauses = []
    params: list[str] = []
    if task_id is not None:
        clauses.append("task_id = ?")
        params.append(task_id)
    if model is not None:
        clauses.append("model = ?")
        params.append(model)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = connect()
    try:
        rows = conn.execute(
            f"SELECT task_id, model, protocol, COUNT(*) AS n FROM llm_results "
            f"{where} GROUP BY task_id, model, protocol ORDER BY n DESC",
            params,
        ).fetchall()
    finally:
        conn.close()
    return [
        {"task_id": t, "model": m, "protocol": p, "count": n}
        for t, m, p, n in rows
    ]
