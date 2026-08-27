"""Lot 0 — Écritures atomiques et verrou de run.

Corrige la corruption observée sur `pipeline_out/senses.jsonl` (2 lignes
tronquées/entrelacées + 329 doublons) : le motif exact — des lignes JSON
coupées au milieu et concaténées à une autre, plus un nombre de doublons
proche de l'écart entre lignes physiques et clés distinctes — est la
signature de DEUX processus ayant chacun ouvert le même fichier en mode
"w" (troncature, pas verrouillage) et écrit concurremment dedans. Un seul
run ne peut pas produire ce motif : `senses.py::run()` écrit chaque ligne
une fois, dans l'ordre, sans jamais relire ce qu'il vient d'écrire.

Deux protections indépendantes, l'une n'excusant pas l'absence de l'autre :

- `atomic_write_jsonl`/`atomic_write_text` : le fichier final n'existe
  qu'une fois COMPLET (écriture dans un fichier temporaire du même
  dossier, puis `os.replace()` atomique sur la plupart des systèmes de
  fichiers). Ça élimine l'entrelacement au niveau octet même si deux runs
  se chevauchent — le dernier `os.replace()` gagne proprement, sans jamais
  produire un fichier à moitié écrit.
- `run_lock()` : empêche qu'un second run parte pour de vrai pendant qu'un
  premier tourne encore, pour éviter le gaspillage de calcul (le run
  perdant a quand même consommé des heures de GlossBERT/LLM pour rien).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from pipeline import config


@contextmanager
def atomic_open(path: Path, mode: str = "w", encoding: str | None = "utf-8", newline=None):
    """Ouvre un fichier temporaire dans le même dossier que `path`, à
    utiliser comme n'importe quel fichier ouvert (y compris avec
    `csv.writer`) ; ne remplace `path` que si le bloc `with` se termine
    sans exception. Base commune de `atomic_write_text`/`atomic_write_jsonl`
    et des cas où l'appelant a besoin d'un vrai objet fichier (csv.DictWriter)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp{os.getpid()}")
    f = tmp_path.open(mode, encoding=encoding, newline=newline)
    try:
        yield f
        f.flush()
        os.fsync(f.fileno())
        f.close()
        os.replace(tmp_path, path)  # atomique sur NTFS/ext4/APFS pour un même volume
    except BaseException:
        f.close()
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Écrit `text` dans `path` de façon atomique : jamais de fichier à
    moitié écrit visible par un autre processus, même en cas de crash ou
    de run concurrent."""

    with atomic_open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def atomic_write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Sérialise `records` en JSON Lines et écrit atomiquement. Retourne
    le nombre de lignes écrites."""

    n = 0
    with atomic_open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


class RunLockError(RuntimeError):
    pass


@contextmanager
def run_lock(stale_after_seconds: float = config.LOCK_STALE_SECONDS):
    """Empêche deux runs de `run_pipeline.py` de tourner en même temps sur
    le même `pipeline_out/`.

    Windows ne permet pas de vérifier fiablement si un PID est encore
    vivant (`os.kill(pid, 0)` renvoie systématiquement WinError 87, vérifié
    empiriquement, y compris pour un PID inexistant) — le verrou se fonde
    donc sur l'ÂGE du fichier, pas sur la vivacité du process : un verrou
    plus vieux que `stale_after_seconds` (par défaut : voir config) est
    considéré abandonné (crash, Ctrl+C) et écrasé avec un avertissement.
    """

    config.ensure_out_dir()
    lock_path = config.LOCK_PATH

    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < stale_after_seconds:
            info = lock_path.read_text(encoding="utf-8")
            raise RunLockError(
                f"Un autre run semble actif ({lock_path}, âgé de {age:.0f}s) :\n{info}"
                f"\nSi ce run est bien terminé (crash, Ctrl+C), supprime {lock_path} "
                f"ou attends {stale_after_seconds:.0f}s."
            )
            # (age >= stale_after_seconds : verrou abandonné, on l'écrase ci-dessous)

    atomic_write_text(
        lock_path,
        f"pid={os.getpid()}\nstarted_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
    )
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
