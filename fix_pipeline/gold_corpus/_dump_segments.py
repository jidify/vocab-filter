"""Helper (throwaway) — dump play_segments (kind != hors_oeuvre) from the
exact production segmentation path (pipeline.corpus.load_segments), so the
gold corpus can be sampled/annotated against real segment_idx/text without
re-deriving segmentation by hand. Not part of the pipeline; not imported
anywhere. Run: python fix_pipeline/gold_corpus/_dump_segments.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.corpus import load_segments  # noqa: E402

segments = load_segments()
play_segments = [s for s in segments if s.kind != "hors_oeuvre"]

out_path = Path(__file__).resolve().parent / "_segments_dump.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for s in play_segments:
        f.write(json.dumps({"idx": s.idx, "en": s.en, "kind": s.kind, "speaker": s.speaker},
                            ensure_ascii=False) + "\n")

print(f"{len(play_segments)} play_segments -> {out_path}")
