"""Mandatory offset check for the_humans_gold_v0.jsonl — independent of
_build_gold.py's own inline check. Reloads segment text directly from
pipeline.corpus.load_segments() (the real production segmentation path,
not the intermediate _selected_segments.jsonl dump) and asserts
text[start_char:end_char] == surface for every gold span. Run:
python fix_pipeline/gold_corpus/verify_offsets.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.corpus import load_segments  # noqa: E402

segments = load_segments()
play_segments = {s.idx: s for s in segments if s.kind != "hors_oeuvre"}

gold_path = Path(__file__).resolve().parent / "the_humans_gold_v0.jsonl"
records = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]

errors = []
n_spans = 0
for rec in records:
    idx = rec["segment_idx"]
    if idx not in play_segments:
        errors.append(f"segment_idx {idx}: not found via pipeline.corpus.load_segments()")
        continue
    live_text = play_segments[idx].en
    if live_text != rec["text"]:
        errors.append(f"segment_idx {idx}: text drift between gold file and live segmentation\n"
                       f"  gold: {rec['text']!r}\n  live: {live_text!r}")
        continue
    for s in rec["gold_spans"]:
        n_spans += 1
        actual = live_text[s["start_char"]:s["end_char"]]
        if actual != s["surface"]:
            errors.append(f"segment_idx {idx}: offset mismatch — expected {s['surface']!r}, "
                           f"got {actual!r} at [{s['start_char']}:{s['end_char']}]")

if errors:
    print(f"{len(errors)} ERROR(S) out of {n_spans} spans checked across {len(records)} segments:")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

print(f"OK — {n_spans} spans across {len(records)} segments, all offsets verified "
      f"against the live pipeline.corpus.load_segments() text.")
