"""Project paths (no torch)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS_JSON = ROOT / "data" / "paths.json"   # where the dataset lives (written by step1_download.py)
INDEX_DIR = ROOT / "data" / "index"         # manifest, stats, plots (written by step2_clean_and_index.py)


def load_paths():
    if not PATHS_JSON.exists():
        raise SystemExit("data/paths.json not found: run step1_download.py first.")
    return json.loads(PATHS_JSON.read_text(encoding="utf-8"))
