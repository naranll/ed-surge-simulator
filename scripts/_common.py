"""Small helpers shared by the training / run scripts (pandas 2.x and 3.x safe)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
NHAMCS_CLEAN = ROOT / "data" / "processed" / "nhamcs_2022_cleaned.csv"


def update_json(path: Path, key: str, value) -> None:
    """Write value under key in a JSON file, keeping the other keys (never deletes anything else)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8")


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def update_md_section(path: Path, name: str, text: str) -> None:
    """Replace (or append) the block between <!-- name:start --> and <!-- name:end --> in a markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    block = f"{start}\n{text.strip()}\n{end}"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in old and end in old:
        head, rest = old.split(start, 1)
        tail = rest.split(end, 1)[1]
        new = head + block + tail
    else:
        new = (old.rstrip() + "\n\n" if old.strip() else "") + block + "\n"
    path.write_text(new, encoding="utf-8")
