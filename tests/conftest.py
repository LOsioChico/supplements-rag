"""pytest path setup — one place instead of a shim per test file.

src/     → the supplements_rag package (works without `pip install -e .`)
evals/   → eval scripts imported as top-level modules (admit, run, verify)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

# the suite exercises the writer path; extractive-lane tests opt back in
os.environ.setdefault("EXTRACTIVE", "0")
