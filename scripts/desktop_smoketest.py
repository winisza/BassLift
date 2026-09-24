"""Test dymny natywnego okna z repo (logika w basslift/smoketest.py).

    .venv/bin/python scripts/desktop_smoketest.py wynik.json
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["BASSLIFT_SMOKETEST"] = str(Path(sys.argv[1] if len(sys.argv) > 1 else "desktop_smoketest.json").resolve())

from basslift.desktop import main  # noqa: E402

main()
