"""BassLift.app instaluje zależności z pyproject.toml, a launcher z requirements.txt — muszą się zgadzać."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def names(lines):
    return {re.split(r"[<>=\[;~!]", l.strip())[0].lower().replace("_", "-")
            for l in lines if l.strip() and not l.strip().startswith("#")}


def test_app_requires_match_requirements_txt():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    app = pyproject["tool"]["briefcase"]["app"]["basslift"]
    assert names(app["requires"]) == names((ROOT / "requirements.txt").read_text().splitlines())


def test_app_sources_exist():
    app = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["briefcase"]["app"]["basslift"]
    assert all((ROOT / s).exists() for s in app["sources"])
