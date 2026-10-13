"""
Copy the canonical Skill into the Claude Code plugin and build the Claude.ai upload ZIP.

    python scripts/build_skill.py          # sync plugins/sozograph/skills and write build/sozograph-passport.zip

Claude.ai accepts one skill per ZIP with a single SKILL.md inside one top-level folder.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "sozograph-passport"
PLUGIN_COPY = ROOT / "plugins" / "sozograph" / "skills" / "sozograph-passport"
ZIP = ROOT / "build" / "sozograph-passport.zip"


def files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file())


def main(check: bool = False) -> int:
    if check:
        mine = {p.relative_to(SKILL): p.read_bytes() for p in files(SKILL)}
        theirs = {p.relative_to(PLUGIN_COPY): p.read_bytes() for p in files(PLUGIN_COPY)} if PLUGIN_COPY.exists() else {}
        if mine != theirs:
            print("plugin skill copy is out of date: run python scripts/build_skill.py", file=sys.stderr)
            return 1
        return 0
    if PLUGIN_COPY.exists():
        shutil.rmtree(PLUGIN_COPY)
    shutil.copytree(SKILL, PLUGIN_COPY)
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files(SKILL):
            archive.write(path, Path("sozograph-passport") / path.relative_to(SKILL))
    print(f"synced {PLUGIN_COPY.relative_to(ROOT)} and wrote {ZIP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check="--check" in sys.argv))
