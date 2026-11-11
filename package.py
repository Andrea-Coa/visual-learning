"""Builds the Canvas submission ZIP (PLAN 8.4): Flask app + static site + data files (nodes and edges) + code + docs.

Usage:
    python package.py             # -> dist/jamona.zip

Excluded: training outputs (raw/), datasets (data/), the paper PDF, UI drafts, virtual environments and caches.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dist" / "jamona.zip"
INCLUDE = ["app.py", "app", "pipeline", "notebooks", "README.md", "PLAN.md", "VIS_PLAN.md", "PRESENTATION.md",
           "task-description.md", "requirements.txt", "package.py"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}


def files():
    for name in INCLUDE:
        path = ROOT / name
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file() and not SKIP_PARTS & set(p.parts))


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files():
            zf.write(path, Path("jamona") / path.relative_to(ROOT))
            count += 1
    print(f"{OUT.relative_to(ROOT)}: {count} files, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
