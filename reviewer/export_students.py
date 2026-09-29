"""Export only approved public files, with no repository history."""

import argparse
from pathlib import Path
import shutil
from physical_ai.scenes import ROOT

FILES = [
    "ASSIGNMENT.md",
    "MUJOCO_GUIDE.md",
    "requirements.txt",
    "requirements-lock.txt",
    "pyproject.toml",
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    ".gitattributes",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "REPORT_TEMPLATE.md",
    "submission.example.json",
]
DIRS = ["physical_ai", "assets", "scenes", "scripts", "configs"]


def export_students(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if (
        destination == ROOT
        or ROOT in destination.parents
        and destination.parts[len(ROOT.parts)] not in ("dist",)
    ):
        raise ValueError("Export outside repository or under dist/")
    destination.mkdir(parents=True)
    try:
        for name in FILES:
            shutil.copy2(ROOT / name, destination / name)
        for name in DIRS:
            shutil.copytree(
                ROOT / name,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        (destination / "tests").mkdir()
        for name in [
            "test_environment.py",
            "test_pipeline.py",
            "test_curriculum.py",
            "test_lerobot.py",
        ]:
            shutil.copy2(ROOT / "tests" / name, destination / "tests" / name)
        readme = (ROOT / "README.md").read_text().split("<!-- ORGANIZER_ONLY -->", 1)[0]
        (destination / "README.md").write_text(readme.rstrip() + "\n")
        (destination / ".github/workflows").mkdir(parents=True)
        shutil.copy2(
            ROOT / ".github/workflows/tests.yml",
            destination / ".github/workflows/tests.yml",
        )
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("destination")
    p.add_argument("--zip", action="store_true")
    a = p.parse_args()
    dest = export_students(a.destination)
    if a.zip:
        print(
            shutil.make_archive(
                str(dest), "zip", root_dir=dest.parent, base_dir=dest.name
            )
        )
    else:
        print(dest)
