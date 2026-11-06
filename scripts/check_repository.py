#!/usr/bin/env python3
"""Check maintained Python entry points, migration destinations and documentation."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/architecture.md", "docs/environments.md", "docs/reproduction.md",
        "docs/migration.md", "docs/validation.md", "act/README.md", "scripts/README.md"]
PROFILES = {
    "act": ("torch", "torchvision", "lerobot", "traker", "numpy", "pyarrow", "pandas", "h5py", "scipy", "scikit-learn", "matplotlib"),
    "simulation": ("robosuite", "robomimic", "mimicgen", "mujoco", "numpy", "imageio"),
    "openpi": ("openpi", "jax", "flax", "torch", "openpi-client"),
    "openvla": ("openvla-oft", "torch", "transformers", "peft", "tensorflow"),
}


def check_structure() -> tuple[list[str], int]:
    errors = []
    files = list((ROOT / "prose").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    files += [ROOT / path for path in ["setup.py", "hf_datasets_compat.py", "act/train_trak_act.py",
              "act/myscripts/cupid_act_influence.py", "act/myscripts/filter_joint_episode_window.py",
              "act/train_act_cupid_reweight_joint.py"]]
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(str(exc))
    for old, new in json.loads((ROOT / "docs/migration.json").read_text(encoding="utf-8")).items():
        if not (ROOT / new).exists():
            errors.append(f"Missing migration destination: {old} -> {new}")
    for name in DOCS:
        path = ROOT / name
        if not path.exists():
            errors.append(f"Missing documentation: {name}")
            continue
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding="utf-8")):
            target = link.split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / unquote(target)).exists():
                errors.append(f"Broken local link in {name}: {link}")
    return errors, len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, help="Also report required installed distributions.")
    args = parser.parse_args()
    errors, count = check_structure()
    print(f"Checked {count} maintained Python files, migration paths, and {len(DOCS)} documents.")
    if args.profile:
        for package in PROFILES[args.profile]:
            try:
                print(f"{package}=={importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"Missing package for {args.profile}: {package}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Checks passed. This does not verify GPU training, dependency compatibility, or reproduction metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
