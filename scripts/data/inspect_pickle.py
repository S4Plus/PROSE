#!/usr/bin/env python3
"""Inspect the structure of one local rollout pickle."""
import argparse
from pathlib import Path
import pickle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Rollout .pkl file to inspect.")
    args = parser.parse_args()
    print(f"size(MB) = {args.path.stat().st_size / 1024 / 1024:.3f}")
    with args.path.open("rb") as stream:
        obj = pickle.load(stream)
    print("type:", type(obj).__name__)
    if isinstance(obj, dict):
        print("dict keys:", list(obj)[:50])
        for key, value in list(obj.items())[:5]:
            print(f"[{key}] type={type(value).__name__}", getattr(value, "shape", ""))
    elif isinstance(obj, (list, tuple)):
        print("length:", len(obj))
        if obj:
            print("first element type:", type(obj[0]).__name__)
    else:
        print("repr:", repr(obj)[:500])


if __name__ == "__main__":
    main()
