#!/usr/bin/env python3
"""Fail fast when a repository is unsafe or incomplete for public release."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git"}
FORBIDDEN_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".smoke_data",
    ".smoke_logs",
    ".smoke_logs_final",
    ".smoke_logs_multihead",
    "tmp",
}
FORBIDDEN_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".bin",
    ".h5", ".hdf5", ".npy", ".npz", ".pkl", ".pickle",
}
TEXT_SUFFIXES = {
    ".py", ".sh", ".ps1", ".md", ".txt", ".json", ".yaml", ".yml", ".cff",
}
PRIVATE_MARKERS = (
    "/home/" + "ubuntu/wzy",
    "/home/" + "s1024244030",
    "/hpai/" + "aios3.0/private/user",
    "C:" + "\\Users\\wang",
)
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
MAX_FILE_BYTES = 25 * 1024 * 1024


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    errors: list[str] = []
    checked = 0

    for path in ROOT.rglob("*"):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in IGNORED_DIRS for part in rel_parts):
            continue
        if any(part in FORBIDDEN_DIRS for part in rel_parts):
            errors.append(f"forbidden transient path: {relative(path)}")
            continue
        if not path.is_file():
            continue
        checked += 1
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"data/checkpoint file: {relative(path)}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(
                f"large file ({path.stat().st_size / 1024 / 1024:.1f} MiB): {relative(path)}"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in PRIVATE_MARKERS:
            if marker in content:
                errors.append(f"private absolute path in {relative(path)}: {marker}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"possible {label} in {relative(path)}")

    for path in ROOT.rglob("*.py"):
        if any(
            part in IGNORED_DIRS or part in FORBIDDEN_DIRS
            for part in path.relative_to(ROOT).parts
        ):
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"Python compile failure in {relative(path)}: {exc}")

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    known_options = set(
        re.findall(r"parser\.add_argument\(\s*['\"](--[A-Za-z0-9_-]+)['\"]", main_source)
    )
    paper_scripts = (
        "scripts/train_paper_evt6.sh",
        "scripts/eval_paper_evt6.sh",
        "scripts/train_paper_evt6.ps1",
        "scripts/eval_paper_evt6.ps1",
    )
    for name in paper_scripts:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"missing paper entry point: {name}")
            continue
        used_options = set(re.findall(r"--[A-Za-z0-9_-]+", path.read_text(encoding="utf-8")))
        for option in sorted(used_options - known_options):
            errors.append(f"unknown main.py option in {name}: {option}")

    for path in ROOT.rglob("*.md"):
        if any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            linked = (path.parent / target).resolve()
            if not linked.exists():
                errors.append(f"broken local Markdown link in {relative(path)}: {target}")

    if errors:
        print("Public-release audit failed:", file=sys.stderr)
        for item in sorted(set(errors)):
            print(f"- {item}", file=sys.stderr)
        return 1

    print(f"Public-release audit passed: {checked} files checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
