from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".vscode",
    ".idea",
    ".venv",
    "venv",
    "env",
}

FORBIDDEN_DIR_PATHS = {
    Path("experiments/runs"),
    Path("runs"),
}

FORBIDDEN_FILE_PATTERNS = (
    re.compile(r"^\.env(?:\..*)?$"),
    re.compile(r".*\.py[cod]$"),
    re.compile(r".*\.(?:log|tmp|bak|pem|key|p12|pfx)$", re.IGNORECASE),
    re.compile(r"^(?:credentials|token).*\.json$", re.IGNORECASE),
)

ALLOW_FILE_NAMES = {".env.example"}

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password|credential)[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9_./+=-]{12,}"),
    re.compile(r"(?i)authorization[ \t]*[:=][ \t]*['\"]?bearer[ \t]+[A-Za-z0-9_./+=-]{12,}"),
)

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".json",
    ".txt",
    ".html",
    ".css",
    ".js",
    ".toml",
    ".yaml",
    ".yml",
    ".example",
    ".gitignore",
}


def rel(path: Path) -> Path:
    return path.relative_to(ROOT)


def is_inside(path: Path, maybe_parent: Path) -> bool:
    try:
        path.relative_to(ROOT / maybe_parent)
    except ValueError:
        return False
    return True


def should_scan_text(path: Path) -> bool:
    if path.name in {".gitignore", ".env.example"}:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def iter_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def add_unique(problems: list[str], seen: set[str], message: str) -> None:
    if message not in seen:
        seen.add(message)
        problems.append(message)


def check_forbidden_paths(files: list[Path]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for path in files:
        relative = rel(path)
        if any(part in FORBIDDEN_DIR_NAMES for part in relative.parts[:-1]):
            cache_dir = next(part for part in relative.parts[:-1] if part in FORBIDDEN_DIR_NAMES)
            add_unique(problems, seen, f"forbidden cache/local directory present: {cache_dir}")
            continue
        matching_artifact = next((forbidden for forbidden in FORBIDDEN_DIR_PATHS if is_inside(path, forbidden)), None)
        if matching_artifact is not None:
            add_unique(problems, seen, f"large artifact directory should not be uploaded: {matching_artifact.as_posix()}")
            continue
        if path.name in ALLOW_FILE_NAMES:
            continue
        if any(pattern.match(path.name) for pattern in FORBIDDEN_FILE_PATTERNS):
            problems.append(f"forbidden file: {relative.as_posix()}")
    return problems


def check_large_files(files: list[Path], max_bytes: int) -> list[str]:
    problems: list[str] = []
    for path in files:
        if any(is_inside(path, forbidden) for forbidden in FORBIDDEN_DIR_PATHS):
            continue
        if path.stat().st_size > max_bytes:
            problems.append(f"large file over {max_bytes} bytes: {rel(path).as_posix()} ({path.stat().st_size} bytes)")
    return problems


def check_secrets(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        if any(is_inside(path, forbidden) for forbidden in FORBIDDEN_DIR_PATHS):
            continue
        if path.name == ".env":
            problems.append(".env exists locally and must not be uploaded")
            continue
        if path.name in ALLOW_FILE_NAMES:
            continue
        if not should_scan_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            problems.append(f"could not read {rel(path).as_posix()}: {exc}")
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                problems.append(f"possible secret in {rel(path).as_posix()}: pattern {pattern.pattern}")
                break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Check WorkflowPilot release files before publishing to GitHub.")
    parser.add_argument("--max-file-mb", type=float, default=5.0, help="Maximum allowed file size outside ignored artifacts.")
    parser.add_argument("--max-output", type=int, default=80, help="Maximum number of problems to print.")
    args = parser.parse_args()

    files = iter_files()
    max_bytes = int(args.max_file_mb * 1024 * 1024)
    problems = []
    problems.extend(check_forbidden_paths(files))
    problems.extend(check_large_files(files, max_bytes))
    problems.extend(check_secrets(files))

    if problems:
        print("Release check failed:")
        for problem in problems[: args.max_output]:
            print(f"- {problem}")
        if len(problems) > args.max_output:
            print(f"- ... {len(problems) - args.max_output} more problems omitted")
        print(f"Total problems: {len(problems)}")
        return 1

    print("Release check passed: no forbidden files, large artifacts, caches, or obvious secrets found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
