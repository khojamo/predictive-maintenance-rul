#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"


def _ensure_python_version() -> None:
    if sys.version_info < (3, 11):
        print(
            f"Python 3.11+ is required. Detected {sys.version_info.major}.{sys.version_info.minor}.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _prompt_yes_no(question: str, default: bool) -> bool:
    if not sys.stdin.isatty():
        return default
    prompt = " [Y/n]" if default else " [y/N]"
    while True:
        ans = input(f"{question}{prompt} ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(venv_dir: Path) -> Path:
    py = _venv_python(venv_dir)
    if not py.exists():
        print(f"Creating virtual environment at {venv_dir}")
        venv.create(venv_dir, with_pip=True)
    return py


def _project_info(extras: str) -> tuple[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project", {})
    name = str(project.get("name", "")).strip()
    reqs = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    if extras:
        reqs.extend(optional.get(extras, []))
    return name, reqs


def _project_installed(project_name: str) -> bool:
    if not project_name:
        return False
    try:
        metadata.distribution(project_name)
        return True
    except metadata.PackageNotFoundError:
        return False


def _missing_requirements(requirements: list[str]) -> list[str] | None:
    try:
        from packaging.requirements import Requirement
    except Exception:
        return None

    missing: list[str] = []
    for req_str in requirements:
        req = Requirement(req_str)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            missing.append(req_str)
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            missing.append(req_str)
    return missing


def _ensure_deps(py: Path, extras: str, skip_install: bool, check_installed: bool) -> None:
    if skip_install:
        return
    pkg = f".[{extras}]" if extras else "."
    project_name, requirements = _project_info(extras)

    if check_installed:
        missing = _missing_requirements(requirements)
        project_ok = _project_installed(project_name)
        if missing is not None and project_ok and not missing:
            print("All required dependencies are already installed.")
            return
        if missing is None:
            print("Could not verify installed dependencies; installing...")
        else:
            if not project_ok:
                missing.append(project_name or "project")
            if missing:
                print("Missing dependencies detected:")
                for req in missing:
                    print(f" - {req}")

    print("Installing dependencies...")
    subprocess.check_call([str(py), "-m", "pip", "install", "-e", pkg], cwd=ROOT)


def _run_api(py: Path, extra_args: list[str]) -> int:
    cmd = [
        str(py),
        "-m",
        "uvicorn",
        "predictive_maintenance.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=ROOT)


def _run_ui(py: Path, extra_args: list[str]) -> int:
    app_path = ROOT / "src" / "predictive_maintenance" / "ui" / "app.py"
    cmd = [str(py), "-m", "streamlit", "run", str(app_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    _ensure_python_version()

    parser = argparse.ArgumentParser(
        description="Launch the predictive maintenance app with auto-installed dependencies."
    )
    parser.add_argument("command", choices=["api", "ui"], help="Which service to run.")
    parser.add_argument(
        "--venv",
        default=str(ROOT / ".venv"),
        help="Virtual environment directory (default: .venv).",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip dependency installation (assumes venv is already ready).",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to uvicorn/streamlit after '--'.",
    )
    args = parser.parse_args()

    extra_args = args.args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    venv_dir = Path(args.venv).expanduser()
    create_venv = _prompt_yes_no(
        f"Create and use a virtual environment at {venv_dir}?", True
    )
    if create_venv:
        py = _ensure_venv(venv_dir)
    else:
        py = Path(sys.executable)

    extras = "api" if args.command == "api" else "app"
    _ensure_deps(py, extras, args.no_install, check_installed=not create_venv)

    if args.command == "api":
        return _run_api(py, extra_args)
    return _run_ui(py, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
