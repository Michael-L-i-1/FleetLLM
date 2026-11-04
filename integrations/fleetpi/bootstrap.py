#!/usr/bin/env python3
"""Reproduce the pinned Pi fork and offline catalog; never downloads model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent

def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--patch", type=Path, help="Use the frozen trial patch in a fresh checkout")
    args = parser.parse_args()
    pin = json.loads((HERE / "UPSTREAM.json").read_text())
    target = HERE / "vendor/pi"
    patch = args.patch.resolve() if args.patch else HERE / "fleetpi.patch"
    if not patch.exists():
        raise SystemExit("Missing fleetpi.patch; integration source export is incomplete")
    expected = json.loads((HERE / "model-data.json").read_text())["sha256"]
    archive = HERE / "model-data.tar.gz"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise SystemExit("Model catalog archive checksum mismatch")
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--no-checkout", pin["repository"], str(target))
        run("git", "checkout", "-b", pin["branch"], pin["commit"], cwd=target)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    if head != pin["commit"]:
        raise SystemExit(f"Refusing to modify fork at unexpected HEAD {head}; expected {pin['commit']}")
    # A reverse check recognizes an already-applied identical patch without clobbering edits.
    applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=target,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not applied:
        run("git", "apply", "--check", str(patch), cwd=target)
        run("git", "apply", str(patch), cwd=target)
    data = target / "packages/ai/src/providers/data"
    data.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data, filter="data")
    if not args.skip_install:
        run("npm", "ci", "--ignore-scripts", cwd=target)
        run("npm", "run", "build:offline", cwd=target)
    print(f"FleetPi ready: {target / 'packages/fleetpi/dist/cli.js'}")

if __name__ == "__main__":
    main()
