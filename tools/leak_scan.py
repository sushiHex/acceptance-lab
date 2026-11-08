"""leak_scan.py: refuse to publish anything that carries private facts.

Scans exactly what a push would publish, read from git, never from the working tree:
  1. every blob staged in the index (what the next commit will contain),
  2. every line added in every commit reachable from any ref (history is published too),
  3. every tracked path name.
Binary blobs are refused unless their extension is on BINARY_OK. Marker patterns come from a
private JSON file passed by path (--markers or LEAK_SCAN_MARKERS) and are never stored here;
built-in checks add local user paths, requisition-style ids, common secret shapes and em-dashes.
Exit 1 on any hit or on any blob it cannot read. Hits are reported by file, never by content.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

BUILTIN = {
    "local user path": r"(?i)[a-z]:[\\/]+users[\\/]+\w+|/home/\w+/|/Users/\w+/",
    "requisition id": r"\bJR\d{5,}\b|\bR\d{6,}\b",
    "secret: nvidia key": r"nvapi-[A-Za-z0-9_\-]{20,}",
    "secret: openai-style key": r"\bsk-[A-Za-z0-9]{20,}",
    "secret: github token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "secret: aws key": r"\bAKIA[0-9A-Z]{16}\b",
    "secret: private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "secret: hf token": r"\bhf_[A-Za-z0-9]{30,}",
    "em-dash": chr(0x2014),
}
BINARY_OK = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def load_markers(path: str | None) -> dict:
    if not path:
        sys.exit("leak_scan: no marker file (pass --markers or set LEAK_SCAN_MARKERS); refusing to pass")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    markers = data.get("markers", data)
    out = {}
    for name, pat in markers.items():
        if pat:
            out[f"marker:{name}"] = re.compile(pat, 0 if name == "identity" else re.IGNORECASE)
    if not out:
        sys.exit("leak_scan: marker file holds no patterns; refusing to pass")
    return out


def git_bytes(*args) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def git(*args) -> str:
    return git_bytes(*args).decode("utf-8", errors="replace")


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8000]


def scan_text(label: str, text: str, checks: dict, hits: list) -> None:
    for name, rx in checks.items():
        if rx.search(text):
            hits.append(f"{name}: {label}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markers", default=os.environ.get("LEAK_SCAN_MARKERS"))
    args = ap.parse_args()
    checks = {k: re.compile(v) for k, v in BUILTIN.items()} | load_markers(args.markers)
    hits: list[str] = []

    # 1. index blobs and 3. path names
    for line in git("ls-files", "-s", "-z").split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        sha = meta.split()[1]
        scan_text(f"path {path}", path, checks, hits)
        data = git_bytes("cat-file", "blob", sha)
        if is_binary(data):
            if os.path.splitext(path)[1].lower() not in BINARY_OK:
                hits.append(f"binary not allowlisted: {path}")
            continue
        scan_text(f"staged {path}", data.decode("utf-8", errors="replace"), checks, hits)

    # 2. every added line in every reachable commit, plus commit messages and authors
    has_head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], capture_output=True).returncode == 0
    if has_head:
        log = git("log", "--all", "-p", "--no-color", "--format=COMMIT %H%n%an <%ae>%n%B")
        current = "commit metadata"
        for line in log.splitlines():
            if line.startswith("COMMIT "):
                current = f"commit {line[7:19]} metadata"
            elif line.startswith("+++ b/"):
                current = f"history {line[6:]}"
            elif line.startswith("+") and not line.startswith("+++"):
                scan_text(current, line[1:], checks, hits)
            elif current.endswith("metadata"):
                scan_text(current, line, checks, hits)

    for h in sorted(set(hits)):
        print(f"HIT {h}")
    print(f"leak_scan: {len(set(hits))} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
