from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "organoid_sources.json"
RAW = ROOT / "data" / "raw"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for name, url in manifest["files"].items():
        destination = RAW / name
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        destination.write_bytes(payload)
        checksums[name] = hashlib.sha256(payload).hexdigest()
        print(f"downloaded {name}: {len(payload):,} bytes")
    manifest["checksums"] = checksums
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
