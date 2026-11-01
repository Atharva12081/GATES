from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/gates_act_v3_sources.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for source in manifest["sources"]:
        path = ROOT / source["local_path"]
        if not path.exists():
            raise FileNotFoundError(f"missing source file: {path}")
        size = path.stat().st_size
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if size != source["bytes"]:
            raise ValueError(f"size mismatch for {path}: {size} != {source['bytes']}")
        if digest != source["sha256"]:
            raise ValueError(f"hash mismatch for {path}: {digest}")
        print(f"verified {source['name']}: {digest}")


if __name__ == "__main__":
    main()
