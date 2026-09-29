from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "retinal_sources.json"
CHUNK_SIZE = 1024 * 1024


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, record: dict[str, object]) -> None:
    expected_size = int(record["size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"size mismatch for {path}: {path.stat().st_size} != {expected_size} bytes"
        )
    for algorithm in ("md5", "sha256"):
        observed = file_digest(path, algorithm)
        expected = str(record[algorithm])
        if observed != expected:
            raise ValueError(f"{algorithm} mismatch for {path}: {observed} != {expected}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    for record in manifest["files"]:
        destination = ROOT / record["destination"]
        if destination.exists():
            validate(destination, record)
            print(f"verified existing {destination.relative_to(ROOT)}")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.exists():
            partial.unlink()
        try:
            request = urllib.request.Request(
                record["download_url"], headers={"User-Agent": "gates-ai4s/0.1 data fetcher"}
            )
            response = urllib.request.urlopen(request, timeout=120)
            with response, partial.open("wb") as out:
                while chunk := response.read(CHUNK_SIZE):
                    out.write(chunk)
            validate(partial, record)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        print(f"downloaded and verified {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
