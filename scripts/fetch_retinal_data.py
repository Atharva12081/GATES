from __future__ import annotations

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "retinal_sources.json"
CHUNK_SIZE = 1024 * 1024
DOWNLOAD_PARTS = 8


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


def download_part(url: str, destination: Path, start: int, end: int) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "gates-ai4s/0.1 data fetcher",
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        if response.status != 206:
            raise ValueError(f"source did not honor byte range {start}-{end}")
        while chunk := response.read(CHUNK_SIZE):
            out.write(chunk)
    expected = end - start + 1
    if destination.stat().st_size != expected:
        raise ValueError(f"incomplete byte range {start}-{end}")


def download(record: dict[str, object], partial: Path) -> None:
    size = int(record["size_bytes"])
    part_size = (size + DOWNLOAD_PARTS - 1) // DOWNLOAD_PARTS
    jobs = []
    part_paths = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_PARTS) as executor:
        for index in range(DOWNLOAD_PARTS):
            start = index * part_size
            end = min(size - 1, start + part_size - 1)
            part_path = partial.with_suffix(partial.suffix + f".{index}")
            part_paths.append(part_path)
            jobs.append(
                executor.submit(
                    download_part, str(record["download_url"]), part_path, start, end
                )
            )
        for job in jobs:
            job.result()
    with partial.open("wb") as out:
        for part_path in part_paths:
            with part_path.open("rb") as source:
                while chunk := source.read(CHUNK_SIZE):
                    out.write(chunk)
            part_path.unlink()


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
            download(record, partial)
            validate(partial, record)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            for part_path in partial.parent.glob(partial.name + ".*"):
                part_path.unlink(missing_ok=True)
            raise
        print(f"downloaded and verified {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
