from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "neural_mea_sources.json"
RAW = ROOT / "data" / "neural_mea_raw"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / manifest["archive"]["filename"]
    if not archive.exists():
        with (
            urllib.request.urlopen(manifest["download_url"]) as response,
            archive.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
    if _sha256(archive) != manifest["archive"]["sha256"]:
        raise RuntimeError("EPA archive checksum mismatch")
    workbook = RAW / manifest["workbook"]["filename"]
    if not workbook.exists():
        with zipfile.ZipFile(archive) as bundle:
            matches = [name for name in bundle.namelist() if name.endswith(workbook.name)]
            if len(matches) != 1:
                raise RuntimeError(f"expected one {workbook.name}, found {matches}")
            with bundle.open(matches[0]) as source, workbook.open("wb") as target:
                shutil.copyfileobj(source, target)
    if _sha256(workbook) != manifest["workbook"]["sha256"]:
        raise RuntimeError("EPA workbook checksum mismatch")
    print(f"verified {archive.name} and {workbook.name}")


if __name__ == "__main__":
    main()
