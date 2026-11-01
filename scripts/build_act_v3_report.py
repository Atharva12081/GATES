from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "act" / "TECHNICAL_REPORT_V3.md"
CSS = ROOT / "docs" / "final" / "report.css"
OUTPUT = ROOT / "output" / "pdf" / "GATES_ACT_Technical_Report_v3.pdf"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def add_page_numbers(source: Path, destination: Path) -> None:
    reader = PdfReader(source)
    writer = PdfWriter()
    total = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        buffer = io.BytesIO()
        overlay = canvas.Canvas(buffer, pagesize=(width, height))
        overlay.setFont("Helvetica", 7.5)
        overlay.setFillColorRGB(0.30, 0.36, 0.40)
        overlay.drawString(48, 24, "GATES-ACT v3 | independently audited")
        overlay.drawRightString(width - 48, 24, f"{index} / {total}")
        overlay.save()
        buffer.seek(0)
        page.merge_page(PdfReader(buffer).pages[0])
        writer.add_page(page)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    if not CHROME.exists() or shutil.which("pandoc") is None:
        raise FileNotFoundError("Google Chrome and pandoc are required")
    with tempfile.TemporaryDirectory(prefix="gates-act-v3-report-") as temporary:
        temporary_path = Path(temporary)
        html, raw_pdf = temporary_path / "report.html", temporary_path / "report.pdf"
        subprocess.run(
            ["pandoc", str(SOURCE), "--standalone", "--embed-resources",
             "--math-method=mathml", "--toc", "--toc-depth=2", "--css", str(CSS),
             "--resource-path", f"{SOURCE.parent}:{ROOT}", "--output", str(html)],
            cwd=ROOT, check=True,
        )
        subprocess.run(
            [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={raw_pdf}", html.as_uri()],
            check=True,
        )
        add_page_numbers(raw_pdf, OUTPUT)


if __name__ == "__main__":
    main()
