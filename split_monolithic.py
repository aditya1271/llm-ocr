#!/home/adiagg6u/github/.venv/bin/python3
"""Split a monolithic Markdown file (from old convert.py) into per-page files.

Also re-renders PDF pages and runs OCR to produce the ocr/ text files
needed for the critique pass.

Usage:
    python split_monolithic.py input.pdf monolithic.md output_dir/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    sys.exit("pymupdf required")

try:
    from rapidocr import RapidOCR
except ImportError:
    sys.exit("rapidocr required")


def split_markdown(md_path: Path, pages_dir: Path) -> dict[int, str]:
    """Split monolithic .md on <!-- Page N --> markers.
    Returns {page_no: content_without_marker}.
    """
    text = md_path.read_text(encoding="utf-8")
    pages_dir.mkdir(parents=True, exist_ok=True)

    # Pattern: optional whitespace, then <!-- Page N -->, optional whitespace
    pattern = re.compile(r"\n?\s*<!--\s*Page\s+(\d+)\s*-->\s*\n?")
    parts = pattern.split(text)

    pages: dict[int, str] = {}
    if not parts:
        return pages

    # parts[0] = prologue before first marker
    # parts[1] = page number (first match)
    # parts[2] = content after first marker up to second marker
    # etc.

    i = 1
    while i + 1 < len(parts):
        try:
            page_no = int(parts[i])
        except ValueError:
            i += 2
            continue
        content = parts[i + 1]
        pages[page_no] = content.strip()
        i += 2

    # Write per-page files
    written = 0
    for page_no, content in sorted(pages.items()):
        f = pages_dir / f"page_{page_no:04d}.md"
        f.write_text(content, encoding="utf-8")
        written += 1

    return pages


def generate_ocr_files(doc: fitz.Document, ocr_dir: Path, dpi: int = 150) -> dict[int, str]:
    """Render each page to PNG, OCR it, save to ocr/page_NNNN.txt."""
    ocr_dir.mkdir(parents=True, exist_ok=True)
    engine = RapidOCR()
    ocr_map: dict[int, str] = {}

    total = len(doc)
    for page_no in range(1, total + 1):
        png = Path(f"/tmp/split_ocr_p{page_no:04d}.png")
        try:
            page = doc[page_no - 1]
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(png))

            result = engine(str(png))
            if hasattr(result, "rec_res") and result.rec_res:
                text = "\n".join(line[1] for line in result.rec_res)
            elif hasattr(result, "txts") and result.txts:
                text = "\n".join(result.txts)
            else:
                text = str(result)

            f = ocr_dir / f"page_{page_no:04d}.txt"
            f.write_text(text, encoding="utf-8")
            ocr_map[page_no] = text

            if page_no % 10 == 0 or page_no == total:
                print(f"  OCR progress: {page_no}/{total} pages")
        finally:
            png.unlink(missing_ok=True)

    return ocr_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Split monolithic .md + generate OCR files")
    parser.add_argument("pdf", help="Original PDF path")
    parser.add_argument("md", help="Monolithic Markdown file from old convert.py")
    parser.add_argument("out_dir", help="Output directory (creates pages/ and ocr/)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    md_path = Path(args.md)
    out_dir = Path(args.out_dir)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    if not md_path.exists():
        print(f"Error: Markdown file not found: {md_path}", file=sys.stderr)
        return 1

    pages_dir = out_dir / "pages"
    ocr_dir = out_dir / "ocr"

    print(f"Reading {md_path.name}...")
    pages = split_markdown(md_path, pages_dir)
    print(f"Split into {len(pages)} page files in {pages_dir}")

    print(f"\nGenerating OCR text files from {pdf_path.name}...")
    doc = fitz.open(str(pdf_path))
    ocr_map = generate_ocr_files(doc, ocr_dir, args.dpi)
    doc.close()
    print(f"OCR files saved to {ocr_dir}")

    print(f"\nDone. Output directory: {out_dir}")
    print(f"  pages/: {len(list(pages_dir.glob('*.md')))} files")
    print(f"  ocr/:   {len(list(ocr_dir.glob('*.txt')))} files")
    print(f"\nNext step: python convert.py critique {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
