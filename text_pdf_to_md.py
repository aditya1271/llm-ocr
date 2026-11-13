#!/usr/bin/env python3
"""Text-based PDF → Markdown converter.

Unlike convert.py (scanned PDFs: render → OCR → vision LLM), this tool
extracts embedded text directly from text-PDFs using PyMuPDF.
No OCR. No vision model. No API calls.

Heuristic cleanup handles:
  • Hyphenated line-breaks ("war- riors" → "warriors")
  • Leader dots in TOC ("Introduction . . . . 5" → "Introduction 5")
  • Standalone page numbers
  • Excess blank lines

Usage:
    python text_pdf_to_md.py input.pdf output.md
    python text_pdf_to_md.py input.pdf output.md --start 5 --end 100
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError as exc:  # pragma: no cover
    sys.exit("pymupdf required. Install: python -m pip install pymupdf")


# ── Text cleanup helpers ──────────────────────────────────────────

# "war- riors" or "charac- ter" split across lines
RE_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")

# Lines that are ONLY a number (page number)
RE_PAGE_NUM = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# TOC "Introduction . . . . . . . . . . 5" → leader dots (spaced or solid)
RE_LEADER_DOTS = re.compile(r"(?:\s*\.\s*){3,}")

# 3+ consecutive newlines → 2
RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def join_hyphenated(text: str) -> str:
    """Fix words split across lines: 'war- riors' → 'warriors'."""
    return RE_HYPHEN.sub(r"\1\2", text)


def remove_solo_page_numbers(text: str) -> str:
    """Remove lines that are just a page number."""
    return RE_PAGE_NUM.sub("", text)


def strip_leader_dots(text: str) -> str:
    """Replace leader dots (spaced or solid) with single spaces."""
    return RE_LEADER_DOTS.sub(" ", text)


def collapse_excess_newlines(text: str) -> str:
    return RE_MULTIPLE_NEWLINES.sub("\n\n", text)


def basic_clean(text: str) -> str:
    """Heuristic cleanup. Fast, free, offline."""
    text = join_hyphenated(text)
    text = strip_leader_dots(text)
    text = remove_solo_page_numbers(text)
    text = collapse_excess_newlines(text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Also remove stray spaces before newlines
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


# ── Converter ─────────────────────────────────────────────────────

def convert_pdf(
    pdf_path: Path,
    out_path: Path,
    *,
    start_page: int = 1,
    end_page: int | None = None,
) -> int:
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    end = end_page or total

    out_lines = [f"<!-- Converted from text PDF | {total} pages -->\n"]

    for page_no in range(start_page, end + 1):
        if page_no > total:
            break
        raw = doc[page_no - 1].get_text()
        if not raw.strip():
            continue

        cleaned = basic_clean(raw)
        print(f"[{page_no}/{end}] {len(cleaned)} chars")

        out_lines.append(f"\n<!-- Page {page_no} -->\n")
        out_lines.append(cleaned)
        out_lines.append("\n")

    doc.close()

    out_path.write_text("".join(out_lines), encoding="utf-8")
    print(f"\nDone: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert text-based PDF to Markdown (no OCR, no vision LLM, no API calls)"
    )
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("output", help="Output Markdown file")
    parser.add_argument("--start", type=int, default=1, help="First page (default: 1)")
    parser.add_argument("--end", type=int, default=0, help="Last page (0 = all)")
    args = parser.parse_args()

    return convert_pdf(
        Path(args.pdf),
        Path(args.output),
        start_page=args.start,
        end_page=args.end or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
