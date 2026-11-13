#!/usr/bin/env python3
"""Text-based PDF to Markdown using PyMuPDF get_text("blocks") + column heuristics.

No vision LLM. No API calls. Works entirely from embedded font metadata.

Uses `page.get_text("blocks")` which gives perfect character accuracy, then
clusters blocks by x-position into columns for correct reading order. Also
fixes cross-page hyphenation and drop-cap issues.

Usage:
    python text_pdf_to_md.py input.pdf output.md
    python text_pdf_to_md.py input.pdf output.md --start 5 --end 150
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


# -- Regexes ---------------------------------------------------------

RE_LEADER_DOTS = re.compile(r"(?:\s*\.{2,}\s*){2,}")
RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
RE_DROP_CAP_H = re.compile(r"^([a-z][a-z]?)\s+([A-Z])")


def fix_drop_cap(text: str) -> str:
    """Some pages start body text with a decorative drop-cap image,
    leaving only 'he D&D...' instead of 'The D&D...'. Try to fix."""
    m = RE_DROP_CAP_H.match(text)
    if m:
        word = m.group(1)
        # If page body starts with a 1-2 letter lowercase word followed by
        # an uppercase word, it's probably a missing drop-cap.
        # Common patterns: he->The, t->It, o->To, a->A
        replacement = {
            "he": "The", "t": "It", "o": "To", "a": "A",
            "i": "I", "whe": "When", "bu": "But",
        }
        if word in replacement:
            text = replacement[word] + text[len(word):]
    return text


def fix_hyphenation(text: str) -> str:
    """Remove hyphenation at end of lines."""
    # word-split with newline in between
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # word-split without newline
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return text


def strip_leader_dots(text: str) -> str:
    """Replace leader dots with a space (TOC entries)."""
    return RE_LEADER_DOTS.sub(" ", text)


def collapse_blanks(text: str) -> str:
    return RE_MULTIPLE_NEWLINES.sub("\n\n", text)


def is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def is_title_case(text: str) -> bool:
    words = text.strip().split()
    return len(words) >= 1 and words[0] and words[0][0].isupper()


def is_standalone_number(text: str) -> bool:
    return bool(re.match(r"^\s*\d+\s*$", text))


# -- Column helpers --------------------------------------------------

def assign_columns(blocks: list[dict], page_w: float) -> dict[str, list[dict]]:
    """Cluster blocks into left/center/right columns by x-center."""
    thirds = (page_w / 3.0, 2.0 * page_w / 3.0)
    cols: dict[str, list[dict]] = {"left": [], "center": [], "right": []}

    for b in blocks:
        cx = (b["x0"] + b["x1"]) / 2
        if cx < thirds[0]:
            cols["left"].append(b)
        elif cx > thirds[1]:
            cols["right"].append(b)
        else:
            cols["center"].append(b)

    # Discard nearly-empty columns
    total = len(blocks)
    active = {k: v for k, v in cols.items() if len(v) >= max(2, total * 0.03)}
    if not active:
        return {"center": blocks}
    return active


# -- Block classifier ------------------------------------------------

SPELL_KEYWORDS = {
    "range:", "duration:", "effect:", "save:", "ac:", "hit dice:", "move:",
    "casting time:", "area of effect:", "components:", "level:", "school:",
    "weight:", "cost:", "xp value:", "alignment:", "armor class:",
}


def classify_block(text: str) -> tuple[str, str]:
    """Return (type, text) for a text block."""
    text = text.strip()
    if not text:
        return ("skip", "")

    text = fix_hyphenation(text)
    text = strip_leader_dots(text)
    text = re.sub(r"  +", " ", text)
    text = text.strip()

    if not text:
        return ("skip", "")

    if is_standalone_number(text):
        return ("skip", "")

    # Skip TSR / copyright fragments
    if "TSR" in text and len(text) < 150:
        return ("skip", "")

    # Bullet points
    m = re.match(r"^([•\-*◦■])\s*(.*)", text)
    if m:
        return ("bullet", f"{m.group(1)} {m.group(2)}")

    # Numbered list items
    m = re.match(r"^(\d+[\.\)])\s+(.*)", text)
    if m:
        return ("bullet", f"{m.group(1)} {m.group(2)}")

    is_short = len(text) < 80
    word_count = len(text.split())
    lower_text = text.lower()

    # Spell stat block header: short title-case with spell keywords in body
    if word_count >= 2 and is_title_case(text) and any(kw in lower_text for kw in SPELL_KEYWORDS):
        return ("h3", text)

    if is_short:
        if is_all_caps(text):
            if len(text) < 45:
                return ("h1", text)
            return ("h2", text)
        if is_title_case(text) and word_count <= 8:
            return ("h3", text)

    # Multi-line short heading
    lines = text.splitlines()
    if len(lines) <= 2 and all(len(ln.strip()) < 45 for ln in lines) and word_count <= 8:
        if is_all_caps(text):
            return ("h2", text)
        if is_title_case(text):
            return ("h3", text)

    return ("paragraph", text)


def is_header_footer(y0: float, page_h: float, text: str) -> bool:
    """Heuristic: skip page numbers / chapter titles in margins."""
    if y0 < page_h * 0.03:
        return len(text.strip()) < 40
    if y0 > page_h * 0.97:
        return len(text.strip()) < 40
    return False


# -- Page processing -------------------------------------------------

def extract_page(doc: fitz.Document, page_no: int) -> list[tuple[str, str]]:
    """Extract (type, text) blocks from page, in correct reading order."""
    page = doc[page_no - 1]
    raw_blocks = page.get_text("blocks")

    page_w = page.rect.width
    page_h = page.rect.height

    # Convert to structured blocks
    blocks = []
    for b in raw_blocks:
        x0, y0, x1, y1, text, _, _ = b
        text = text.strip()
        if not text or len(text) < 2:
            continue
        blocks.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": text,
        })

    if not blocks:
        return []

    # Assign columns
    columns = assign_columns(blocks, page_w)

    # Process each column independently, top-to-bottom
    results: list[tuple[str, str]] = []
    for col_name in ("left", "center", "right"):
        if col_name not in columns:
            continue
        col_blocks = sorted(columns[col_name], key=lambda b: b["y0"])

        for b in col_blocks:
            if is_header_footer(b["y0"], page_h, b["text"]):
                continue

            btype, text = classify_block(b["text"])
            text = text.strip()
            if not text:
                continue

            # Fix drop-cap on body paragraphs that start page body
            if btype == "paragraph":
                text = fix_drop_cap(text)

            results.append((btype, text))

    return results


def merge_across_pages(pages: list[list[tuple[str, str]]]) -> list[list[tuple[str, str]]]:
    """Fix cross-page hyphenation: if page N ends with word-\n and page N+1
    starts with that word's continuation, merge them."""
    for i in range(len(pages) - 1):
        if not pages[i] or not pages[i + 1]:
            continue

        # Look at last block of page i
        last_type, last_text = pages[i][-1]
        first_type, first_text = pages[i + 1][0]

        # Check for hyphenation: word ending in hyphen or dash
        m_last = re.search(r"([A-Za-z])-\s*$", last_text)
        if m_last and first_type in ("paragraph", "h3"):
            m_first = re.match(r"([A-Za-z]+)", first_text)
            if m_first:
                # Join: remove trailing dash and prepend remainder
                last_text = re.sub(r"([A-Za-z])-\s*$", r"\1", last_text)
                first_text = m_first.group(1) + first_text[m_first.end():]
                pages[i][-1] = (last_type, last_text)
                pages[i + 1][0] = (first_type, first_text)

    return pages


# -- Formatter -------------------------------------------------------

def format_page(blocks: list[tuple[str, str]]) -> str:
    """Convert list of (type, text) tuples into Markdown."""
    output: list[str] = []
    for btype, text in blocks:
        if btype == "h1":
            output.append(f"# {text}")
        elif btype == "h2":
            output.append(f"## {text}")
        elif btype == "h3":
            output.append(f"### {text}")
        elif btype == "bullet":
            output.append(text)
        elif btype == "paragraph":
            output.append(text)

    text = "\n\n".join(output)
    text = collapse_blanks(text)
    return text.strip()


# -- Converter -------------------------------------------------------

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

    # Collect all pages first for cross-page hyphenation
    all_pages: list[list[tuple[str, str]]] = []

    for page_no in range(start_page, end + 1):
        if page_no > total:
            break

        blocks = extract_page(doc, page_no)
        if not blocks:
            all_pages.append([])
            continue

        all_pages.append(blocks)

    # Merge cross-page hyphenation
    all_pages = merge_across_pages(all_pages)

    for idx, blocks in enumerate(all_pages):
        page_no = start_page + idx
        formatted = format_page(blocks) if blocks else ""
        if formatted:
            print(f"[{page_no}/{end}] {len(formatted)} chars ({len(blocks)} blocks)")
        if idx > 0:
            out_lines.append("\n\n---\n\n")  # visual page break
        out_lines.append(f"<!-- Page {page_no} -->\n")
        if formatted:
            out_lines.append(formatted)
            out_lines.append("\n")
    doc.close()

    out_path.write_text("".join(out_lines), encoding="utf-8")
    print(f"\nDone: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Text PDF to Markdown via pymupdf blocks + column heuristics"
    )
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("output", help="Output Markdown file")
    parser.add_argument("--start", type=int, default=1, help="First page")
    parser.add_argument("--end", type=int, default=0, help="Last page (0=all)")
    args = parser.parse_args()

    return convert_pdf(
        Path(args.pdf),
        Path(args.output),
        start_page=args.start,
        end_page=args.end or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
