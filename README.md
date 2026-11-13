# llm-ocr

PDF → Markdown conversion — two tools for two types of PDFs.

| Tool | PDF type | Cost | Speed | Accuracy |
|------|----------|------|-------|----------|
| **`text_pdf_to_md.py`** | Text-based PDF (selectable text) | **Free** | Fast (seconds) | Good (raw text + heuristic cleanup) |
| **`convert.py`** | Scanned/image PDF (needs OCR) | ~$3-4 per book (vision LLM + OCR) | Slow (minutes) | Excellent (vision model reads actual image) |

## Quick Start

### 1. Text PDF → Markdown (no API needed)

```bash
pip install pymupdf

python text_pdf_to_md.py book.pdf book.md
python text_pdf_to_md.py book.pdf book.md --start 5 --end 100   # page range
```

This extracts embedded text directly via PyMuPDF, then cleans up:
- Fixes hyphenated line-breaks (`"war- riors"` → `"warriors"`)
- Strips TOC leader dots (`"Introduction . . . . 5"` → `"Introduction 5"`)
- Removes standalone page numbers
- Collapses excess blank lines

No OCR. No vision models. No API calls. Works offline.

### 2. Scanned PDF → Markdown (vision LLM + OCR)

```bash
pip install pymupdf rapidocr Pillow

export SENSENOVA_API_KEY="your-key"

python convert.py pass1  book.pdf  ./output  --dpi 150
python convert.py merge  ./output  book.md
```

Full pipeline: `pass1` → `critique` → `pass2` → `merge` (see below).

---

## `convert.py` (Scanned PDFs)

### Commands

| Command | Purpose |
|---------|---------|
| `pass1` | Render → OCR → vision LLM → per-page Markdown |
| `critique` | Batched quality scan → `critique.json` |
| `pass2` | Re-transcribe flagged pages with feedback |
| `merge` | Concatenate `pages/*.md` into one file |
| `all` | Run `pass1 → critique → pass2 → merge` in one shot |

### Example — full pipeline

```bash
python convert.py all book.pdf book_final.md --dpi 150 --start 1 --end 50
```

### Per-page crash safety

Each page is written to `output/pages/page_NNNN.md` as soon as it completes.
If a run is interrupted, re-running `pass1` resumes seamlessly — already-done pages are skipped.

### Models

Default: **SenseNova** (`sensenova-6.8-flash` for vision, `sensenova-6.8-flash-lite` for critique).
Change `VISION_MODEL` and `CRITIQUE_MODEL` in `convert.py` to switch providers (OpenAI, etc.).

---

## `text_pdf_to_md.py` (Text PDFs)

For PDFs where text is already embedded as computer-readable characters (not images).
This is common for modern PDFs generated from Word/InDesign, not scanned books.

### How to tell which tool you need

| Check | Text PDF | Scanned PDF |
|-------|----------|-------------|
| Can you select/copy text in a PDF reader? | Yes | No |
| Is text character-perfect but layout is broken? | Yes | — |
| Are there photos of pages (scan artifacts)? | No | Yes |
| Is the file size unusually large for the page count? | No | Yes |

### Usage

```bash
# All pages
python text_pdf_to_md.py book.pdf book.md

# Specific range
python text_pdf_to_md.py book.pdf book.md --start 10 --end 50

# Background job (305-page example)
python text_pdf_to_md.py DnD_Rules_Cyclopedia.pdf cyclopedia.md
```

### Output format

```markdown
<!-- Converted from text PDF | 305 pages -->

<!-- Page 2 -->
Table of Contents
Introduction
What Is Role-Playing?
Setting Up
...

<!-- Page 5 -->
The D&D Cyclopedia is as accurate a name...
```

Page boundaries are marked with `<!-- Page N -->` comments.

---

## Files

| File | Description |
|------|-------------|
| `convert.py` | Scanned PDF → Markdown via OCR + vision LLM |
| `text_pdf_to_md.py` | Text PDF → Markdown via direct extraction + heuristic cleanup |
| `split_monolithic.py` | Split old monolithic `.md` into per-page files + regenerate OCR |

## License

MIT — use it however you like.
