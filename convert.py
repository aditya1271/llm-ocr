#!/home/adiagg6u/github/.venv/bin/python3
"""Two-pass scanned PDF → Markdown converter with batched critique.

Commands:
  pass1    — render every page → OCR → vision LLM → per-page Markdown
  critique — batched quality scan of all pages → flags.json
  pass2    — re-transcribe flagged pages with critique feedback
  merge    — concatenate all pages into final .md
  all      — run pass1 + critique + pass2 + merge in sequence

Usage:
    python convert.py pass1 input.pdf output_dir/ --dpi 150
    python convert.py critique output_dir/ --max-tokens 2000
    python convert.py pass2 input.pdf output_dir/ --auto
    python convert.py merge output_dir/ final.md
    python convert.py all input.pdf final.md --dpi 150

Dependencies (gemCrawler venv):
    pip install pymupdf rapidocr Pillow
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

try:
    import fitz  # pymupdf
except ImportError as exc:  # pragma: no cover
    sys.exit("pymupdf required. Install: python -m pip install pymupdf")

try:
    from rapidocr import RapidOCR
except ImportError as exc:  # pragma: no cover
    sys.exit("rapidocr required. Install: python -m pip install rapidocr")


# ── Config ────────────────────────────────────────────────────────
ENV_PATH = Path("/bata/adiagg6u/github/free-llm-list/.env")
API_BASE = "https://token.sensenova.ai/v1"
VISION_MODEL = "sensenova-6.8-flash"
CRITIQUE_MODEL = "sensenova-6.8-flash-lite"
DEFAULT_DPI = 150
REQUEST_TIMEOUT = 600  # 10 minutes per API call - handles slow connections
RETRIES = 3
RETRY_BACKOFF = 5

# Prompts
TRANSCRIBE_PROMPT = """You are a precise document transcription assistant.
Here is OCR text extracted from a scanned D&D rulebook page. It is a DRAFT and contains errors.
Use it as a hint only. Trust the attached PAGE IMAGE as the final authority.

Draft OCR text (may have errors):
---
{ocr_text}
---

Transcribe the attached page image into clean, simple Markdown.
Rules:
- Use # for main headings, ## for subheadings.
- Preserve paragraph structure.
- Bold important terms using **bold**.
- Convert tables to Markdown tables.
- For boxed text / stat blocks, use fenced code blocks (```).
- Ignore decorative headers, footers, and page numbers.
- Do NOT add any text not visible in the image.
- Output ONLY the Markdown content, nothing else.
"""

RETRANSCRIBE_PROMPT = """You are a precise document transcription assistant.
A previous transcription of this page had these issues:
{flags_str}

Original OCR draft (for reference, may have errors):
---
{ocr_text}
---

Trust the attached PAGE IMAGE as the ONLY authority.
Transcribe it into clean, simple Markdown, fixing the above issues.
Rules: # headings, ## subheadings, **bold** terms, preserve paragraphs,
tables as Markdown tables, boxed text as fenced code blocks.
Ignore headers/footers/page numbers. No commentary, only Markdown.
"""

CRITIQUE_PROMPT = """You are a quality checker for OCR-to-Markdown transcription.

STEP 1: Check if the transcription is an error marker:
- If it starts with "<!-- ERROR:" → this is a PROCESSING FAILURE, not content.
  Extract the actual error message (e.g., "HTTP Error 400", "'content' key missing", etc.)
  Return severity="critical", flag=["api_error"], and describe the error in the note.

STEP 2: Only if NOT an error marker, check for these structural issues:
1. TRUNCATION — text ends mid-sentence without terminal punctuation (e.g., "...characters whose powers far ex")
2. HYPHEN_ARTIFACTS — words incorrectly split with hyphens from OCR line breaks (e.g., "Im- mortals", "Drag- ons")
3. TOO_SHORT — Markdown is suspiciously short (<30%% of OCR draft length) suggesting massive content omission
4. GARBLED — contains literal (cid:...) or � symbols that were not cleaned up
5. NO_STRUCTURE — page should have headings (#/##) but output is flat paragraphs only
6. DUPLICATES — same paragraph block appears verbatim more than once
7. HEADING_MISMATCH — first heading in output differs significantly from first heading in OCR draft
8. UNUSUAL_BREVITY — output is under 100 bytes for a page that should have substantial text (not title pages/maps)

Original OCR draft (for reference):
---
{ocr_text}
---

Final Markdown transcription:
---
{markdown}
---

Return strict JSON only:
{{
  "issues_found": true|false,
  "severity": "critical"|"minor"|"none",
  "flags": ["truncation", "api_error", ...],
  "note": "Specific description: what is wrong and why. For errors, quote the exact error. For quality issues, state the symptom (e.g., 'Text ends abruptly after any with no period')"
}}

Be precise in the note field — it will be used to decide whether to re-transcribe this page.
"""


def load_api_key() -> str:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"Env file not found: {ENV_PATH}")
    with ENV_PATH.open() as fh:
        for line in fh:
            if line.startswith("SENSENOVA_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("SENSENOVA_API_KEY not found in .env")


def call_api(payload: dict, api_key: str) -> dict:
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last_exc: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  HTTP {e.code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise last_exc or RuntimeError("API call failed")


def extract_json(text: str) -> dict:
    """Extract JSON object from model output that may have markdown fences."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)


# ── File layout helpers ───────────────────────────────────────────
def page_dir(out_dir: Path, sub: str) -> Path:
    d = out_dir / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_done_pages(out_dir: Path) -> set[int]:
    """Scan pages/ directory for already-processed page numbers."""
    d = out_dir / "pages"
    done: set[int] = set()
    if not d.exists():
        return done
    for f in d.glob("page_*.md"):
        try:
            num = int(f.stem.split("_")[1])
            done.add(num)
        except (IndexError, ValueError):
            pass
    return done


# ── Per-page pipeline ─────────────────────────────────────────────
class PagePipeline:
    def __init__(self, api_key: str, dpi: int = DEFAULT_DPI):
        self.api_key = api_key
        self.dpi = dpi
        self.ocr_engine = RapidOCR()

    def render(self, doc: fitz.Document, page_no: int) -> Path:
        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=self.dpi)
        out = Path(f"/tmp/pdf_to_md_p{page_no:04d}.png")
        pix.save(str(out))
        return out

    def ocr(self, png: Path) -> str:
        result = self.ocr_engine(str(png))
        if hasattr(result, "rec_res") and result.rec_res:
            return "\n".join(line[1] for line in result.rec_res)
        if hasattr(result, "txts") and result.txts:
            return "\n".join(result.txts)
        return str(result)

    def encode(self, png: Path) -> str:
        with png.open("rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")

    def transcribe(
        self,
        doc: fitz.Document,
        page_no: int,
        model: str = VISION_MODEL,
        extra_prompt: str | None = None,
    ) -> tuple[str, dict]:
        """Render → OCR → vision API → Markdown."""
        png = self.render(doc, page_no)
        try:
            ocr = self.ocr(png)
            img_b64 = self.encode(png)
            prompt = extra_prompt if extra_prompt else TRANSCRIBE_PROMPT.format(ocr_text=ocr[:4000])

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    },
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
            }
            data = call_api(payload, self.api_key)
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return content, usage
        finally:
            png.unlink(missing_ok=True)

    def transcribe_text_only(
        self,
        ocr: str,
        model: str = CRITIQUE_MODEL,
    ) -> tuple[str, dict]:
        """Text-only call (for re-transcribe without image)."""
        prompt = RETRANSCRIBE_PROMPT.format(ocr_text=ocr[:4000], flags_str="[see prompt]")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise document transcription assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        data = call_api(payload, self.api_key)
        return data["choices"][0]["message"]["content"], data.get("usage", {})


# ── Commands ──────────────────────────────────────────────────────

def cmd_pass1(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf)
    out = Path(args.out_dir)
    if not pdf.exists():
        print(f"Error: PDF not found: {pdf}", file=sys.stderr)
        return 1

    api_key = load_api_key()
    pipe = PagePipeline(api_key, dpi=args.dpi)
    doc = fitz.open(str(pdf))
    total = len(doc)
    end = args.end if args.end else total

    done = list_done_pages(out)
    pages_d = page_dir(out, "pages")
    ocr_d = page_dir(out, "ocr")

    for p in range(args.start, end + 1):
        if p in done and not args.force:
            print(f"[{p}/{end}] Already done, skipping (--force to override)")
            continue

        print(f"[{p}/{end}] Transcribing...", end=" ", flush=True)
        t0 = time.time()
        try:
            md, usage = pipe.transcribe(doc, p)
            dur = time.time() - t0
            print(f"OK ({dur:.1f}s | {usage.get('prompt_tokens',0)}in/{usage.get('completion_tokens',0)}out)")

            (pages_d / f"page_{p:04d}.md").write_text(md, encoding="utf-8")
            # Save OCR for critique pass
            png = pipe.render(doc, p)
            ocr_text = pipe.ocr(png)
            (ocr_d / f"page_{p:04d}.txt").write_text(ocr_text, encoding="utf-8")
            png.unlink(missing_ok=True)
        except Exception as exc:
            print(f"FAILED: {exc}")

    doc.close()
    print(f"\nPass 1 complete. Pages saved to: {pages_d}")
    return 0


def cmd_critique(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    pages_d = out / "pages"
    ocr_d = out / "ocr"

    if not pages_d.exists():
        print(f"Error: No pages found in {pages_d}", file=sys.stderr)
        return 1

    api_key = load_api_key()
    results: list[dict] = []
    files = sorted(pages_d.glob("page_*.md"))

    print(f"Critiquing {len(files)} pages with {CRITIQUE_MODEL}...\n")
    for i, md_file in enumerate(files, 1):
        page_no = int(md_file.stem.split("_")[1])
        ocr_file = ocr_d / f"page_{page_no:04d}.txt"

        ocr_text = ocr_file.read_text(encoding="utf-8") if ocr_file.exists() else ""
        md_text = md_file.read_text(encoding="utf-8")

        prompt = CRITIQUE_PROMPT.format(ocr_text=ocr_text[:2000], markdown=md_text[:4000])
        payload = {
            "model": CRITIQUE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a strict quality checker."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }

        print(f"[{i}/{len(files)}] Page {page_no}...", end=" ", flush=True)
        try:
            data = call_api(payload, api_key)
            raw = data["choices"][0]["message"]["content"]
            verdict = extract_json(raw)
            verdict["page"] = page_no
            results.append(verdict)

            if verdict.get("issues_found"):
                sev = verdict.get("severity", "unknown")
                flags = ", ".join(verdict.get("flags", []))
                print(f"{sev.upper()}: {flags}")
            else:
                print("OK")
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({"page": page_no, "issues_found": False, "severity": "none", "flags": [], "note": f"critique failed: {exc}"})

    # Save results
    critique_path = out / "critique.json"
    with critique_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    flagged = [r for r in results if r.get("issues_found")]
    critical = [r for r in results if r.get("severity") == "critical"]
    print(f"\n{'='*50}")
    print(f"Critique complete. {len(flagged)} flagged, {len(critical)} critical.")
    print(f"Saved to: {critique_path}")
    if flagged:
        print("\nFlagged pages:")
        for r in flagged:
            print(f"  Page {r['page']:3d}  {r['severity']:8s}  {', '.join(r.get('flags',[]))}")
    print("="*50)
    return 0


def cmd_pass2(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf)
    out = Path(args.out_dir)
    if not pdf.exists():
        print(f"Error: PDF not found: {pdf}", file=sys.stderr)
        return 1

    critique_path = out / "critique.json"
    if not critique_path.exists():
        print(f"Error: No critique.json found. Run 'critique' first.", file=sys.stderr)
        return 1

    with critique_path.open() as fh:
        results: list[dict] = json.load(fh)

    # Filter to retry
    to_retry: list[dict] = []
    if args.severity:
        allowed = set(args.severity.split(","))
        to_retry = [r for r in results if r.get("severity") in allowed and r.get("issues_found")]
    else:
        to_retry = [r for r in results if r.get("issues_found")]

    if not to_retry:
        print("No pages match retry criteria. All clean!")
        return 0

    print(f"Re-transcribing {len(to_retry)} pages...")
    api_key = load_api_key()
    pipe = PagePipeline(api_key, dpi=args.dpi)
    doc = fitz.open(str(pdf))
    pages_d = page_dir(out, "pages")
    ocr_d = out / "ocr"

    for r in to_retry:
        page_no = r["page"]
        flags = r.get("flags", [])
        note = r.get("note", "")
        flags_str = "; ".join(flags)
        if note:
            flags_str += f" | {note}"

        print(f"[{page_no}] Re-transcribing (issues: {flags_str})...", end=" ", flush=True)
        t0 = time.time()
        try:
            # Build enhanced prompt with critique feedback
            ocr_file = ocr_d / f"page_{page_no:04d}.txt"
            ocr = ocr_file.read_text(encoding="utf-8") if ocr_file.exists() else ""
            extra = RETRANSCRIBE_PROMPT.format(ocr_text=ocr[:4000], flags_str=flags_str)

            md, usage = pipe.transcribe(doc, page_no, extra_prompt=extra)
            dur = time.time() - t0
            print(f"OK ({dur:.1f}s | {usage.get('prompt_tokens',0)}in/{usage.get('completion_tokens',0)}out)")

            # Overwrite
            (pages_d / f"page_{page_no:04d}.md").write_text(md, encoding="utf-8")
        except Exception as exc:
            print(f"FAILED: {exc}")

    doc.close()
    print(f"\nPass 2 complete.")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    final = Path(args.output)
    pages_d = out / "pages"

    if not pages_d.exists():
        print(f"Error: No pages dir found: {pages_d}", file=sys.stderr)
        return 1

    files = sorted(pages_d.glob("page_*.md"))
    if not files:
        print("Error: No page files found.", file=sys.stderr)
        return 1

    with final.open("w", encoding="utf-8") as fh:
        fh.write(f"<!-- Converted from PDF | {len(files)} pages -->\n\n")
        for f in files:
            page_no = int(f.stem.split("_")[1])
            text = f.read_text(encoding="utf-8")
            fh.write(f"\n\n<!-- Page {page_no} -->\n\n")
            fh.write(text)

    print(f"Merged {len(files)} pages into: {final}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run full pipeline: pass1 → critique → pass2 → merge."""
    pdf = Path(args.pdf)
    out = Path(args.out_dir)
    final = Path(args.output)

    # Pass 1
    print("\n" + "="*50)
    print("PASS 1: Transcribe all pages")
    print("="*50)
    r = cmd_pass1(argparse.Namespace(pdf=str(pdf), out_dir=str(out), dpi=args.dpi, start=args.start, end=args.end, force=False))
    if r != 0:
        return r

    # Critique
    print("\n" + "="*50)
    print("CRITIQUE: Batch quality scan")
    print("="*50)
    r = cmd_critique(argparse.Namespace(out_dir=str(out)))
    if r != 0:
        return r

    # Pass 2 (auto-retry critical + severe)
    print("\n" + "="*50)
    print("PASS 2: Re-transcribe flagged pages")
    print("="*50)
    r = cmd_pass2(argparse.Namespace(pdf=str(pdf), out_dir=str(out), dpi=args.dpi, severity="critical,minor"))
    if r != 0:
        return r

    # Merge
    print("\n" + "="*50)
    print("MERGE: Concatenate to final Markdown")
    print("="*50)
    r = cmd_merge(argparse.Namespace(out_dir=str(out), output=str(final)))
    return r


# ── CLI ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-pass PDF→Markdown with batched critique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Commands:
  pass1     [...]   Render every page → OCR → vision LLM
  critique  [...]   Batched quality scan → critique.json
  pass2     [...]   Re-transcribe flagged pages only
  merge     [...]   Concatenate pages/ → single .md file
  all       [...]   Run pass1+critique+pass2+merge in sequence
""",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # pass1
    p1 = sub.add_parser("pass1", help="Transcribe all pages")
    p1.add_argument("pdf", help="Input PDF path")
    p1.add_argument("out_dir", help="Output directory (creates pages/, ocr/)")
    p1.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p1.add_argument("--start", type=int, default=1)
    p1.add_argument("--end", type=int, default=0)
    p1.add_argument("--force", action="store_true", help="Overwrite existing pages")
    p1.set_defaults(func=cmd_pass1)

    # critique
    pc = sub.add_parser("critique", help="Quality scan all transcribed pages")
    pc.add_argument("out_dir", help="Directory containing pages/ and ocr/")
    pc.set_defaults(func=cmd_critique)

    # pass2
    p2 = sub.add_parser("pass2", help="Re-transcribe flagged pages")
    p2.add_argument("pdf", help="Original PDF path")
    p2.add_argument("out_dir", help="Directory containing pages/, ocr/, critique.json")
    p2.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p2.add_argument("--severity", default="critical,minor", help="Comma-list severities to retry (e.g. critical,minor)")
    p2.add_argument("--auto", action="store_true", help="Auto-retry without prompting")
    p2.set_defaults(func=cmd_pass2)

    # merge
    pm = sub.add_parser("merge", help="Concatenate pages to final Markdown")
    pm.add_argument("out_dir", help="Directory containing pages/")
    pm.add_argument("output", help="Final .md file path")
    pm.set_defaults(func=cmd_merge)

    # all
    pa = sub.add_parser("all", help="Run full pipeline")
    pa.add_argument("pdf", help="Input PDF path")
    pa.add_argument("output", help="Final .md file path")
    pa.add_argument("--out-dir", default="/tmp/pdf_to_md_output", dest="out_dir")
    pa.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    pa.add_argument("--start", type=int, default=1)
    pa.add_argument("--end", type=int, default=0)
    pa.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
