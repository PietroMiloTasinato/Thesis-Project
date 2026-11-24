"""
Extract the running text of a scientific paper from a PDF and save it as JSON.

 * Paragraphs are rebuilt from the text layer (two-column layouts are handled,
   paragraphs split across columns/pages are merged, end-of-line hyphenation
   is repaired).
 * Images are ignored.
 * Formulas are ignored when they can't be read as text: display equations,
   equation numbers, and garbage glyphs (private-use / "(cid:12)" / U+FFFD)
   are dropped, while readable inline math such as "x2" or "α = 0.5" stays.
 * Running headers/footers, page numbers, tiny figure labels and number-only
   fragments (axis ticks, table cells) are dropped too.

Usage:
    from pdf_to_json import pdf_to_json
    data = pdf_to_json("paper.pdf", "paper.json")

    python pdf_to_json.py paper.pdf paper.json --stop-at-references
"""

import json
import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
import pymupdf 


_MATH_FONT_HINTS = ("cmmi", "cmsy", "cmex", "msam", "msbm", "mtmi", "mtsy", "mtextra", "symbol", "math", "eufm", "rsfs")
_MATH_SYMBOLS = set("∑∏∫∮∂∇√∞≈≠≡≤≥≪≫±∓×÷·∘∈∉⊂⊃⊆⊇∪∩∧∨¬∀∃∅→←↔⇒⇐⇔↦∝∼≃≅⊕⊗′″")
_OPERATORS = set("=+−-*/^<>|~_")
_WORD_RE = re.compile(r"[^\W\d_]{3,}")                       # real words (3+ letters)
_JUNK_RE = re.compile(r"\(cid:\d+\)|[\ufffd\u00ad\u200b\ue000-\uf8ff\x00-\x08\x0b-\x1f]®")
_CAPTION_RE = re.compile(r"^\s*(fig(ure)?s?|table|tab)\.?\s*[0-9IVX]+", re.I)
_REFERENCES_RE = re.compile(r"^\W*(\d+\.?\s*)?(references|bibliography|literature cited)\W*$", re.I)
_SENTENCE_END_RE = re.compile(r"[.!?:;][\"')\]”’]*$")
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
_NUMBER_RE = re.compile(r"^[-+−]?\d+(?:\.\d+)?$")

@dataclass
class _Block:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[str]
    size: float            # character-weighted mean font size
    bold_ratio: float      # share of characters in bold
    math_ratio: float      # share of characters set in a math font

    @property
    def text(self) -> str:
        return reduce(_join, self.lines, "")

    @property
    def n_chars(self) -> int:
        return sum(len(line) for line in self.lines)


@dataclass
class _Page:
    number: int
    width: float
    height: float
    has_images: bool
    blocks: list[_Block] = field(default_factory=list)


def _join(a: str, b: str) -> str:    # join two text fragments
    a, b = a.rstrip(), b.lstrip()
    if not a:
        return b
    if not b:
        return a
    if re.search(r"[A-Za-z]-$", a) and b[0].islower():
        return a[:-1] + b          
    return f"{a} {b}"


def _clean(text: str) -> str:   # remove unusable symbles
    return _JUNK_RE.sub("", text)


def _is_math_char(c: str) -> bool: # detects if there is a mathematical symbol being used
    return (c in _MATH_SYMBOLS
            or "\u0370" <= c <= "\u03ff"            # greek letters
            or "\U0001d400" <= c <= "\U0001d7ff")   # alphanumerics


def _is_formula(text: str, math_ratio: float) -> bool: # true for display equations, equation numbers, numeric fragments, etc."""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    n_words = len(_WORD_RE.findall(text))
    if n_words >= 6:                                    # clearly prose
        return False
    n = len(compact)
    alpha = sum(c.isalpha() for c in compact)
    symbols = sum(_is_math_char(c) or c in _OPERATORS for c in compact)
    if math_ratio >= 0.5 and n_words <= 3:              # set in a math font
        return True
    if n_words <= 2 and symbols / n >= 0.2:             # symbol-dense, almost no words
        return True
    if n_words == 0 and alpha / n < 0.6:                # numbers, "(1)", "[12]", ...
        return True
    return False


# --------------------------------------------------------------------------- #
# Page reading
# --------------------------------------------------------------------------- #
def _read_page(page: pymupdf.Page, number: int) -> _Page:
    result = _Page(number, page.rect.width, page.rect.height, bool(page.get_images()))

    for raw in page.get_text("dict")["blocks"]:
        if raw["type"] != 0:                            # type 1 = image -> ignored
            continue

        lines = []                                      # (text, chars, math_chars)
        size_sum = bold_chars = math_chars = total = 0
        for line in raw["lines"]:
            if abs(line["dir"][0]) < 0.9:               # rotated text (arXiv stamp, axis labels)
                continue
            parts, l_chars, l_math = [], 0, 0
            for span in line["spans"]:
                text = _clean(span["text"])
                parts.append(text)
                n = len(text.strip())
                font = span["font"].lower()
                l_chars += n
                size_sum += n * span["size"]
                if span["flags"] & 16 or "bold" in font:
                    bold_chars += n
                if any(h in font for h in _MATH_FONT_HINTS):
                    l_math += n
            text = "".join(parts).strip()
            if not text:
                continue
            # A display equation glued into a paragraph block: drop only that line.
            if len(raw["lines"]) > 1 and l_chars and l_math / l_chars >= 0.6 \
                    and not _WORD_RE.search(text):
                continue
            lines.append((text, l_chars, l_math))
            total += l_chars
            math_chars += l_math

        if not lines or total == 0:
            continue
        x0, y0, x1, y1 = raw["bbox"]
        result.blocks.append(_Block(
            page=number, x0=x0, y0=y0, x1=x1, y1=y1,
            lines=[t for t, _, _ in lines],
            size=size_sum / total,
            bold_ratio=bold_chars / total,
            math_ratio=math_chars / total,
        ))
    return result


def _body_font_size(pages: list[_Page]) -> float:
    counter: Counter = Counter()
    for p in pages:
        for b in p.blocks:
            counter[round(b.size * 2) / 2] += b.n_chars
    return counter.most_common(1)[0][0] if counter else 10.0


def _repeated_margin_texts(pages: list[_Page]) -> set[str]:
    """Texts that show up in the top/bottom margin of many pages (headers/footers)."""
    seen: dict[str, set[int]] = {}
    for p in pages:
        for b in p.blocks:
            if b.y1 < 0.08 * p.height or b.y0 > 0.92 * p.height:
                seen.setdefault(_signature(b.text), set()).add(p.number)
    threshold = max(2, math.ceil(0.4 * len(pages)))
    return {sig for sig, nums in seen.items() if len(nums) >= threshold}


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _reading_order(blocks: list[_Block], page_width: float) -> list[_Block]:
    """Top-to-bottom order; inside a multi-column band: left column, then right."""
    mid, margin = page_width / 2, page_width * 0.05
    ordered: list[_Block] = []
    left: list[_Block] = []
    right: list[_Block] = []

    def flush() -> None:
        ordered.extend(sorted(left, key=lambda b: b.y0))
        ordered.extend(sorted(right, key=lambda b: b.y0))
        left.clear()
        right.clear()

    for b in sorted(blocks, key=lambda b: (b.y0, b.x0)):
        if b.x0 < mid - margin and b.x1 > mid + margin:     # spans both columns
            flush()
            ordered.append(b)
        elif (b.x0 + b.x1) / 2 < mid:
            left.append(b)
        else:
            right.append(b)
    flush()
    return ordered


def _is_heading(b: _Block, body_size: float) -> bool:
    n_words = len(b.text.split())
    if n_words == 0 or n_words > 20 or len(b.lines) > 3:
        return False
    larger = b.size >= body_size * 1.12
    bold_title = b.bold_ratio >= 0.9 and n_words <= 15 and not b.text.endswith(".")
    return larger or bold_title


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def pdf_to_json(
    pdf_path: str | Path,
    json_path: str | Path | None = None,
    *,
    skip_captions: bool = False,
    stop_at_references: bool = False,
) -> dict:
    """
    Read a scientific paper from `pdf_path` and write its paragraphs to `json_path`
    (defaults to the PDF's name with a .json extension). Returns the data written.

    Output format:
        {
          "source": "paper.pdf",
          "title": "<PDF metadata title or null>",
          "num_pages": 9,
          "paragraphs": [
            {"id": 1, "page": 1, "section": "1 Introduction", "text": "..."},
            ...
          ],
          "skipped_pages": [4]      # pages with no text layer (e.g. scanned) -> nothing to read
        }

    Parameters
    ----------
    skip_captions       drop "Figure N ..." / "Table N ..." captions.
    stop_at_references  stop reading when the "References" section starts.
    """
    pdf_path = Path(pdf_path)
    json_path = Path(json_path) if json_path else pdf_path.with_suffix(".json")

    with pymupdf.open(pdf_path) as doc:
        metadata = doc.metadata or {}
        pages = [_read_page(page, i + 1) for i, page in enumerate(doc)]

    body_size = _body_font_size(pages)
    repeated = _repeated_margin_texts(pages)
    skipped_pages = [p.number for p in pages if not p.blocks and p.has_images]

    paragraphs: list[dict] = []
    current: dict | None = None
    section: str | None = None

    def flush() -> None:
        nonlocal current
        if current and current["text"].strip():
            paragraphs.append(current)
        current = None

    done = False
    for page in pages:
        keep = []
        for b in page.blocks:
            text = b.text
            n_words = len(text.split())
            if _signature(text) in repeated:                          # header / footer
                continue
            if _is_formula(text, b.math_ratio):              # formulas, numbers
                continue
            if b.size < body_size * 0.85 and n_words < 6:             # figure labels
                continue
            keep.append(b)

        for b in _reading_order(keep, page.width):
            text = b.text
            if _is_heading(b, body_size):
                flush()
                section = text
                if stop_at_references and _REFERENCES_RE.match(text):
                    done = True
                    break
                continue
            if skip_captions and _CAPTION_RE.match(text):
                continue
            # Continuation of the previous paragraph (next column/page, or text that
            # resumes after a removed display equation).
            if current and not _SENTENCE_END_RE.search(current["text"]) and text[0].islower():
                current["text"] = _join(current["text"], text)
            else:
                flush()
                current = {"page": b.page, "section": section, "text": text}
        if done:
            break
    flush()

    result = {
        "source": pdf_path.name,
        "title": metadata.get("title") or None,
        "num_pages": len(pages),
        "paragraphs": [{"id": i, **p} for i, p in enumerate(paragraphs, start=1)],
        "skipped_pages": skipped_pages,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _is_axis_noise(text: str) -> bool:
    """Short fragments made mostly of numbers, e.g. plot ticks: '0 25 50 75 100 Etot (GeV)'."""
    tokens = text.split()
    if not tokens or len(tokens) > 15:
        return not tokens
    return sum(bool(_NUMBER_RE.match(t)) for t in tokens) / len(tokens) >= 0.4
 
 
def _merge(parts: list[str], separator: str, repair_hyphens: bool) -> str:
    out = ""
    for part in parts:
        if not out:
            out = part
        elif repair_hyphens and len(out) > 1 and out[-1] == "-" and out[-2].isalpha() \
                and part[0].islower():
            out = out[:-1] + part            # "classi-" + "fier"  ->  "classifier"
        else:
            out += separator + part
    return out
 
 
def json_to_sections(
    json_path: str | Path,
    *,
    separator: str = " ",
    clean: bool = True,
    captions: str = "end",
    untitled: str = "(no section)",
) -> dict[str, str]:
    """
    Return {section name: joined text}, in the order the sections first appear.
 
    separator  string placed between the pieces of a section (use "\\n\\n" to keep
               paragraph breaks).
    untitled   key used for text that appears before any heading.
    clean      True -> also: expand ligatures (ﬁ -> fi), drop number-only fragments
               such as plot axis ticks, and repair words hyphenated across pieces.
    captions   only with clean=True. "end": move "FIG. n." / "Table n." captions to the
               end of their section (so they don't cut a paragraph in two),
               "drop": remove them, "keep": leave them where they are.
    """
    if captions not in {"end", "drop", "keep"}:
        raise ValueError("captions must be 'end', 'drop' or 'keep'")
 
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    bodies: dict[str, list[str]] = {}
    caps: dict[str, list[str]] = {}
 
    for par in data["paragraphs"]:
        name = (par.get("section") or "").strip() or untitled
        text = (par.get("text") or "").strip()
        if clean:
            name = name.translate(_LIGATURES)
            text = text.translate(_LIGATURES)
            if _is_axis_noise(text):
                continue
        bodies.setdefault(name, [])
        if not text:
            continue
        if clean and captions != "keep" and _CAPTION_RE.match(text):
            if captions == "end":
                caps.setdefault(name, []).append(text)
            continue
        bodies[name].append(text)
 
    return {
        name: _merge(parts, separator, clean) if not caps.get(name)
        else separator.join(filter(None, [_merge(parts, separator, clean), *caps[name]]))
        for name, parts in bodies.items()
        if parts or caps.get(name)
    }