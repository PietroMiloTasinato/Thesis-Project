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
_WORD_RE = re.compile(r"[^\W\d_]{3,}")
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


def _join(a: str, b: str) -> str:    # join two text from the same paragraph
    a, b = a.rstrip(), b.lstrip()
    if not a:
        return b
    if not b:
        return a
    if re.search(r"[A-Za-z]-$", a) and b[0].islower():
        return a[:-1] + b          
    return f"{a} {b}"


def _clean(text: str) -> str:   # remove unusable symboles
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
    if n_words >= 6:                                    # non mathematical text
        return False
    n = len(compact)
    alpha = sum(c.isalpha() for c in compact)
    symbols = sum(_is_math_char(c) or c in _OPERATORS for c in compact)
    if math_ratio >= 0.5 and n_words <= 3:              # set in a math font
        return True
    if n_words <= 2 and symbols / n >= 0.2:             # lots of symbols and almost no words
        return True
    if n_words == 0 and alpha / n < 0.6:                # numbers
        return True
    return False


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


def _read_footnotes(pages: list[_Page]) -> set[str]: # method to hanle footnotes

    seen: dict[str, set[int]] = {}
    for p in pages:
        for b in p.blocks:
            if b.y1 < 0.08 * p.height or b.y0 > 0.92 * p.height:
                seen.setdefault(_signature(b.text), set()).add(p.number)
    threshold = max(2, math.ceil(0.4 * len(pages)))
    return {sig for sig, nums in seen.items() if len(nums) >= threshold}


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _reading_order(blocks: list[_Block], page_width: float) -> list[_Block]: # method to hanle the case with 2 coplumns

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


"""
notes: handles two-column layouts, and paragraphs split across columns/pages
images are ignored, formulas are ignored when they can't be read as text (display equation in latex)
handles headers and footers
"""

def pdf_to_json(
    pdf_path: str | Path,
    json_path: str | Path | None = None,
    *,
    skip_captions: bool = False,
) -> dict:
    
    pdf_path = Path(pdf_path)
    json_path = Path(json_path) if json_path else pdf_path.with_suffix(".json")

    with pymupdf.open(pdf_path) as doc:
        metadata = doc.metadata or {}
        pages = [_read_page(page, i + 1) for i, page in enumerate(doc)]

    body_size = _body_font_size(pages)
    footnotes = _read_footnotes(pages)
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
            if _signature(text) in footnotes:
                continue
            if _is_formula(text, b.math_ratio):              # formulas, numbers
                continue
            if b.size < body_size * 0.85 and n_words < 6:    # figure labels
                continue
            keep.append(b)

        for b in _reading_order(keep, page.width):
            text = b.text
            if _is_heading(b, body_size):
                flush()
                section = text
                continue
            if skip_captions and _CAPTION_RE.match(text):
                continue
            # print()
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
    return result # print(result)


"""
unifies different sections in a json file into a dict that follows this structure:
{section_name : text} where the name of the section is the title of the paragraph in the paper (abstract, results, references,...)
text is the complete text of the paragraph that was prevoiusly collected by pdf_to_json
"""

def json_to_sections(
    json_path: str | Path,
    *,
    separator: str = " ",
    clean: bool = True,
    captions: str = "end",
    untitled: str = "(no section)",
) -> dict[str, str]:

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

def _is_axis_noise(text: str) -> bool: # ignore values generated by images and tables

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
 
