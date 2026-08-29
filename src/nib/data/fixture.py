"""A synthetic stand-in for IAM, in IAM's own directory and file layout.

Why this exists: IAM is ~1.5 GB behind a registration wall, and every piece of the
data pipeline -- parser, dataset, collate, packing, metrics -- needs *something* to
read while it is being written. This module manufactures that something in a few
seconds, so none of that work waits on a download.

What it is not: this does not look like handwriting, and it is not meant to. Its
job is to exercise code paths, not models. What it does guarantee is the two
properties the pipeline actually depends on:

  * each synthetic writer is visually *consistent* with themselves
  * different writers are visually *distinguishable* from each other

which is enough to give a writer-retrieval metric or a style encoder something
non-trivial to succeed or fail at.

Rendering uses Pillow's built-in font, deliberately. Depending on system fonts
would make the fixture differ between this laptop and a Colab VM, and a fixture
that is not identical everywhere is useless as a test baseline.

.. warning::
   The XML schema below is a reconstruction of IAM's, written without access to a
   real IAM file (the FKI site was unreachable). It is close enough to develop the
   parser against, but the parser MUST be re-validated against a genuine IAM XML
   file once the download completes. This is tracked in PROGRESS.md.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nib.data import charset as cs

# A small fixed vocabulary. Fixed rather than random so that word lengths are
# realistic and every run produces the same corpus for the same seed.
VOCABULARY = [
    "the",
    "and",
    "for",
    "not",
    "but",
    "you",
    "all",
    "any",
    "can",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "man",
    "new",
    "now",
    "old",
    "see",
    "two",
    "way",
    "who",
    "boy",
    "did",
    "about",
    "after",
    "again",
    "below",
    "could",
    "every",
    "first",
    "found",
    "great",
    "house",
    "large",
    "learn",
    "never",
    "other",
    "place",
    "plant",
    "point",
    "right",
    "small",
    "sound",
    "spell",
    "still",
    "study",
    "their",
    "there",
    "these",
    "thing",
    "think",
    "three",
    "water",
    "where",
    "which",
    "world",
    "would",
    "write",
    "years",
    "young",
    "London",
    "England",
    "government",
    "yesterday",
    "important",
    "1960",
    "23",
]

_BASE_FONT_SIZE = 30
_INK_THRESHOLD = 250  # pixels darker than this count as ink when cropping


@dataclass(frozen=True)
class WriterStyle:
    """The handful of parameters that make one synthetic writer look like itself.

    Derived deterministically from the writer index, so writer 007 is identical in
    every run and on every machine.
    """

    writer_id: str
    slant: float  # horizontal shear; stands in for handwriting slope
    stroke: int  # -1 thin, 0 unchanged, +1 thick
    char_spacing: int  # pixels between characters
    baseline_jitter: float  # vertical wobble per character
    scale: float
    ink: int  # 0 = black, higher = fainter


def style_for_writer(index: int, seed: int = 0) -> WriterStyle:
    """Build a stable style for writer ``index``.

    Seeded from a hash of (seed, index) rather than from a shared global RNG, so
    that generating writer 5 alone gives the same result as generating writers
    0..99 and taking the fifth.
    """
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return WriterStyle(
        writer_id=f"{index:03d}",
        slant=rng.uniform(-0.35, 0.35),
        stroke=rng.choice([-1, 0, 0, 1]),
        char_spacing=rng.randint(0, 4),
        baseline_jitter=rng.uniform(0.0, 2.5),
        scale=rng.uniform(0.85, 1.25),
        ink=rng.randint(0, 90),
    )


def render_word(text: str, style: WriterStyle, height: int, seed: int = 0) -> np.ndarray:
    """Render one word as a grayscale image: dark ink on a light background.

    Returns a uint8 array of shape (height, width). Width follows the text, which
    is the property that makes the variable-width collate path worth testing.
    """
    if not text:
        raise ValueError("cannot render an empty word")

    key = f"{seed}:{style.writer_id}:{text}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    font = ImageFont.load_default(size=max(8, int(_BASE_FONT_SIZE * style.scale)))

    widths = [max(1, int(font.getlength(c))) for c in text]
    pad = 24
    canvas_w = sum(widths) + style.char_spacing * len(text) + 2 * pad
    canvas_h = int(_BASE_FONT_SIZE * style.scale * 3) + 2 * pad

    image = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(image)

    x = pad
    baseline = canvas_h // 2
    for char, width in zip(text, widths, strict=True):
        dy = rng.uniform(-style.baseline_jitter, style.baseline_jitter)
        draw.text((x, baseline + dy), char, font=font, fill=style.ink)
        x += width + style.char_spacing

    array = np.array(image)
    array = _apply_slant(array, style.slant)
    array = _apply_stroke(array, style.stroke)
    array = cv2.GaussianBlur(array, (3, 3), 0.6)
    return _crop_and_fit(array, height)


def _apply_slant(array: np.ndarray, slant: float) -> np.ndarray:
    h, w = array.shape
    matrix = np.array([[1.0, slant, -slant * h / 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(array, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _apply_stroke(array: np.ndarray, stroke: int) -> np.ndarray:
    """Thicken or thin the ink.

    Ink is dark on a light background, so the morphology is inverted relative to
    the usual convention: eroding the image spreads dark pixels.
    """
    if stroke == 0:
        return array
    kernel = np.ones((2, 2), np.uint8)
    if stroke > 0:
        return cv2.erode(array, kernel, iterations=stroke)
    return cv2.dilate(array, kernel, iterations=-stroke)


def _crop_and_fit(array: np.ndarray, height: int) -> np.ndarray:
    """Crop to the ink, then scale to the target height keeping aspect ratio."""
    ink = np.argwhere(array < _INK_THRESHOLD)
    if ink.size == 0:  # nothing was drawn; return a blank strip
        return np.full((height, height), 255, dtype=np.uint8)

    y0, x0 = ink.min(axis=0)
    y1, x1 = ink.max(axis=0) + 1
    cropped = array[y0:y1, x0:x1]

    margin = max(2, height // 16)
    cropped = cv2.copyMakeBorder(
        cropped, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
    )

    h, w = cropped.shape
    new_w = max(1, round(w * height / h))
    return cv2.resize(cropped, (new_w, height), interpolation=cv2.INTER_AREA)


@dataclass
class FixtureSummary:
    root: Path
    num_writers: int
    num_words: int
    forms: list[str]


def build(
    root: Path | str,
    num_writers: int = 20,
    words_per_writer: int = 50,
    height: int = 64,
    seed: int = 0,
    charset_name: str = "english",
) -> FixtureSummary:
    """Generate the whole fixture on disk, in IAM's layout.

        <root>/words/<prefix>/<form-id>/<word-id>.png
        <root>/xml/<form-id>.xml
        <root>/ascii/words.txt

    Existing content is overwritten, so the fixture is always reproducible from
    the seed alone.
    """
    root = Path(root)
    alphabet = cs.get(charset_name)

    vocabulary = [w for w in VOCABULARY if alphabet.supports(w)]
    if not vocabulary:
        raise ValueError(f"charset {charset_name!r} supports none of the vocabulary")

    (root / "words").mkdir(parents=True, exist_ok=True)
    (root / "xml").mkdir(parents=True, exist_ok=True)
    (root / "ascii").mkdir(parents=True, exist_ok=True)

    ascii_lines: list[str] = []
    forms: list[str] = []
    total_words = 0

    for writer_index in range(num_writers):
        style = style_for_writer(writer_index, seed=seed)
        prefix = f"f{writer_index // 100:02d}"
        form_id = f"{prefix}-{writer_index:03d}"
        forms.append(form_id)

        word_dir = root / "words" / prefix / form_id
        word_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(f"{seed}:words:{writer_index}")
        words = [rng.choice(vocabulary) for _ in range(words_per_writer)]

        form = ET.Element("form", {"id": form_id, "writer-id": style.writer_id})
        handwritten = ET.SubElement(form, "handwritten-part")

        # Ten words per line, mirroring IAM's line-then-word nesting.
        for line_index, start in enumerate(range(0, len(words), 10)):
            chunk = words[start : start + 10]
            line_id = f"{form_id}-{line_index:02d}"
            # IAM marks lines whose automatic word segmentation failed. Roughly one
            # line in ten here, deterministically, so the parser's drop path is
            # exercised by the fixture rather than only by real data.
            digest = hashlib.sha256(f"{seed}:seg:{line_id}".encode()).digest()
            segmentation = "err" if digest[0] < 26 else "ok"
            line = ET.SubElement(
                handwritten,
                "line",
                {"id": line_id, "text": " ".join(chunk), "segmentation": segmentation},
            )

            for word_index, text in enumerate(chunk):
                word_id = f"{line_id}-{word_index:02d}"
                image = render_word(text, style, height=height, seed=seed)
                cv2.imwrite(str(word_dir / f"{word_id}.png"), image)

                h, w = image.shape
                word = ET.SubElement(line, "word", {"id": word_id, "text": text})
                ET.SubElement(
                    word,
                    "cmp",
                    {"id": f"{word_id}-00", "x": "0", "y": "0", "width": str(w), "height": str(h)},
                )
                ascii_lines.append(f"{word_id} ok {style.ink} 1 0 0 {w} {h} XX {text}")
                total_words += 1

        ET.ElementTree(form).write(
            root / "xml" / f"{form_id}.xml", encoding="utf-8", xml_declaration=True
        )

    header = [
        "#--- synthetic fixture, not real IAM ---",
        "#word_id status graylevel n_components x y w h tag transcription",
    ]
    (root / "ascii" / "words.txt").write_text(
        "\n".join(header + ascii_lines) + "\n", encoding="utf-8"
    )

    return FixtureSummary(root=root, num_writers=num_writers, num_words=total_words, forms=forms)
