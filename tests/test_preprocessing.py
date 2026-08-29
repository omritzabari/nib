"""Tests for page normalisation.

The claim this module makes is that the *conditions a photo was taken under* stop
mattering. That is measurable, and the tests at the bottom measure it on Amri's
five real photographs rather than asserting it by eye.

Everything above them is a synthetic check on one step at a time, so that when
the real-photo tests move, the cause can be located instead of guessed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nib.config import find_repo_root
from nib.data.preprocessing import (
    DEFAULT,
    NormaliseConfig,
    _to_gray,
    detect_ruling,
    find_page,
    flatten_light,
    normalise_ink,
    normalise_page,
    suppress_ruling,
    warp_page,
)

PHOTOS = find_repo_root() / "data" / "raw" / "personal"
_REAL = [p for p in sorted(PHOTOS.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
needs_photos = pytest.mark.skipif(len(_REAL) < 3, reason=f"fewer than 3 photographs under {PHOTOS}")


# ---------------------------------------------------------------------------
# synthetic builders
# ---------------------------------------------------------------------------


def blank_page(h=400, w=300, value=245) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def with_writing(page: np.ndarray) -> np.ndarray:
    """Short curved dark strokes -- the thing that must survive every step."""
    out = page.copy()
    for y in range(60, page.shape[0] - 60, 45):
        for x in range(30, page.shape[1] - 60, 70):
            cv2.ellipse(out, (x + 20, y), (16, 11), 20, 0, 260, 25, 3)
    return out


def with_ruling(page: np.ndarray, step=20) -> np.ndarray:
    """A printed grid: long, straight, axis-aligned, thin, and lighter than ink."""
    out = page.copy()
    for y in range(0, page.shape[0], step):
        cv2.line(out, (0, y), (page.shape[1], y), 205, 1)
    for x in range(0, page.shape[1], step):
        cv2.line(out, (x, 0), (x, page.shape[0]), 205, 1)
    return out


def with_shadow(page: np.ndarray) -> np.ndarray:
    gradient = np.linspace(1.0, 0.35, page.shape[0], dtype=np.float32)[:, None]
    return np.clip(page.astype(np.float32) * gradient, 0, 255).astype(np.uint8)


def on_desk(page: np.ndarray, margin=90, desk=95) -> np.ndarray:
    """A page on a textured dark surface, in colour, as a photo would be."""
    h, w = page.shape
    rng = np.random.default_rng(0)
    frame = (desk + rng.normal(0, 22, (h + 2 * margin, w + 2 * margin))).clip(0, 255)
    frame = frame.astype(np.uint8)
    frame[margin : margin + h, margin : margin + w] = page
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# lighting
# ---------------------------------------------------------------------------


def test_flatten_light_removes_a_gradient():
    page = with_writing(blank_page())
    lit = with_shadow(page)
    assert page[:50].mean() - page[-50:].mean() < 25, "precondition: the page is evenly lit"
    assert lit[:50].mean() - lit[-50:].mean() > 80, "precondition: the shadow is strong"

    flat = flatten_light(lit)
    assert abs(flat[:50].mean() - flat[-50:].mean()) < 25, "the gradient survived"


def test_flatten_light_keeps_the_writing():
    """The failure mode: a background kernel smaller than a stroke absorbs the
    strokes into the background and erases them."""
    page = with_writing(blank_page())
    flat = flatten_light(page)
    assert (flat < 128).sum() > 0.3 * (page < 128).sum(), "the writing was erased"


def test_flatten_light_does_not_invent_ink_on_a_blank_page():
    flat = flatten_light(blank_page())
    assert (flat < 128).mean() < 0.01


# ---------------------------------------------------------------------------
# ruling
# ---------------------------------------------------------------------------


def test_ruling_is_detected_on_squared_paper():
    """Regression guard on a real bug: Otsu split paper from *ink*, putting the
    threshold at 183 while the grid sat at 182-232, and the mask came back
    covering 0.07% of the page -- that is, nothing."""
    ruled = with_ruling(blank_page())
    mask = detect_ruling(ruled)
    assert mask.mean() / 255 > 0.02, "the ruling detector found essentially nothing"


def test_handwriting_is_not_mistaken_for_ruling():
    """The dangerous direction. Erasing strokes is far worse than leaving grid."""
    page = with_writing(blank_page())
    mask = detect_ruling(page)
    assert mask.mean() / 255 < 0.01, "the detector is eating handwriting"


def test_suppressing_ruling_removes_grid_and_spares_strokes():
    page = with_writing(blank_page())
    ruled = with_ruling(page)
    ink_before = (page < 128).sum()

    cleaned = suppress_ruling(ruled)
    assert (cleaned < 215).sum() < 0.6 * (ruled < 215).sum(), "the grid is still there"
    assert (cleaned < 128).sum() > 0.7 * ink_before, "strokes were removed with the grid"


def test_a_blank_unruled_page_is_left_alone():
    page = blank_page()
    assert np.array_equal(suppress_ruling(page), page)


# ---------------------------------------------------------------------------
# page detection
# ---------------------------------------------------------------------------


def test_a_page_on_a_desk_is_found_and_cropped_out():
    photo = on_desk(with_writing(blank_page(400, 300)))
    corners = find_page(photo)
    assert corners is not None, "the page was not found"

    cropped = warp_page(_to_gray(photo), corners)
    assert 0.5 < (cropped.shape[1] / cropped.shape[0]) / (300 / 400) < 2.0
    assert cropped.mean() > _to_gray(photo).mean(), "the crop kept more desk than page"


def test_no_page_is_reported_rather_than_a_wrong_crop():
    """A wrong crop silently deletes handwriting. Uniform noise contains no page,
    and the honest answer is None."""
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    assert find_page(noise) is None


def test_an_implausibly_shaped_quad_is_rejected():
    """A long thin bright strip is not a sheet of paper."""
    frame = np.full((400, 400), 60, dtype=np.uint8)
    frame[190:210, 20:380] = 250
    assert find_page(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)) is None


def test_corner_ordering_is_fixed():
    """Unordered corners let the warp mirror or rotate the page -- spectacular,
    and easy to miss."""
    photo = on_desk(with_writing(blank_page(400, 300)))
    corners = find_page(photo)
    assert corners is not None
    tl, tr, br, bl = corners
    assert tl[0] < tr[0] and bl[0] < br[0], "left corners are not left of right ones"
    assert tl[1] < bl[1] and tr[1] < br[1], "top corners are not above bottom ones"


# ---------------------------------------------------------------------------
# ink
# ---------------------------------------------------------------------------


def test_normalise_ink_stretches_a_flat_image_to_the_full_range():
    faint = np.full((100, 100), 200, dtype=np.uint8)
    faint[40:60, 40:60] = 150
    stretched = normalise_ink(faint)
    assert stretched.min() < 40 and stretched.max() > 215


def test_one_dark_speck_does_not_set_the_black_point():
    """Percentiles rather than min and max, so a dust mote cannot decide the
    contrast of a whole page."""
    page = np.full((200, 200), 200, dtype=np.uint8)
    page[100, 100] = 0
    stretched = normalise_ink(page)
    assert np.array_equal(stretched, page), (
        "one speck was allowed to set the contrast for the whole page"
    )


def test_a_uniform_image_is_returned_unchanged():
    flat = np.full((50, 50), 128, dtype=np.uint8)
    assert np.array_equal(normalise_ink(flat), flat)


# ---------------------------------------------------------------------------
# colour handling
# ---------------------------------------------------------------------------


def test_blue_ink_stays_dark_in_grayscale():
    """Luminance conversion makes blue ink nearly as bright as paper, and blue is
    the pen colour in both CVL and Amri's own samples."""
    image = np.full((10, 10, 3), 250, dtype=np.uint8)
    image[5, 5] = (200, 60, 60)  # BGR: blue ink
    gray = _to_gray(image)
    assert gray[5, 5] < 100, f"blue ink came out at {gray[5, 5]}, nearly paper-white"


# ---------------------------------------------------------------------------
# the whole pipeline, on the real photographs
# ---------------------------------------------------------------------------


def _measure(gray: np.ndarray) -> dict[str, float]:
    paper = float(np.percentile(gray, 90))
    ink = float(np.percentile(gray, 2))
    return {"ink %": 100.0 * float((gray < 128).mean()), "paper": paper, "contrast": paper - ink}


@needs_photos
def test_lighting_conditions_converge_on_the_real_photographs():
    """The claim, measured. Statistics that should not depend on how a photo was
    taken must spread less after normalising than before.

    Deliberately not a pixel comparison: two photos of the same page at different
    angles never align, so a pixel metric would measure alignment, not lighting.
    """
    before = {p.stem: _measure(_to_gray(cv2.imread(str(p)))) for p in _REAL}
    after = {p.stem: _measure(normalise_page(cv2.imread(str(p)))) for p in _REAL}

    def spread(rows, key):
        values = [row[key] for row in rows.values()]
        return max(values) - min(values)

    for key in ["ink %", "paper", "contrast"]:
        b, a = spread(before, key), spread(after, key)
        print(f"\n{key}: spread {b:.1f} -> {a:.1f}")
        assert a <= b, f"{key} spread grew from {b:.1f} to {a:.1f}"

    assert spread(after, "paper") < 5.0, "paper level still depends on the photo"
    assert spread(after, "contrast") < 5.0, "contrast still depends on the photo"


@needs_photos
def test_every_real_photograph_yields_a_plausible_page():
    for path in _REAL:
        out = normalise_page(cv2.imread(str(path)))
        assert out.ndim == 2 and out.dtype == np.uint8
        assert min(out.shape) > 200, f"{path.name} normalised to {out.shape}"
        ink = (out < 128).mean()
        assert 0.005 < ink < 0.30, f"{path.name} has {ink:.1%} ink, which is implausible"


@needs_photos
def test_normalisation_is_deterministic():
    image = cv2.imread(str(_REAL[0]))
    assert np.array_equal(normalise_page(image), normalise_page(image))


@needs_photos
def test_the_light_kernel_size_matters():
    """The default is a real choice, not an arbitrary number: changing it changes
    the result. An earlier version of this test asserted that a tiny kernel
    *erases* strokes, which measurement disproved -- on these photos it adds noise
    instead. The claim was replaced rather than the number tuned to fit it."""
    image = cv2.imread(str(_REAL[0]))
    good = (normalise_page(image) < 128).mean()
    tiny = (normalise_page(image, config=NormaliseConfig(light_kernel_ratio=0.002)) < 128).mean()
    assert abs(tiny - good) > 0.002, "kernel size has no effect, so the default is unjustified"
    assert DEFAULT.light_kernel_ratio > 0.01
