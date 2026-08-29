"""Normalise a photographed page so that only the handwriting survives.

This is the module the project brief names as failure cause number one. The
training and reference data are flatbed scans: white paper, even light, no
perspective. Real input is a phone photo on a desk, at an angle, with a shadow
across it. Feed both to a style encoder without normalising and it learns to
encode *lighting* as style, and every downstream number becomes meaningless.

The pipeline is five steps, each usable on its own so that a failure can be
located rather than guessed at::

    find_page          where is the sheet of paper in this photo
    warp_page          flatten the perspective and rotation
    flatten_light      remove shadows and uneven exposure
    suppress_ruling    remove the printed grid of squared or lined paper
    normalise_ink      bring stroke darkness and contrast to a fixed range

**On ruled paper.** Amri's samples are written on squared notebook paper, which
none of the reference datasets use. The grid is not incidental: to a threshold it
is ink, so word segmentation would carve up grid cells and a style encoder would
learn the ruling as part of his hand. Two properties separate ruling from
handwriting and both are used below -- ruling is *lighter*, and it is *long,
straight and axis-aligned* where strokes are short and curved.

**The invariant this module exists to satisfy.** The same page photographed under
different conditions must come out looking the same. That is testable, and it is
what ``tests/test_preprocessing.py`` measures on the five real photos rather than
asserting by eye.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NormaliseConfig:
    """Every tunable in one place, so an experiment is a config change."""

    work_max_side: int = 1600
    """Longest side used for page-level work. Phone photos are far larger than
    the detail needed, and the operations below are quadratic in area."""

    page_min_area_ratio: float = 0.25
    """A detected quadrilateral smaller than this fraction of the frame is not a
    page; better to keep the whole photo than to crop to a napkin."""

    light_kernel_ratio: float = 0.04
    """Background estimation kernel, as a fraction of the image's shorter side.
    Must be comfortably larger than a stroke is thick, or the strokes themselves
    are absorbed into the background and erased."""

    ruling_length_ratio: float = 0.35
    """A run this long, relative to the image dimension, is ruling and not part
    of a letter. No handwritten stroke is a third of a page long and straight."""

    ruling_max_thickness: int = 3
    """Ruling is printed thin. Anything thicker is treated as ink."""

    paper_percentile: float = 80.0
    """Brightness taken as the paper level, for both ruling and page detection."""

    ruling_contrast: float = 12.0
    """How much darker than paper a pixel must be to be a ruling candidate.
    Deliberately small: printed ruling is faint, and geometry -- not darkness --
    is what separates it from handwriting."""

    ink_percentile: float = 2.0
    """Darkest percentile taken as full ink when stretching contrast. A hard
    minimum would let one dust speck set the black point for the whole page."""

    background_percentile: float = 85.0


DEFAULT = NormaliseConfig()


# ---------------------------------------------------------------------------
# 1. find the page
# ---------------------------------------------------------------------------


def find_page(image: np.ndarray, config: NormaliseConfig = DEFAULT) -> np.ndarray | None:
    """Return the paper's four corners, or None if no convincing sheet is found.

    Brightness, not edges. Canny plus contour approximation is the textbook
    recipe and it failed on every one of the real samples: wood grain produces
    edges as strong as the paper's own border, and the largest quadrilateral was
    a patch of desk. What separates paper from a desk is not that it has edges --
    it is that paper is *bright and smooth* and a desk is neither.

    So: threshold the light-flattened image, close the holes that the writing
    punches in the resulting mask, take the largest component, and fit a rotated
    rectangle to it. That also recovers the rotation for free.

    Returning None rather than guessing is deliberate. A wrong crop silently
    throws away handwriting and warps what is left, while no crop merely leaves
    some desk in the frame, which the later steps tolerate.
    """
    gray = _to_gray(image)
    scale = min(1.0, config.work_max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale < 1.0 else gray

    # Otsu rather than a percentile. A percentile assumes how much of the frame
    # the page occupies: at the 80th, a photo where the page fills a fifth of the
    # frame puts the cutoff below the desk, and the whole image becomes "paper".
    # Otsu splits bright from dark by the shape of the histogram instead, which
    # holds however large or small the sheet is.
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Order matters here, and getting it backwards is what kept the desk in the
    # crop. Roughly a fifth of the wood grain is bright enough to pass the
    # threshold; closing first fuses those specks into a slab that touches the
    # paper, and the two become one component. So: open first to delete the
    # specks, then close to bridge the gaps the writing punches in the sheet.
    speck = max(3, int(min(small.shape) * 0.012) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (speck, speck))
    )
    gap = max(3, int(min(small.shape) * 0.03) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    frame_area = small.shape[0] * small.shape[1]
    if stats[largest, cv2.CC_STAT_AREA] < config.page_min_area_ratio * frame_area:
        return None

    points = cv2.findNonZero((labels == largest).astype(np.uint8))
    corners = _order_corners(cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float32))
    if not _is_plausible_page(corners, small.shape, config):
        return None
    return corners / scale


def _is_plausible_page(corners: np.ndarray, shape: tuple[int, ...], config) -> bool:
    """Reject quadrilaterals that cannot be a sheet of paper.

    A wrong crop is worse than no crop: it silently throws away handwriting and
    warps what remains. Two cheap sanity checks catch most bad detections -- an
    implausible aspect ratio, and opposite sides of wildly different lengths,
    which means the shape is a trapezoid from some object rather than a page seen
    at an angle.
    """
    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    if min(top, bottom, left, right) < 0.15 * max(shape[:2]):
        return False

    aspect = max(top, bottom) / max(1e-6, max(left, right))
    if not 0.35 < aspect < 2.8:
        return False

    # Opposite sides should be within a factor of ~2 of each other. Beyond that
    # the perspective would be more extreme than a phone photo of a desk.
    return not (
        max(top, bottom) / max(1e-6, min(top, bottom)) > 2.0
        or max(left, right) / max(1e-6, min(left, right)) > 2.0
    )


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left.

    Without a fixed order the warp below can mirror or rotate the page, which is
    a spectacular and easily missed failure.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def warp_page(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Flatten the page to a rectangle, undoing perspective and rotation together."""
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 1), max(height, 1)

    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# 2. lighting
# ---------------------------------------------------------------------------


def flatten_light(gray: np.ndarray, config: NormaliseConfig = DEFAULT) -> np.ndarray:
    """Remove shadows and uneven exposure by dividing out an estimated background.

    A morphological closing with a kernel wider than any stroke keeps only what
    varies slowly across the page -- which is exactly the lighting. Dividing by it
    leaves the ink and discards the shadow. This is what makes a photo taken in
    the dark and one taken by a window converge.
    """
    size = max(3, int(min(gray.shape) * config.light_kernel_ratio) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), size / 4.0)

    # +1 avoids a divide by zero on a pure-black background pixel.
    flattened = gray.astype(np.float32) / (background.astype(np.float32) + 1.0)
    return np.clip(flattened * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. ruling
# ---------------------------------------------------------------------------


def detect_ruling(gray: np.ndarray, config: NormaliseConfig = DEFAULT) -> np.ndarray:
    """Mask of printed ruling: long, straight, axis-aligned, thin.

    Returned separately from its removal so the mask can be inspected. If ruling
    removal ever eats handwriting, this is the picture that shows why.
    """
    # Otsu is wrong here, and measurably so: it splits paper from *ink*, landing
    # at 183 on a real sample while the grid sits at 182-232 -- just on the paper
    # side, so the mask came back covering 0.07% of the page. Ruling detection
    # needs a permissive threshold that catches anything meaningfully darker than
    # paper, and then lets geometry decide what is ruling and what is a letter.
    paper = float(np.percentile(gray, config.paper_percentile))
    cutoff = max(1.0, paper - config.ruling_contrast)
    binary = (gray < cutoff).astype(np.uint8) * 255

    height, width = gray.shape
    min_h = max(10, int(width * config.ruling_length_ratio))
    min_v = max(10, int(height * config.ruling_length_ratio))

    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_h, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v))
    )
    mask = cv2.bitwise_or(horizontal, vertical)

    # A stroke that merely happens to lie along a ruling line would be caught by
    # the above. Thick runs are therefore excluded: printed ruling is thin.
    thick = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.ruling_max_thickness + 2, config.ruling_max_thickness + 2),
        ),
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(thick))
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def suppress_ruling(gray: np.ndarray, config: NormaliseConfig = DEFAULT) -> np.ndarray:
    """Paint the detected ruling back to paper colour.

    Inpainting rather than whitening flat, so a stroke that crosses a grid line
    is bridged from its neighbours instead of being cut in two.
    """
    mask = detect_ruling(gray, config)
    if not mask.any():
        return gray
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# 4. ink
# ---------------------------------------------------------------------------


def normalise_ink(gray: np.ndarray, config: NormaliseConfig = DEFAULT) -> np.ndarray:
    """Stretch contrast so ink and paper land at fixed levels on every image.

    Percentiles rather than min and max: one dust speck or one dark pixel at the
    page edge would otherwise set the black point for the whole image, and a page
    photographed in dim light would stay grey.
    """
    dark = float(np.percentile(gray, config.ink_percentile))
    light = float(np.percentile(gray, config.background_percentile))
    if light - dark < 1.0:  # a blank or uniform image; nothing to stretch
        return gray
    stretched = (gray.astype(np.float32) - dark) / (light - dark)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# the whole pipeline
# ---------------------------------------------------------------------------


def normalise_page(
    image: np.ndarray,
    config: NormaliseConfig = DEFAULT,
    remove_ruling: bool = True,
) -> np.ndarray:
    """Photo in, clean grayscale page out.

    Works on a scan too, where page detection simply finds nothing to crop and
    the lighting is already flat -- which is what lets photos and scans be
    compared on equal terms.
    """
    gray = _to_gray(image)

    scale = min(1.0, config.work_max_side / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Lighting is flattened *before* the page is looked for, not after. On the
    # raw photo the desk and the shadowed half of the sheet produce edges as
    # strong as the paper's own, and detection picked the wrong quadrilateral.
    # Once the light is flat the paper is uniformly bright and the desk is not,
    # which is a much easier thing to find.
    gray = flatten_light(gray, config)

    corners = find_page(gray, config)
    if corners is not None:
        gray = warp_page(gray, corners)

    if remove_ruling:
        gray = suppress_ruling(gray, config)
    return normalise_ink(gray, config)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale, but weighted so coloured ink stays dark.

    A plain luminance conversion makes light blue ink nearly as bright as paper.
    Taking the darkest channel keeps blue and black ink dark while leaving white
    paper white -- and blue is the most common pen colour in both CVL and Amri's
    own samples.
    """
    if image.ndim == 2:
        return image
    return image.min(axis=2).astype(np.uint8)


def normalise_word(image: np.ndarray, height: int, config: NormaliseConfig = DEFAULT) -> np.ndarray:
    """Normalise an already-cropped word image to a fixed height.

    Deliberately *not* :func:`normalise_page`. A word crop from a dataset is a few
    dozen pixels tall, already cut from a flat scan: there is no page to find, no
    perspective to undo, and no ruling long enough to detect. Running the page
    pipeline on one would at best waste time and at worst have the background
    estimator swallow the whole word, since the strokes are then a large fraction
    of the image.

    Height is fixed and width follows the aspect ratio, which is what keeps the
    variable-width collate path meaningful.
    """
    gray = _to_gray(image)
    if gray.size == 0:
        raise ValueError("cannot normalise an empty image")

    gray = normalise_ink(gray, config)

    h, w = gray.shape
    width = max(1, round(w * height / h))
    interpolation = cv2.INTER_AREA if h > height else cv2.INTER_CUBIC
    return cv2.resize(gray, (width, height), interpolation=interpolation)
