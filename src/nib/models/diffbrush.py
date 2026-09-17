"""DiffBrush behind the project's Generator interface.

DiffBrush (Dai et al., *Beyond Isolated Words: Diffusion Brush for Handwritten
Text-Line Generation*, ICCV 2025, MIT) is a latent diffusion model that writes a
whole line: a UNet denoises Stable Diffusion 1.5's VAE latents over 50 DDIM steps,
conditioned on one style line and on the text drawn as Unifont glyphs.

It is here because of what Emuru could not do. On Amri's page Emuru dropped
i-dots, colons and short strokes, and three pipeline causes were ruled out; the
likely one is that Emuru regresses each slice to the average of what could come
next, and the average of an uncertain dot is nothing. A diffusion model samples
instead of averaging.

Five things about it matter before reading any number it produces.

**It was trained on IAM alone, and never tested off it.** Its authors train one
model per dataset and release only the IAM one. Every IAM-trained model measured
off IAM so far lost its imitation there. CVL writers and Amri are off IAM.

**It does not need the style line's transcription.** Emuru does, which puts a
recogniser in front of the product.

**Its canvas is fixed at 64 x 1024.** Style lines are cut to their first 1024
pixels, as its own loader does; the output is always 1024 wide and is cut back
here to the last ink. Ink that reaches the right edge is counted as truncated.

**Its charset is 80 characters** (:data:`CHARSET`). Anything else is refused
before the model runs, rather than drawn as a wrong glyph.

**The code is not vendored.** The repository is cloned at a pinned commit (see
:data:`PINNED_COMMIT`) and its directory is passed in, from the config. Its
``models`` package is imported from there, and the adapter checks that the
``models`` it got is the one from that directory, not some other package of the
same generic name.
"""

from __future__ import annotations

import contextlib
import pickle
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from nib.models.emuru import Truncation, TruncationLog
from nib.models.generator import EmptyGeneration, GenerationRequest, GeneratorError

REPOSITORY = "https://github.com/dailenson/DiffBrush"
PINNED_COMMIT = "da9addc"
"""The commit the adapter was written against, 2025-11-24."""

VAE_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
"""Stable Diffusion 1.5, whose VAE DiffBrush denoises in; the ``vae`` subfolder."""

NATIVE_HEIGHT = 64
CANVAS_WIDTH = 1024
LATENT_FACTOR = 8
"""The VAE's spatial compression: the latent canvas is 8 x 128."""

DDIM_STEPS = 50
"""The released generation script's default, and the paper's."""

CHARSET = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
"""From its ``IAMDataset.py``. Covers every line of CVL and both dictated passages."""

UNET = {
    "in_channels": 4,
    "model_channels": 512,
    "out_channels": 4,
    "num_res_blocks": 1,
    "attention_resolutions": (1, 1),
    "channel_mult": (1, 1),
    "num_heads": 4,
    "context_dim": 512,
    "nb_classes": 496,
}
"""The released checkpoint's architecture: ``configs/IAM.yml`` and ``generate.py``.
``nb_classes`` is IAM's training writers, used only by a training loss; the
checkpoint loads with no missing and no unexpected keys under exactly this."""

INK_THRESHOLD = 128
"""Below this a pixel is ink when finding where the written line ends. Decoder
noise in the empty part of the canvas sits well above it."""

EDGE = 4
"""Ink this close to the canvas's right edge means the line did not fit."""

CODE_DIRNAME = "DiffBrush"
"""The clone's name under ``paths.third_party``."""

CHECKPOINT = ("diffbrush", "DiffBrush-ckpt.pt")
"""The released checkpoint's place under ``paths.checkpoints``."""


def locate(cfg) -> tuple[Path, Path]:
    """Where the config says the DiffBrush code and checkpoint are."""
    from nib.config import get_path

    return (
        get_path(cfg, "third_party") / CODE_DIRNAME,
        get_path(cfg, "checkpoints").joinpath(*CHECKPOINT),
    )


def load_glyphs(path: Path | str) -> np.ndarray:
    """The 16x16 Unifont glyph of every :data:`CHARSET` character, in charset order."""
    with open(path, "rb") as handle:
        table = {entry["idx"][0]: entry["mat"] for entry in pickle.load(handle)}
    return np.stack([np.asarray(table[ord(char)], np.float32) for char in CHARSET])


def encode_text(text: str, glyphs: np.ndarray) -> np.ndarray:
    """The text as the model reads it: one inverted glyph per character.

    Inverted because the released loader does ``1.0 - glyph``; the model never saw
    them the other way round.
    """
    missing = sorted({char for char in text if char not in CHARSET})
    if missing:
        raise GeneratorError(
            f"DiffBrush cannot write {', '.join(repr(char) for char in missing)} "
            f"in {text!r}: its charset has 80 characters."
        )
    return 1.0 - glyphs[[CHARSET.index(char) for char in text]]


def prepare_style(
    image: np.ndarray, height: int = NATIVE_HEIGHT, width: int = CANVAS_WIDTH
) -> np.ndarray:
    """A style line as the model takes it: ``height`` high, at most ``width`` wide,
    ink dark in [0, 1]. Wider lines keep their first ``width`` pixels, as the
    released loader does, rather than being squeezed."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise GeneratorError(f"style image must be grayscale, got shape {array.shape}")
    if array.shape[0] != height:
        scale = height / array.shape[0]
        array = cv2.resize(
            array,
            (max(1, round(array.shape[1] * scale)), height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    return array[:, :width].astype(np.float32) / 255.0


def crop_output(
    canvas: np.ndarray, margin: int = 8, ink_threshold: int = INK_THRESHOLD
) -> tuple[np.ndarray, bool] | None:
    """The written line cut from the fixed canvas, and whether it ran to the edge.

    None when the canvas holds no ink at all -- the caller's to report as an empty
    generation, never to pass on as a blank image.
    """
    columns = np.flatnonzero((np.asarray(canvas) < ink_threshold).any(axis=0))
    if columns.size == 0:
        return None
    last = int(columns[-1])
    truncated = last >= canvas.shape[1] - EDGE
    return canvas[:, : min(canvas.shape[1], last + 1 + margin)], truncated


@contextlib.contextmanager
def _cuda_calls_stay_on_cpu(torch) -> Iterator[None]:
    """The released model calls ``.cuda()`` on training-only proxies while it is
    built. Without a GPU that raises; this makes the call a no-op for the duration
    of construction, and only when there is no GPU."""
    if torch.cuda.is_available():
        yield
        return
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.cuda = original


class DiffBrushGenerator:
    """The released DiffBrush checkpoint, adapted to this project's interface.

    Uses one style line per request -- the first -- because that is how the model
    was trained. Several lines are a pool for :class:`nib.models.candidates.CandidateGenerator`
    to draw from, one per draw, never an image to join.
    """

    def __init__(
        self,
        code_dir: Path | str,
        checkpoint: Path | str,
        device: str = "cpu",
        output_height: int = NATIVE_HEIGHT,
        steps: int = DDIM_STEPS,
        seed: int | None = None,
        vae_id: str = VAE_ID,
    ) -> None:
        try:
            import torch
            from diffusers import AutoencoderKL
        except ImportError as exc:  # pragma: no cover
            raise GeneratorError("DiffBrush needs torch and diffusers") from exc

        code_dir = Path(code_dir)
        if not (code_dir / "models" / "unet.py").is_file():
            raise GeneratorError(
                f"no DiffBrush code in {code_dir}. Clone {REPOSITORY} there and check out "
                f"{PINNED_COMMIT}."
            )
        if not Path(checkpoint).is_file():
            raise GeneratorError(f"no DiffBrush checkpoint at {checkpoint}")

        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from models.diffusion import Diffusion
        from models.unet import UNetModel

        # Its ``models`` has no __init__.py, so ask the module file, not the package.
        imported_from = Path(sys.modules["models.unet"].__file__).resolve().parent
        if imported_from != (code_dir / "models").resolve():
            raise GeneratorError(
                f"'models' was imported from {imported_from}, not from {code_dir}: another "
                "package of that name is on the path."
            )

        self.torch = torch
        self.device = torch.device(device)
        self.steps = steps
        self._output_height = output_height
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)
        self.truncations = TruncationLog()

        with _cuda_calls_stay_on_cpu(torch):
            unet = UNetModel(**UNET)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        unet.load_state_dict(state, strict=True)
        self.unet = unet.to(self.device).eval()
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae").to(self.device).eval()
        self.diffusion = Diffusion(device=self.device)
        self.glyphs = load_glyphs(code_dir / "files" / "unifont.pickle")

    @property
    def name(self) -> str:
        return "diffbrush"

    @property
    def output_height(self) -> int:
        return self._output_height

    def generate(self, requests: Sequence[GenerationRequest]) -> list[np.ndarray]:
        """One image per request, in order. The text is checked against the
        charset for every request before any is drawn."""
        contents = [encode_text(request.text, self.glyphs) for request in requests]
        return [
            self._generate_one(request, content)
            for request, content in zip(requests, contents, strict=True)
        ]

    def _generate_one(self, request: GenerationRequest, content: np.ndarray) -> np.ndarray:
        torch = self.torch
        style = prepare_style(request.style_images[0])
        noise = torch.randn(
            (
                1,
                self.unet.in_channels,
                NATIVE_HEIGHT // LATENT_FACTOR,
                CANVAS_WIDTH // LATENT_FACTOR,
            ),
            generator=self._generator,
        ).to(self.device)
        with torch.no_grad():
            sampled = self.diffusion.ddim_sample(
                self.unet,
                self.vae,
                1,
                noise,
                torch.from_numpy(style)[None, None].to(self.device),
                torch.from_numpy(content)[None].to(self.device),
                self.steps,
                0.0,
            )
        canvas = (sampled[0].mean(dim=0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)

        cropped = crop_output(canvas)
        self.truncations.generated += 1
        if cropped is None:
            raise EmptyGeneration(f"DiffBrush drew no ink for {request.text!r}")
        image, truncated = cropped
        if truncated:
            self.truncations.events.append(
                Truncation(
                    text=request.text,
                    width=int(image.shape[1]),
                    budget=CANVAS_WIDTH // LATENT_FACTOR,
                )
            )
        if self._output_height != NATIVE_HEIGHT:
            scale = self._output_height / NATIVE_HEIGHT
            image = cv2.resize(
                image,
                (max(1, round(image.shape[1] * scale)), self._output_height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
        return image
