"""Fréchet Inception Distance: do the generated images look like handwriting?

FID summarises a *set* of images as a Gaussian in the feature space of a
pre-trained vision model, then measures the distance between two such Gaussians.
Lower is better. It says nothing about whether the text is correct or whether the
style matches the right person -- which is exactly why this project reports three
metrics rather than one.

Two things about FID that are easy to get wrong and are handled here.

**It is biased by sample count.** FID computed on 500 images is systematically
higher than on 5000 of the same distribution, so two runs are only comparable if
they used the same number of samples. The count is therefore part of the result
and printed with it, rather than left as a footnote nobody records.

**The covariance term needs a matrix square root**, which is numerically fragile:
scipy's sqrtm returns a small imaginary component on nearly-singular matrices.
Discarding it silently is standard practice, but the size of what is discarded is
checked here rather than assumed to be negligible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MIN_SAMPLES = 50


class FidError(RuntimeError):
    pass


@dataclass
class FidResult:
    value: float
    num_real: int
    num_generated: int
    imaginary_residual: float = 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                f"FID              {self.value:.2f}",
                f"samples          {self.num_generated} generated vs {self.num_real} real",
                "                 (FID falls as sample count rises -- only compare "
                "runs with equal counts)",
            ]
        )


def gaussian_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and covariance matrix of a feature set."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise FidError(f"expected features of shape (n, d), got {features.shape}")
    if features.shape[0] < 2:
        raise FidError("need at least two samples to estimate a covariance")
    return features.mean(axis=0), np.cov(features, rowvar=False)


def frechet_distance(
    mu_a: np.ndarray,
    sigma_a: np.ndarray,
    mu_b: np.ndarray,
    sigma_b: np.ndarray,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Fréchet distance between two Gaussians. Returns the value and the size of
    the imaginary residual discarded from the matrix square root."""
    # scipy dropped sqrtm's `disp` argument, and with it the (matrix, error)
    # tuple, at 1.18. Most FID implementations still pass disp=False and would
    # raise here. Handle both shapes rather than pinning a scipy version.
    from scipy import linalg

    def _sqrtm(matrix: np.ndarray) -> np.ndarray:
        result = linalg.sqrtm(matrix)
        return result[0] if isinstance(result, tuple) else result

    diff = mu_a - mu_b
    covmean = _sqrtm(sigma_a.dot(sigma_b))

    if not np.isfinite(covmean).all():
        # Nearly-singular product: nudge the diagonals and retry. Standard
        # practice, and the offset is tiny relative to the feature scale.
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = _sqrtm((sigma_a + offset).dot(sigma_b + offset))

    residual = 0.0
    if np.iscomplexobj(covmean):
        residual = float(np.max(np.abs(covmean.imag)))
        covmean = covmean.real

    value = float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))
    return max(value, 0.0), residual


def compute_fid(real_features: np.ndarray, generated_features: np.ndarray) -> FidResult:
    """FID between two feature sets.

    Refuses to answer on tiny sets. The estimate is so biased below a few hundred
    samples that a number would be actively misleading, and a loud refusal is
    better than a figure someone puts in a README.
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    generated_features = np.asarray(generated_features, dtype=np.float64)

    if real_features.shape[1:] != generated_features.shape[1:]:
        raise FidError(
            f"feature dimensions differ: {real_features.shape[1:]} vs "
            f"{generated_features.shape[1:]}"
        )
    for name, features in (("real", real_features), ("generated", generated_features)):
        if len(features) < MIN_SAMPLES:
            raise FidError(
                f"only {len(features)} {name} samples. FID is strongly biased below "
                f"{MIN_SAMPLES} and the number would mislead more than it informs."
            )

    mu_r, sigma_r = gaussian_statistics(real_features)
    mu_g, sigma_g = gaussian_statistics(generated_features)
    value, residual = frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

    if residual > 1e-3:
        raise FidError(
            f"the matrix square root left an imaginary residual of {residual:.2e}, "
            "which is too large to discard. The covariance estimate is unreliable -- "
            "usually too few samples for the feature dimension."
        )

    return FidResult(
        value=value,
        num_real=len(real_features),
        num_generated=len(generated_features),
        imaginary_residual=residual,
    )


class InceptionFeatures:
    """Pool3 features from ImageNet InceptionV3, the standard FID backbone.

    Handwriting is grayscale and wide; Inception wants 299x299 RGB. The images are
    therefore replicated across channels and resized. That is what every FID
    implementation does, and it is worth knowing that it is happening: a metric
    built on a network trained on photographs is a proxy, not a measurement.
    """

    def __init__(self, device: str = "cpu") -> None:
        import torch
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.torch = torch
        self.device = torch.device(device)
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()  # keep the 2048-d pool features
        self.model = model.eval().to(self.device)

    def __call__(self, images: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        out = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunk = images[start : start + batch_size]
                tensors = []
                for image in chunk:
                    array = np.asarray(image, dtype=np.float32)
                    if array.max() > 1.5:
                        array = array / 255.0
                    tensor = torch.from_numpy(array)[None, None]
                    tensor = F.interpolate(
                        tensor, size=(299, 299), mode="bilinear", align_corners=False
                    )
                    tensors.append(tensor.repeat(1, 3, 1, 1))
                batch = torch.cat(tensors).to(self.device)
                out.append(self.model(batch).cpu().numpy())
        return np.concatenate(out, axis=0)
