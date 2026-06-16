"""Augmentation and normalization for sonar tensors and optical RGB images."""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SonarTransform:
    """Sonar tile augmentations (single-channel, no ImageNet normalization)."""

    def __init__(
        self,
        augment: bool,
        img_size: int,
        noise_std: float,
        crop_scale_min: float,
        crop_scale_max: float,
        crop_ratio_min: float,
        crop_ratio_max: float,
        hflip_p: float,
    ) -> None:
        """Initialize transforms from config scalars.

        Args:
            augment: Whether to apply random augmentations.
            img_size: Spatial size (square).
            noise_std: Gaussian noise standard deviation after scaling to [0, 1].
            crop_scale_min: Minimum scale for random resized crop.
            crop_scale_max: Maximum scale for random resized crop.
            crop_ratio_min: Minimum aspect ratio for random resized crop.
            crop_ratio_max: Maximum aspect ratio for random resized crop.
            hflip_p: Probability of horizontal flip.
        """
        self.augment = augment
        self.img_size = img_size
        self.noise_std = noise_std
        self.crop_scale_min = crop_scale_min
        self.crop_scale_max = crop_scale_max
        self.crop_ratio_min = crop_ratio_min
        self.crop_ratio_max = crop_ratio_max
        self.hflip_p = hflip_p

    def _clamp01(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp tensor values to [0, 1]."""
        return torch.clamp(x, 0.0, 1.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sonar transforms. Input/output shape ``(1, H, W)``, float32.

        Pipeline:
          1. log1p compression  — compresses heavy-tailed acoustic backscatter
          2. per-tile min-max   — maps each tile to [0, 1] independently
          3. spatial augmentation (train only)
          4. Gaussian noise (train only)

        Args:
            x: Raw sonar tensor (arbitrary positive floats from NPY).

        Returns:
            Transformed sonar tensor ``(1, img_size, img_size)`` in [0, 1].
        """
        x = x.float()

        # ── 1. log1p compression ────────────────────────────────────────────
        # Sonar backscatter is log-normally distributed; log1p maps it to a
        # roughly Gaussian distribution expected by the ViT patch embedder.
        x = torch.log1p(x.clamp(min=0.0))

        # ── 2. per-tile min-max normalisation ───────────────────────────────
        # Normalize each tile independently so intensity baseline differences
        # between survey tracks do not become confounders in the embedding space.
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)

        # ── 3. Spatial augmentation (train only) ────────────────────────────
        if self.augment:
            if random.random() < self.hflip_p:
                x = TF.hflip(x)

            i, j, h, w = T.RandomResizedCrop.get_params(
                x,
                scale=(self.crop_scale_min, self.crop_scale_max),
                ratio=(self.crop_ratio_min, self.crop_ratio_max),
            )
            x = TF.crop(x, i, j, h, w)
            x = TF.resize(x, [self.img_size, self.img_size], antialias=True)

            # ── 4. Gaussian noise ────────────────────────────────────────────
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
        else:
            if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
                x = TF.resize(x, [self.img_size, self.img_size], antialias=True)

        x = self._clamp01(x).float()
        return x



class OpticalTransform:
    """Optical image augmentations with ImageNet normalization."""

    def __init__(
        self,
        augment: bool,
        out_size: int,
        hflip_p: float,
        color_jitter_p: float,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        grayscale_p: float,
        blur_p: float,
        blur_kernel: int,
    ) -> None:
        """Initialize optical transforms from config.

        Args:
            augment: Whether to apply strong augmentations.
            out_size: Output square side (e.g. 224).
            hflip_p: Horizontal flip probability.
            color_jitter_p: Probability of applying color jitter.
            brightness: Color jitter brightness factor range (symmetric).
            contrast: Color jitter contrast factor range.
            saturation: Color jitter saturation factor range.
            hue: Color jitter hue fraction range.
            grayscale_p: Random grayscale probability.
            blur_p: Gaussian blur probability.
            blur_kernel: Gaussian blur kernel size (odd).
        """
        self.augment = augment
        self.out_size = out_size
        self.hflip_p = hflip_p
        self.color_jitter_p = color_jitter_p
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.grayscale_p = grayscale_p
        self.blur_p = blur_p
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optical transforms. Input ``(3, H, W)`` in [0,1].

        Args:
            x: RGB tensor.

        Returns:
            Tensor ``(3, out_size, out_size)`` normalized with ImageNet stats.
        """
        x = x.float()
        x = torch.clamp(x, 0.0, 1.0)

        if self.augment:
            i, j, h, w = T.RandomResizedCrop.get_params(
                x, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            x = TF.crop(x, i, j, h, w)
            x = TF.resize(x, [self.out_size, self.out_size], antialias=True)

            if random.random() < self.hflip_p:
                x = TF.hflip(x)
            if random.random() < self.color_jitter_p:
                x = self.color_jitter(x)
            if random.random() < self.grayscale_p:
                x = TF.rgb_to_grayscale(x, num_output_channels=3)
            if random.random() < self.blur_p:
                x = TF.gaussian_blur(x, kernel_size=[self.blur_kernel, self.blur_kernel])
        else:
            x = TF.resize(x, [self.out_size, self.out_size], antialias=True)

        x = TF.normalize(x, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
        return x


def default_sonar_transform(augment: bool, img_size: int) -> SonarTransform:
    """Factory for sonar transforms with GeoCAM default hyperparameters."""
    return SonarTransform(
        augment=augment,
        img_size=img_size,
        noise_std=0.01,
        crop_scale_min=0.8,
        crop_scale_max=1.0,
        crop_ratio_min=0.9,
        crop_ratio_max=1.1,
        hflip_p=0.5,
    )


def default_optical_transform(augment: bool, out_size: int) -> OpticalTransform:
    """Factory for optical transforms with GeoCAM default hyperparameters."""
    return OpticalTransform(
        augment=augment,
        out_size=out_size,
        hflip_p=0.5,
        color_jitter_p=0.8,
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1,
        grayscale_p=0.2,
        blur_p=0.5,
        blur_kernel=5,
    )
