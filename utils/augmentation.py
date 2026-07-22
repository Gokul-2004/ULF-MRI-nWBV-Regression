"""
Data augmentation utilities for MRI data
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import rotate, zoom, shift


class MRIAugmentation:
    """Data augmentation for 3D MRI volumes"""

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-15, 15),
        translation_range: Tuple[float, float] = (-10, 10),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.05,
        flip_prob: float = 0.5,
        intensity_shift: float = 0.0,
        intensity_scale: Tuple[float, float] = (1.0, 1.0),
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply random augmentations"""
        # Random rotation
        if self.rotation_range[1] > 0:
            angle = np.random.uniform(*self.rotation_range)
            volume = self._rotate(volume, angle)

        # Random translation
        if self.translation_range[1] > 0:
            translation = np.random.uniform(*self.translation_range, size=3)
            volume = self._translate(volume, translation)

        # Random scaling
        if self.scale_range[1] != 1.0:
            scale = np.random.uniform(*self.scale_range)
            volume = self._scale(volume, scale)

        # Random flip
        if np.random.random() < self.flip_prob:
            axis = np.random.randint(0, 3)
            volume = np.flip(volume, axis=axis)

        # Random intensity shift (brightness)
        if self.intensity_shift > 0:
            shift_val = np.random.uniform(-self.intensity_shift, self.intensity_shift)
            volume = volume + shift_val

        # Random intensity scale (contrast)
        if self.intensity_scale[0] != 1.0 or self.intensity_scale[1] != 1.0:
            scale_val = np.random.uniform(*self.intensity_scale)
            mean_val = volume.mean()
            volume = (volume - mean_val) * scale_val + mean_val

        # Add noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, volume.shape)
            volume = volume + noise

        return np.clip(volume, 0, 1)

    def _rotate(self, volume: np.ndarray, angle: float) -> np.ndarray:
        """Rotate volume around random axis"""
        axis = np.random.randint(0, 3)
        return rotate(volume, angle, axes=((axis+1)%3, (axis+2)%3), reshape=False, order=1)

    def _translate(self, volume: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Translate volume"""
        return shift(volume, translation, order=1, mode='constant', cval=0)

    def _scale(self, volume: np.ndarray, scale: float) -> np.ndarray:
        """Scale volume"""
        new_shape = tuple(int(s * scale) for s in volume.shape)
        factors = [n / o for n, o in zip(new_shape, volume.shape)]
        scaled = zoom(volume, factors, order=1)
        if scaled.shape != volume.shape:
            result = np.zeros_like(volume)
            slices = tuple(slice(0, min(s, o)) for s, o in zip(scaled.shape, volume.shape))
            result[slices] = scaled[slices]
            return result
        return scaled


def get_augmentation(config: dict, split: str = "train"):
    """Get augmentation based on config and split"""
    if split != "train":
        return None  # No augmentation for val/test

    aug_config = config.get('augmentation', {})
    if not aug_config.get('enabled', False):
        return None

    return MRIAugmentation(
        rotation_range=tuple(aug_config.get('rotation_range', (-15, 15))),
        translation_range=tuple(aug_config.get('translation_range', (-10, 10))),
        scale_range=tuple(aug_config.get('scale_range', (0.9, 1.1))),
        noise_std=aug_config.get('noise_std', 0.05),
        flip_prob=aug_config.get('flip_prob', 0.5),
        intensity_shift=aug_config.get('intensity_shift', 0.0),
        intensity_scale=tuple(aug_config.get('intensity_scale', (1.0, 1.0))),
    )
