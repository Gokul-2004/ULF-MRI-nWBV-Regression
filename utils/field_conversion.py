"""
High-field to Low-field MRI conversion utilities
Physics-informed simulation of Hyperfine Swoop 64mT scanner characteristics.

Parameters measured directly from Zenodo paired dataset BIDS JSON sidecars
(van den Broek et al., 2025 — Hyperfine Swoop scanner at LUMC):

  T2w sequence:  TR=2000ms, TE=194.8ms, ETL=80, res=1.6×1.6×5.0mm
  FLAIR sequence: TR=3500ms, TE=162ms, TI=1301ms, ETL=68, res=1.7×1.7×5.0mm

  Field strength: 64 mT (0.064 T)
  Manufacturer: Hyperfine, Inc. — model: swoop
  Rician noise: dominant noise model in magnitude MRI images
  B0 inhomogeneity: poor shimming → smooth spatial signal variation
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def gaussian_filter(*args, **kwargs):
        raise ImportError("scipy is required. Install with: pip install scipy")


# ── Measured Hyperfine Swoop 64mT parameters (from BIDS JSON sidecars) ───────
HYPERFINE_FIELD_T    = 0.064      # T
REFERENCE_FIELD_T    = 3.0        # T

# T2w sequence (primary sequence used for our simulation)
T2W_TR_MS            = 2000.0     # ms  — measured from scanner
T2W_TE_MS            = 194.8      # ms  — measured from scanner
T2W_INPLANE_MM       = 1.6        # mm  — measured from scanner (PixelSpacing)
T2W_SLICE_MM         = 5.0        # mm  — measured from scanner (SliceThickness)
HIRES_RES_MM         = 1.0        # mm  — typical 3T isotropic reference

# Relaxation times at 64mT vs 3T (from published literature)
# T1 shortens at low field; T2 is similar or slightly longer (less susceptibility)
T1_WM_3T    = 832.0;  T1_GM_3T  = 1330.0;  T1_CSF_3T  = 4000.0   # ms at 3T
T1_WM_64MT  = 400.0;  T1_GM_64MT = 500.0;  T1_CSF_64MT = 3000.0   # ms at 64mT
T2_WM_3T    = 70.0;   T2_GM_3T  = 83.0;    T2_CSF_3T  = 2000.0    # ms at 3T
T2_WM_64MT  = 80.0;   T2_GM_64MT = 100.0;  T2_CSF_64MT = 2000.0   # ms at 64mT

# SNR: scales roughly as B0^(7/4) empirically; Hyperfine compensates with
# averaging (ETL=80) and post-processing. Effective SNR loss factor ~8-12x.
# NOTE: The real Hyperfine scanner achieves ~309 SNR (on normalized images) via
# ETL=80 averaging (√80≈9x), 2 averages (√2≈1.41x), and compressed sensing
# reconstruction (~5x). Our simulation models conservative lower-bound physics.
SNR_LOSS_FACTOR      = 10.0       # empirical; 3T SNR / 64mT effective SNR
SNR_3T               = 40.0       # typical 3T brain SNR
SNR_64MT_EFFECTIVE   = SNR_3T / SNR_LOSS_FACTOR   # ≈ 4.0 (after Hyperfine averaging)

# B0 inhomogeneity — worse at low field due to poor shimming
B0_INHOMO_SIGMA      = 15.0       # spatial smoothness in voxels
B0_INHOMO_STRENGTH   = 0.10       # amplitude (fraction of signal)

# Legacy (for non-hyperfine modes)
RESOLUTION_FACTOR    = HIRES_RES_MM / T2W_INPLANE_MM   # 0.625
# ─────────────────────────────────────────────────────────────────────────────


class FieldConverter:
    """
    Converts high-field MRI images to simulate Hyperfine Swoop 64mT scanner.

    Three modes:
      'hyperfine'  — Physics-informed Hyperfine 64mT simulation (recommended)
      'combined'   — Legacy combined degradation
      'simulation' — Legacy simple simulation
      'degradation'— Legacy blur + noise
    """

    def __init__(self, config: Dict):
        self.config = config
        # Legacy params (used only in non-hyperfine modes)
        self.noise_level       = config.get('noise_level', 0.1)
        self.resolution_factor = config.get('resolution_factor', RESOLUTION_FACTOR)
        self.contrast_reduction= config.get('contrast_reduction', 0.3)
        self.method            = config.get('method', 'hyperfine')

    def convert(self, high_field_volume: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        method = method or self.method
        if method == 'hyperfine':
            return self._hyperfine_simulation(high_field_volume)
        elif method == 'simulation':
            return self._simulate_low_field(high_field_volume)
        elif method == 'degradation':
            return self._degrade_image(high_field_volume)
        elif method == 'combined':
            return self._combined_conversion(high_field_volume)
        else:
            raise ValueError(f"Unknown conversion method: {method}")

    # ── Physics-informed Hyperfine 64mT simulation ────────────────────────────

    def _hyperfine_simulation(self, volume: np.ndarray) -> np.ndarray:
        """
        Physics-informed simulation of Hyperfine Swoop 64mT T2w scan.

        Uses parameters measured directly from Zenodo BIDS JSON sidecars:
          TR=2000ms, TE=194.8ms, in-plane 1.6mm, slice 5.0mm

        Steps:
        1. T1w → T2w contrast transformation using tissue signal model at 64mT
        2. Anisotropic resolution: 1.6mm in-plane (axial), 5.0mm through-plane
        3. B0 inhomogeneity bias field
        4. Rician noise calibrated to effective 64mT SNR
        """
        # 1. T2-weighted contrast at 64mT ─────────────────────────────────────
        #    Input is T1w (WM=bright, GM=medium, CSF=dark).
        #    Classify tissue by intensity thresholds, apply T2w signal model.
        vol = volume.astype(np.float32)
        mask = vol > 0.05   # brain mask

        # Soft tissue classification: normalize within brain
        brain_vals = vol[mask]
        p33, p66   = np.percentile(brain_vals, 33), np.percentile(brain_vals, 66)

        # Compute T2w signal per tissue using spin-echo model:
        #   S = (1 - exp(-TR/T1)) * exp(-TE/T2)
        def se_signal(T1, T2):
            return (1 - np.exp(-T2W_TR_MS / T1)) * np.exp(-T2W_TE_MS / T2)

        s_wm  = se_signal(T1_WM_64MT,  T2_WM_64MT)   # ~0.92 * 0.089 = 0.082
        s_gm  = se_signal(T1_GM_64MT,  T2_GM_64MT)   # ~0.98 * 0.145 = 0.142
        s_csf = se_signal(T1_CSF_64MT, T2_CSF_64MT)  # ~0.49 * 0.906 = 0.445

        # Normalise to [0,1] range
        s_max = s_csf
        s_wm  /= s_max;  s_gm /= s_max;  s_csf /= s_max

        # Soft-blend each voxel toward its tissue signal
        # CSF = low T1w intensity, WM = high T1w intensity
        w_csf = np.clip((p33 - vol) / (p33 + 1e-6), 0, 1)
        w_wm  = np.clip((vol - p66) / (1.0 - p66 + 1e-6), 0, 1)
        w_gm  = np.clip(1 - w_csf - w_wm, 0, 1)

        t2w = w_csf * s_csf + w_gm * s_gm + w_wm * s_wm
        t2w = np.where(mask, t2w, 0.0).astype(np.float32)
        t2w = np.clip(t2w, 0, 1)

        # 2. Anisotropic resolution ────────────────────────────────────────────
        #    In-plane: 1.0mm → 1.6mm  (factor 0.625)
        #    Through-plane: 1.0mm → 5.0mm  (factor 0.2)
        inplane_factor  = HIRES_RES_MM / T2W_INPLANE_MM   # 0.625
        slicewise_factor= HIRES_RES_MM / T2W_SLICE_MM     # 0.2
        zoom_factors    = (inplane_factor, inplane_factor, slicewise_factor)

        low_res   = self._downsample_aniso(t2w, zoom_factors)
        upsampled = self._upsample(low_res, t2w.shape)

        # 3. B0 inhomogeneity bias field ──────────────────────────────────────
        bias = gaussian_filter(
            np.random.randn(*upsampled.shape).astype(np.float32),
            sigma=B0_INHOMO_SIGMA
        )
        bias = (bias - bias.mean()) / (bias.std() + 1e-8)
        bias = 1.0 + B0_INHOMO_STRENGTH * bias
        biased = np.clip(upsampled * bias, 0, 1)

        # 4. Rician noise at effective 64mT SNR ───────────────────────────────
        signal_mean = max(biased[biased > 0.05].mean() if biased.max() > 0.05 else 0.1, 0.01)
        sigma = signal_mean / SNR_64MT_EFFECTIVE
        sigma = np.clip(sigma, 0.005, 0.3)
        noisy = self._add_rician_noise(biased, sigma)

        return np.clip(noisy, 0, 1).astype(np.float32)

    # ── Legacy methods (kept for compatibility) ───────────────────────────────

    def _simulate_low_field(self, volume: np.ndarray) -> np.ndarray:
        low_res      = self._downsample(volume, self.resolution_factor)
        noisy        = self._add_noise(low_res, self.noise_level)
        low_contrast = self._reduce_contrast(noisy, self.contrast_reduction)
        upsampled    = self._upsample(low_contrast, volume.shape)
        return upsampled

    def _degrade_image(self, volume: np.ndarray) -> np.ndarray:
        if HAS_SCIPY:
            blurred = gaussian_filter(volume, sigma=1.5)
        else:
            from torch.nn import AvgPool3d
            tensor  = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
            blurred = AvgPool3d(kernel_size=3, stride=1, padding=1)(tensor).squeeze().numpy()
        return self._add_noise(blurred, self.noise_level)

    def _combined_conversion(self, volume: np.ndarray) -> np.ndarray:
        low_res      = self._downsample(volume, self.resolution_factor)
        noisy        = self._add_noise(low_res, self.noise_level)
        noisy        = self._add_rician_noise(noisy, self.noise_level * 0.5)
        low_contrast = self._reduce_contrast(noisy, self.contrast_reduction)
        if HAS_SCIPY:
            blurred  = gaussian_filter(low_contrast, sigma=0.5)
        else:
            from torch.nn import AvgPool3d
            tensor   = torch.from_numpy(low_contrast).unsqueeze(0).unsqueeze(0)
            blurred  = AvgPool3d(kernel_size=3, stride=1, padding=1)(tensor).squeeze().numpy()
        return self._upsample(blurred, volume.shape)

    # ── Spatial ops ───────────────────────────────────────────────────────────

    def _downsample_aniso(self, volume: np.ndarray, zoom_factors: tuple) -> np.ndarray:
        """Anisotropic downsample with per-axis zoom factors."""
        if not HAS_SCIPY:
            # Fallback: isotropic with mean factor
            factor = float(np.mean(zoom_factors))
            return self._downsample(volume, factor)
        return ndimage.zoom(volume, zoom=zoom_factors, order=1)

    def _downsample(self, volume: np.ndarray, factor: float) -> np.ndarray:
        if volume.size == 0 or factor <= 0:
            return volume.copy()
        if not HAS_SCIPY:
            new_shape = tuple(max(1, int(s * factor)) for s in volume.shape)
            if factor >= 1.0:
                return volume.copy()
            indices   = tuple(slice(None, None, max(1, int(1/factor))) for _ in volume.shape)
            downsampled = volume[indices]
            if downsampled.shape != new_shape:
                from torch.nn.functional import interpolate
                tensor      = torch.from_numpy(downsampled).unsqueeze(0).unsqueeze(0)
                resized     = interpolate(tensor, size=new_shape, mode='trilinear', align_corners=False)
                downsampled = resized.squeeze().numpy()
            return downsampled
        return ndimage.zoom(volume, zoom=factor, order=1)

    def _upsample(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        if volume.size == 0 or any(s == 0 for s in volume.shape):
            return np.zeros(target_shape, dtype=volume.dtype)
        if not HAS_SCIPY:
            from torch.nn.functional import interpolate
            tensor  = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
            resized = interpolate(tensor, size=target_shape, mode='trilinear', align_corners=False)
            return resized.squeeze().numpy()
        factors = [t / s if s > 0 else 1.0 for t, s in zip(target_shape, volume.shape)]
        return ndimage.zoom(volume, zoom=factors, order=1)

    # ── Noise ─────────────────────────────────────────────────────────────────

    def _add_noise(self, volume: np.ndarray, noise_level: float) -> np.ndarray:
        noise = np.random.normal(0, noise_level, volume.shape)
        return np.clip(volume + noise, 0, 1)

    def _add_rician_noise(self, volume: np.ndarray, noise_level: float) -> np.ndarray:
        """Rician noise — correct physical noise model for MRI magnitude images."""
        real_noise  = np.random.normal(0, noise_level, volume.shape).astype(np.float32)
        imag_noise  = np.random.normal(0, noise_level, volume.shape).astype(np.float32)
        noisy       = np.sqrt((volume + real_noise)**2 + imag_noise**2)
        return np.clip(noisy, 0, 1)

    def _reduce_contrast(self, volume: np.ndarray, reduction_factor: float) -> np.ndarray:
        mean_val = np.mean(volume)
        return np.clip(mean_val + (volume - mean_val) * (1 - reduction_factor), 0, 1)

    # ── Batch inference ───────────────────────────────────────────────────────

    def convert_batch(self, high_field_batch: torch.Tensor) -> torch.Tensor:
        low_field_batch = []
        for i in range(high_field_batch.shape[0]):
            volume    = high_field_batch[i].cpu().numpy()
            low_field = self.convert(volume)
            low_field_batch.append(torch.from_numpy(low_field).float())
        return torch.stack(low_field_batch)
