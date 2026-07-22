"""
Data utilities for loading and preprocessing MRI data.

Provides two dataset classes:
- BiomarkerMRIDataset: pairs low-field MRI inputs with biomarker labels
  extracted from corresponding high-field MRI volumes.
- MRIDataset: legacy dataset for synthetic/single-directory data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
import random

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from scipy.ndimage import zoom as scipy_zoom

from utils.augmentation import get_augmentation
from utils.biomarker_extraction import extract_biomarkers_for_dataset


# ---------------------------------------------------------------------------
# New primary dataset: pairs low-field input with high-field biomarker labels
# ---------------------------------------------------------------------------

class BiomarkerMRIDataset(Dataset):
    """Dataset that pairs low-field MRI volumes with biomarker labels
    extracted from corresponding high-field MRI volumes."""

    def __init__(
        self,
        low_field_dir: str,
        high_field_dir: str,
        biomarker_labels: Dict[str, np.ndarray],
        split: str = "train",
        target_size: Tuple[int, int, int] = (64, 64, 64),
        augmentation=None,
        num_augmented_per_volume: int = 20,
    ):
        self.low_field_dir = Path(low_field_dir)
        self.high_field_dir = Path(high_field_dir)
        self.biomarker_labels = biomarker_labels
        self.split = split
        self.target_size = target_size
        self.augmentation = augmentation if split == "train" else None
        self.num_augmented = num_augmented_per_volume if split == "train" else 1

        # Pair high-field stems with low-field files
        self.pairs = self._pair_volumes()

        # Deterministic split
        self.samples = self._split(self.pairs)

    def _pair_volumes(self) -> List[Dict]:
        """Match each high-field volume with its low-field counterpart."""
        pairs = []
        for stem, labels in sorted(self.biomarker_labels.items()):
            # Low-field file is named {stem}_low_field.npy
            low_field_path = self.low_field_dir / f"{stem}_low_field.npy"
            if not low_field_path.exists():
                # Try without suffix in case naming varies
                candidates = list(self.low_field_dir.glob(f"*{stem}*"))
                if candidates:
                    low_field_path = candidates[0]
                else:
                    print(f"  Warning: no low-field file for {stem}, skipping")
                    continue

            pairs.append({
                "stem": stem,
                "low_field_path": str(low_field_path),
                "labels": labels,
            })
        return pairs

    def _split(self, pairs: List[Dict]) -> List[Dict]:
        """Deterministic split: first ~67% train, next ~17% val, last ~17% test."""
        n = len(pairs)
        if n <= 2:
            # Too few volumes — use all for requested split
            return pairs

        n_test = max(1, n // 6)
        n_val = max(1, n // 6)
        n_train = n - n_val - n_test

        if self.split == "train":
            return pairs[:n_train]
        elif self.split == "val":
            return pairs[n_train:n_train + n_val]
        else:
            return pairs[n_train + n_val:]

    def __len__(self):
        return len(self.samples) * self.num_augmented

    def __getitem__(self, idx):
        base_idx = idx // self.num_augmented
        sample = self.samples[base_idx]

        # Load low-field volume
        volume = np.load(sample["low_field_path"]).astype(np.float32)

        # Handle 4D
        if volume.ndim == 4:
            volume = volume[0]

        # Normalize to [0, 1]
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        # Resize to target
        if volume.shape != self.target_size:
            factors = [t / s for t, s in zip(self.target_size, volume.shape)]
            volume = scipy_zoom(volume, factors, order=1)

        # Apply augmentation (only for training)
        if self.augmentation is not None:
            volume = self.augmentation(volume)

        # Ensure contiguous float32
        volume = np.ascontiguousarray(volume, dtype=np.float32)

        # Add channel dimension: [1, D, H, W]
        volume = volume[np.newaxis, ...]

        labels = sample["labels"].astype(np.float32)

        return {
            "volume": torch.from_numpy(volume),
            "label": torch.from_numpy(labels),
            "sample_id": f"{sample['stem']}_aug{idx % self.num_augmented}",
        }


# ---------------------------------------------------------------------------
# Legacy dataset (kept for backward compatibility / synthetic testing)
# ---------------------------------------------------------------------------

class MRIDataset(Dataset):
    """Dataset class for 3D MRI data with biomarker labels (legacy)"""

    def __init__(
        self,
        data_dir: str,
        transform=None,
        split: str = "train",
        use_synthetic: bool = False,
        synthetic_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int, int]] = None,
        num_classes: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.use_synthetic = use_synthetic
        self.synthetic_size = synthetic_size
        self.image_size = image_size or (64, 64, 64)
        self.num_classes = num_classes
        self.samples = self._load_samples()

    def _load_samples(self):
        if self.use_synthetic:
            return self._create_synthetic_samples()
        metadata_file = self.data_dir / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                all_samples = json.load(f)
        else:
            all_samples = self._discover_files()
        if len(all_samples) == 0:
            return self._create_synthetic_samples()
        return self._split_samples(all_samples)

    def _discover_files(self) -> List[Dict]:
        samples = []
        extensions = ['.nii', '.nii.gz', '.npy', '.npz']
        for ext in extensions:
            for file_path in self.data_dir.glob(f'*{ext}'):
                samples.append({
                    'id': file_path.stem,
                    'path': str(file_path),
                    'biomarker': np.random.uniform(0.0, 1.0, self.num_classes).tolist(),
                })
        return samples

    def _split_samples(self, all_samples):
        random.seed(42)
        np.random.seed(42)
        np.random.shuffle(all_samples)
        n = len(all_samples)
        t1 = int(0.7 * n)
        t2 = int(0.85 * n)
        if self.split == "train":
            return all_samples[:t1] or all_samples
        elif self.split == "val":
            return all_samples[t1:t2] or all_samples[-1:]
        else:
            return all_samples[t2:] or all_samples[-1:]

    def _create_synthetic_samples(self):
        size = self.synthetic_size or (100 if self.split == "train" else 20)
        samples = []
        for i in range(size):
            samples.append({
                'id': f'synthetic_{self.split}_{i}',
                'path': None,
                'biomarker': np.random.uniform(0.0, 1.0, self.num_classes).tolist(),
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if sample['path'] is None or not Path(sample['path']).exists():
            volume = self._generate_synthetic_volume(sample['biomarker'])
        else:
            volume = self._load_volume(sample['path'])

        # Normalize
        volume = volume.astype(np.float32)
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        # Resize if needed
        if volume.shape != tuple(self.image_size):
            factors = [t / s for t, s in zip(self.image_size, volume.shape)]
            volume = scipy_zoom(volume, factors, order=1)

        if self.transform:
            volume = self.transform(volume)

        if len(volume.shape) == 3:
            volume = volume[np.newaxis, ...]

        label = np.array(sample['biomarker'], dtype=np.float32)

        return {
            'volume': torch.FloatTensor(volume),
            'label': torch.FloatTensor(label),
            'sample_id': sample['id'],
        }

    def _load_volume(self, path: str) -> np.ndarray:
        path = Path(path)
        if path.suffix == '.npy':
            volume = np.load(path)
        elif path.suffix == '.npz':
            data = np.load(path)
            volume = data['volume'] if 'volume' in data else data[list(data.keys())[0]]
        elif path.suffix in ['.nii', '.gz']:
            if not HAS_NIBABEL:
                raise ImportError("nibabel required for NIfTI files")
            img = nib.load(path)
            volume = img.get_fdata()
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        if volume.ndim == 4:
            volume = volume[0]
        return volume

    def _generate_synthetic_volume(self, biomarker) -> np.ndarray:
        volume = np.zeros(self.image_size, dtype=np.float32)
        center = np.array(self.image_size) // 2
        # Vectorized distance computation
        coords = np.mgrid[0:self.image_size[0], 0:self.image_size[1], 0:self.image_size[2]]
        dist = np.sqrt(
            (coords[0] - center[0])**2 +
            (coords[1] - center[1])**2 +
            (coords[2] - center[2])**2
        )
        volume = np.exp(-dist / (max(self.image_size) * 0.3))
        bm_val = biomarker[0] if isinstance(biomarker, (list, np.ndarray)) else biomarker
        volume += bm_val * 0.2 * np.sin(coords[0] * 0.1) * np.cos(coords[1] * 0.1)
        volume += np.random.normal(0, 0.05, volume.shape)
        return np.clip(volume, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_data_loaders(
    config: Dict,
    data_dir: str = None,
    use_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.

    Uses BiomarkerMRIDataset when high_field_dir/low_field_dir are present
    in the config and contain data. Falls back to legacy MRIDataset otherwise.
    """
    image_size = tuple(config.get('data', {}).get('image_size', [64, 64, 64]))
    batch_size = config.get('data', {}).get('batch_size', 4)
    num_workers = config.get('data', {}).get('num_workers', 0)
    num_classes = config.get('biomarkers', {}).get('num_classes', 4)

    high_field_dir = config.get('data', {}).get('high_field_dir', 'data/high_field')
    low_field_dir = config.get('data', {}).get('low_field_dir', 'data/low_field')
    augmentations_per_vol = config.get('data', {}).get('augmentations_per_volume', 20)

    # Decide which dataset to use
    use_biomarker_dataset = (
        not use_synthetic
        and Path(high_field_dir).exists()
        and Path(low_field_dir).exists()
        and len(list(Path(high_field_dir).glob("*.npy"))) > 0
    )

    if use_biomarker_dataset:
        # Extract biomarkers from high-field data
        biomarker_labels = extract_biomarkers_for_dataset(high_field_dir)

        if len(biomarker_labels) == 0:
            print("Warning: no biomarker labels extracted, falling back to synthetic")
            use_biomarker_dataset = False

    if use_biomarker_dataset:
        augmentation = get_augmentation(config, "train")

        train_dataset = BiomarkerMRIDataset(
            low_field_dir=low_field_dir,
            high_field_dir=high_field_dir,
            biomarker_labels=biomarker_labels,
            split="train",
            target_size=image_size,
            augmentation=augmentation,
            num_augmented_per_volume=augmentations_per_vol,
        )
        val_dataset = BiomarkerMRIDataset(
            low_field_dir=low_field_dir,
            high_field_dir=high_field_dir,
            biomarker_labels=biomarker_labels,
            split="val",
            target_size=image_size,
            augmentation=None,
            num_augmented_per_volume=1,
        )
        test_dataset = BiomarkerMRIDataset(
            low_field_dir=low_field_dir,
            high_field_dir=high_field_dir,
            biomarker_labels=biomarker_labels,
            split="test",
            target_size=image_size,
            augmentation=None,
            num_augmented_per_volume=1,
        )
    else:
        # Fallback to legacy synthetic dataset
        d = data_dir or config.get('data', {}).get('processed_data_dir', 'data/processed')
        train_dataset = MRIDataset(d, split="train", use_synthetic=True,
                                    image_size=image_size, num_classes=num_classes)
        val_dataset = MRIDataset(d, split="val", use_synthetic=True,
                                  image_size=image_size, num_classes=num_classes)
        test_dataset = MRIDataset(d, split="test", use_synthetic=True,
                                   image_size=image_size, num_classes=num_classes)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader
