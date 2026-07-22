# Scripts Directory

## Available Scripts

### 1. `get_and_convert_mri.py` ⭐ **RECOMMENDED**
**All-in-one script to get high-field MRI and convert to low-field**

```bash
# Generate synthetic data and convert (easiest)
python3 scripts/get_and_convert_mri.py --source synthetic --num-volumes 10

# Try to download real data (requires nilearn)
python3 scripts/get_and_convert_mri.py --source real --num-volumes 5

# Auto mode: tries real first, falls back to synthetic
python3 scripts/get_and_convert_mri.py --source auto --num-volumes 5
```

**Options:**
- `--source`: `synthetic`, `real`, or `auto`
- `--num-volumes`: Number of volumes to generate/download
- `--size`: Volume size (e.g., `--size 256 256 256`)
- `--high-field-dir`: Output directory for high-field (default: `data/high_field`)
- `--low-field-dir`: Output directory for low-field (default: `data/low_field`)

---

### 2. `download_real_mri.py`
**Download real MRI data from public datasets**

```bash
# Download from nilearn datasets
python3 scripts/download_real_mri.py --source nilearn --num-samples 5

# Show download instructions
python3 scripts/download_real_mri.py --source instructions
```

**Note:** Requires `nilearn` package:
```bash
pip install --user nilearn
```

---

### 3. `convert_high_to_low.py`
**Convert existing high-field images to low-field**

```bash
# Convert a single file
python3 scripts/convert_high_to_low.py --input data/high_field/volume.npy --output data/low_field/volume_low.npy

# Convert all files in a directory
python3 scripts/convert_high_to_low.py --input data/high_field --output data/low_field
```

---

### 4. `generate_high_field.py`
**Generate synthetic high-field MRI volumes**

```bash
# Generate 10 volumes
python3 scripts/generate_high_field.py --num-volumes 10 --size 256 256 256 --complexity high
```

---

## Quick Start

**Easiest way to get started:**

```bash
# Generate synthetic data and convert (no dependencies needed)
python3 scripts/get_and_convert_mri.py --source synthetic --num-volumes 5 --size 128 128 128
```

This will:
1. Generate 5 realistic high-field MRI volumes
2. Convert them all to low-field
3. Save both in `data/high_field/` and `data/low_field/`

**For real data:**

1. Install nilearn: `pip install --user nilearn`
2. Run: `python3 scripts/get_and_convert_mri.py --source real --num-volumes 5`

---

## Data Sources

### Synthetic Data (No Installation Required)
- ✅ Fast generation
- ✅ Realistic brain-like structures
- ✅ Good for testing the pipeline
- ✅ No dependencies

### Real Data (Requires nilearn)
- ✅ Actual MRI scans
- ✅ Multiple datasets available:
  - Haxby 2001 (small, fast)
  - OASIS (larger dataset)
  - ABIDE (autism brain imaging)
- ⚠️ Requires: `pip install --user nilearn`

---

## Next Steps

After generating/converting data:

1. **Verify data:**
   ```bash
   ls -lh data/high_field/
   ls -lh data/low_field/
   ```

2. **Run Stage 1:**
   ```bash
   python3 main.py --stage stage1 --use-synthetic
   ```

3. **Or use your converted data:**
   ```bash
   python3 main.py --stage stage1 --data-dir data/processed
   ```




