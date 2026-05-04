# Ch.3 Data Validation — Generator Scripts

This directory contains Python scripts to generate visualizations for Ch.3 (Data Validation & Drift Detection).

## Scripts

### 1. `gen_distribution_shift.py`
Generates the main distribution shift visualization showing California (training) vs Portland (production) income distributions.

**Output**: `../img/ch03-distribution-shift.png`

**Usage**:
```bash
python gen_distribution_shift.py
```

**What it shows**:
- Overlaid histograms of MedInc (California vs Portland)
- Mean lines showing 37% shift (3.8 → 5.2)
- Visual evidence of distribution drift

---

### 2. `gen_ks_test_results.py`
Generates bar chart showing KS test p-values for all features, highlighting which features have drifted.

**Output**: `../img/ch03-ks-test-results.png`

**Usage**:
```bash
python gen_ks_test_results.py
```

**What it shows**:
- p-values for 8 features (MedInc, HouseAge, etc.)
- Red bars: p < 0.05 (drift detected)
- Green bars: p > 0.05 (no drift)
- Threshold line at p=0.05

---

## Running All Scripts

To generate all visualizations at once:

```bash
# From this directory
python gen_distribution_shift.py
python gen_ks_test_results.py
```

Or from the repository root:

```bash
cd notes/01-ml/00_data_fundamentals/ch03_data_validation/gen_scripts
python gen_distribution_shift.py && python gen_ks_test_results.py
```

---

## Dependencies

All scripts require:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`

These are standard ML dependencies, already included in the `ml-foundations` environment.

---

## Design Conventions

All generated plots follow the track's visual conventions:
- **Dark theme**: `facecolor='#1a1a2e'`
- **Color palette**: Blue (#3b82f6) for training, Orange (#f59e0b) for production, Red (#b91c1c) for alerts
- **Resolution**: 150 DPI
- **Format**: PNG with transparent background support

---

## Notes

- Scripts simulate Portland data by scaling California MedInc by 1.37 (37% increase)
- Real-world Portland data would come from production logs or census data
- KS test uses `scipy.stats.ks_2samp` with default parameters
- All scripts use `random_state=42` for reproducibility
