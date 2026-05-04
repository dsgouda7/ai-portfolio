# Feature Candidacy Decision Flow Animation

## Overview

This script generates a **50-second slow-motion animation** that converts the static Feature Candidacy Decision Flow mermaid chart into an interactive learning experience. Four real California Housing features flow through the diagnostic decision tree, each taking a different path based on their M1/M2/M3/VIF/Δ_interact scores.

## What It Shows

### Features Demonstrated:

1. **MedInc** (Green path)
   - M1 = 0.47 (high), VIF = 1.5 (low), M3 = 0.33 (high)
   - Path: START → M1(Yes) → VIF(No) → M3(Yes) → **✅ STRONG** (independent predictor)
   - Duration: 9 seconds

2. **Latitude + Longitude** (Blue path)
   - M1 ≈ 0 (low), M2/M3 high, Δ_interact = +$10k (positive)
   - Path: START → M1(No) → M2M3(Yes) → JOINT(Yes) → **✅ IRREPLACEABLE** (complementary)
   - Duration: 9 seconds

3. **AveRooms** (Orange path)
   - M1 = 0.02 (moderate), VIF = 7.2 (high)
   - Path: START → M1(Yes) → VIF(Yes) → **⚡ COLLINEAR** (unstable weights)
   - Duration: 9 seconds

4. **Population** (Red path)
   - M1 ≈ 0, M2 ≈ 0, M3 ≈ 0 (all near-zero)
   - Path: START → M1(No) → M2M3(No) → **❌ DROP** (no signal)
   - Duration: 9 seconds

### Animation Sequence:

```
00:00 - 00:03  │ Empty tree (all nodes visible)
───────────────┼──────────────────────────────────────────
00:03 - 00:05  │ MedInc introduction + scores
00:05 - 00:12  │ MedInc path lights up step-by-step
───────────────┼──────────────────────────────────────────
00:12 - 00:14  │ Lat+Lon introduction + scores
00:14 - 00:21  │ Lat+Lon path lights up step-by-step
───────────────┼──────────────────────────────────────────
00:21 - 00:23  │ AveRooms introduction + scores
00:23 - 00:30  │ AveRooms path lights up step-by-step
───────────────┼──────────────────────────────────────────
00:30 - 00:32  │ Population introduction + scores
00:32 - 00:39  │ Population path lights up step-by-step
───────────────┼──────────────────────────────────────────
00:39 - 00:44  │ All paths overlaid (legend shown)
───────────────┼──────────────────────────────────────────
00:44 - 00:50  │ Loop back to start
```

## Visual Design

- **Dark theme**: `#1a1a2e` background matching chapter aesthetics
- **Node types**:
  - Start node: Purple (#8b5cf6)
  - Decision nodes: Blue (#3b82f6)
  - Outcome nodes: 
    - Green (#10b981) for positive verdicts
    - Orange (#f59e0b) for warnings
    - Red (#ef4444) for drop candidates
- **Active path highlighting**:
  - Edges glow golden (#fbbf24) when active
  - Nodes get thicker borders when reached
  - Final outcomes have bold text
- **Feature-specific colors**:
  - MedInc: #34d399 (green)
  - Lat+Lon: #60a5fa (blue)
  - AveRooms: #fb923c (orange)
  - Population: #f87171 (red)

## Technical Details

### Dependencies:
```python
numpy
matplotlib
matplotlib.patches (FancyBboxPatch, FancyArrowPatch)
matplotlib.animation (FuncAnimation)
```

### Output:
- **File**: `notes/01-ml/01_regression/ch03_feature_importance/img/ch03-feature-candidacy-flow.gif`
- **Dimensions**: 1400 × 1200 pixels
- **Frame rate**: 10 fps
- **Total frames**: ~500 frames
- **Duration**: 50 seconds
- **File size**: ~2-3 MB (optimized with Pillow)

### Rendering Time:
- **Typical**: 30-45 seconds on modern hardware
- **Frames rendered**: 500 (10 fps × 50 sec)
- Uses matplotlib's FuncAnimation with Pillow writer

## Usage

### To generate the animation:

```bash
cd c:\repos\ai-portfolio
python scripts\generate_feature_candidacy_flow.py
```

### Expected output:
```
============================================================
Generating Feature Candidacy Decision Flow Animation
============================================================

Generating ch03-feature-candidacy-flow.gif...
✓ Saved notes/01-ml/01_regression/ch03_feature_importance/img/ch03-feature-candidacy-flow.gif
  Duration: 50.0 seconds
  Features shown: MedInc, Lat+Lon, AveRooms, Population

============================================================
✓ Decision flow animation complete!
============================================================

Animation shows:
  1. Empty decision tree (3 sec)
  2. MedInc → Strong independent predictor
  3. Lat+Lon → Jointly irreplaceable
  4. AveRooms → Collinear signal (VIF > 5)
  5. Population → Drop candidate
  6. All paths overlaid (5 sec)

Total duration: ~50 seconds (slow-motion for clarity)
```

## Educational Value

### Why This Animation Matters:

1. **Concrete examples**: Students see real California Housing features, not abstract "Feature j"
2. **Step-by-step logic**: Each decision node pauses, showing the test being applied
3. **Diagnostic divergence**: MedInc passes all tests; Lat+Lon needs joint testing; AveRooms hits VIF block
4. **Color-coded outcomes**: Green = keep, Orange = monitor, Red = drop
5. **Memory aid**: Visual paths are easier to remember than static flowcharts

### Learning Outcomes:

After watching this animation, students should be able to:
- ✅ Explain why MedInc is the dominant predictor (high on all three methods)
- ✅ Understand why Lat+Lon need joint permutation testing (low M1, high M2/M3)
- ✅ Recognize VIF as an early-exit criterion (AveRooms stops at VIF node)
- ✅ Identify drop candidates (Population fails all tests)
- ✅ Apply the same diagnostic flow to new datasets

## Integration with README

The animation is embedded in `README.md` with:
- Descriptive caption explaining what to watch for
- Collapsible `<details>` section containing the static mermaid flowchart as reference
- Cross-references to the three-method convergence table

## Customization

### To add new features to the animation:

Edit the `FEATURE_PATHS` dictionary in the script:

```python
FEATURE_PATHS = {
    'YourFeature': {
        'path': ['START', 'M1', ...],  # Node names from tree
        'color': '#hexcolor',           # Feature color
        'label': 'YourFeature: scores', # Display label
        'scores': 'M1 ✓ | M2 ✗ | ...'  # Score summary
    }
}
```

### To adjust animation speed:

- **Faster**: Reduce `interval` in FuncAnimation (e.g., 50 = 20fps)
- **Slower**: Increase frame counts in the sequence logic
- **Pause longer**: Increase frame count at each decision node (currently 15 frames per step)

## Maintenance

### If the decision tree structure changes:

1. Update `build_decision_tree()` node positions
2. Update edge connections in the edges list
3. Update feature paths in `FEATURE_PATHS`
4. Re-run the script to regenerate

### If new diagnostic metrics are added:

1. Add new decision nodes to the tree
2. Update feature scores to include new metrics
3. Adjust text sizes if nodes become too crowded

## Related Files

- **README**: `notes/01-ml/01_regression/ch03_feature_importance/README.md`
- **MI visuals script**: `scripts/generate_mi_visuals.py`
- **Documentation**: `scripts/MI_VISUAL_ASSETS_README.md`
- **Summary**: `scripts/MI_ENHANCEMENT_SUMMARY.md`

---

**Generated with ❤️ for clear, visual machine learning education**
