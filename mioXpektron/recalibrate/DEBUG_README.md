# Diagnostic Calibration Debug Tools

## Files Created

1. **`flexible_calibrator_debug.py`** - Debug version with extensive logging
2. **`test_diagnostic.py`** - Example usage script
3. **`flexible_calibration_debug.ipynb`** - Interactive notebook (UPDATED!)

## Quick Start - Using the Notebook

1. Open `flexible_calibration_debug.ipynb` in Jupyter
2. Run all cells in order
3. Watch the diagnostic output carefully

## What to Look For

### 1. In the Console Output (during parabolic method run):

Look for the **[DIAGNOSTIC SUMMARY]** section at the end:

```
[DIAGNOSTIC SUMMARY] ==========================================
[DIAGNOSTIC SUMMARY] Total targets: 17
[DIAGNOSTIC SUMMARY] Parabolic fit succeeded: 12/17 (70.6%)
[DIAGNOSTIC SUMMARY] Parabolic fit failed (fallback to max): 5/17 (29.4%)
[DIAGNOSTIC SUMMARY] When parabolic succeeded:
[DIAGNOSTIC SUMMARY]   - Same channel as max: 12/12 (100.0%)  ← KEY METRIC!
[DIAGNOSTIC SUMMARY]   - Different channel from max: 0/12 (0.0%)
[DIAGNOSTIC SUMMARY] ==========================================
```

### 2. Key Scenarios

#### Scenario A: Parabolic Falls Back to Max
```
[DIAGNOSTIC] Target 22.9892: PARABOLIC FALLBACK to MAX -> ch=12345
```
**Meaning:** Parabolic fitting failed (edge case, bad data, etc.), so it used max method instead.

#### Scenario B: Parabolic Succeeds But Chooses Same Channel
```
[DIAGNOSTIC] Target 38.9631: PARABOLIC refined_mz=38.9633 -> ch=12346 (max_ch=12346, SAME=True)
```
**Meaning:** Parabolic refined the peak center from 38.9631 to 38.9633, BUT when mapping back to discrete channels, it still chose channel 12346 (same as max).

**This is the most likely reason for identical PPM errors!** The parabolic interpolation works, but your data's channel resolution is too coarse for the refinement to matter.

#### Scenario C: Parabolic Succeeds And Chooses Different Channel
```
[DIAGNOSTIC] Target 104.1075: PARABOLIC refined_mz=104.1081 -> ch=15789 (max_ch=15788, SAME=False)
```
**Meaning:** Parabolic successfully refined the peak and selected a different channel than max. This should lead to different PPM errors!

### 3. Understanding the Results

If you see **"Same channel as max: 100%"**, it means:

- ✓ Parabolic fitting is working correctly
- ✓ The algorithm is refining peak positions
- ✗ BUT the refinement is too small to change the discrete channel selection
- → This is why you get identical PPM errors!

### 4. Possible Solutions

If parabolic isn't helping:

1. **Use 'gaussian' or 'voigt' methods** - More sophisticated peak fitting
2. **Increase data resolution** - If possible, acquire data with finer channel spacing
3. **Accept that 'max' is sufficient** - For your data resolution, simple max might be adequate
4. **Use continuous m/z values** - If your data has pre-calibrated m/z values, use those instead of bootstrapping from channels

## Testing Other Methods

Edit cell 7 in the notebook and change:
```python
autodetect_method="gaussian",  # Try: "gaussian", "voigt", "centroid"
```

Then re-run and compare!

## Need More Detail?

Change logging level in cell 1:
```python
logging.basicConfig(level=logging.DEBUG)  # More detailed output
```

This will show every single peak detection decision.
