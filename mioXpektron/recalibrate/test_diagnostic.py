#!/usr/bin/env python3
"""
Test script to demonstrate the diagnostic logging in flexible_calibrator_debug.py

This script shows how to enable debug logging and compare 'max' vs 'parabolic' methods.
"""

import logging
import sys

# Set up logging to see all diagnostic messages
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all diagnostic messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Example usage:
#
# from flexible_calibrator_debug import FlexibleCalibrator, FlexibleCalibConfig
#
# # Test with 'max' method
# config_max = FlexibleCalibConfig(
#     reference_masses=[22.9892207021, 38.9631579065, 58.065674, 86.096976, 104.107539, 184.073320],
#     calibration_method="quad_sqrt",
#     autodetect_method="max",  # Use 'max' for peak detection
#     output_folder="calibrated_max_debug"
# )
# calibrator_max = FlexibleCalibrator(config_max)
# summary_max = calibrator_max.calibrate(your_file_list)
#
# # Test with 'parabolic' method
# config_parabolic = FlexibleCalibConfig(
#     reference_masses=[22.9892207021, 38.9631579065, 58.065674, 86.096976, 104.107539, 184.073320],
#     calibration_method="quad_sqrt",
#     autodetect_method="parabolic",  # Use 'parabolic' for peak detection
#     output_folder="calibrated_parabolic_debug"
# )
# calibrator_parabolic = FlexibleCalibrator(config_parabolic)
# summary_parabolic = calibrator_parabolic.calibrate(your_file_list)
#
# # Compare the results
# print("\n" + "="*80)
# print("COMPARISON SUMMARY")
# print("="*80)
# print(f"MAX method - Mean PPM error: {summary_max['ppm_error'].mean():.2f}")
# print(f"PARABOLIC method - Mean PPM error: {summary_parabolic['ppm_error'].mean():.2f}")
# print("="*80)

print("""
To use the debug version:

1. Import the debug version instead of the regular version:
   from flexible_calibrator_debug import FlexibleCalibrator, FlexibleCalibConfig

2. Set logging level to see diagnostic output:
   import logging
   logging.basicConfig(level=logging.INFO)  # Use DEBUG for even more detail

3. The diagnostic output will show:
   - Whether parabolic fitting succeeded or fell back to max
   - When it succeeded, whether it chose the same or different channel as max
   - Summary statistics showing why you might be getting identical results

4. Look for messages like:
   [DIAGNOSTIC] Target 22.9892: PARABOLIC FALLBACK to MAX -> ch=12345
   [DIAGNOSTIC] Target 38.9631: PARABOLIC refined_mz=38.9633 -> ch=12346 (max_ch=12346, SAME=True)
   [DIAGNOSTIC SUMMARY] Same channel as max: 5/6 (83.3%)

This will tell you whether:
   - The parabolic fit is failing (and falling back to max)
   - The parabolic fit succeeds but chooses the same discrete channel
""")
