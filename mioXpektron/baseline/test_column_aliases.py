#!/usr/bin/env python3
"""
Test script to verify that column name aliasing works correctly
for various naming conventions (Channel, channel, ch, idx, etc.)
"""

from pathlib import Path
import polars as pl
import tempfile

# Import the functions we need to test
from .baseline_base import COL_ALIASES, read_spectrum_table


def test_column_aliases():
    """Test that various column naming conventions are properly recognized."""

    print("Testing Column Aliasing System")
    print("=" * 70)
    print("\nDefined aliases:")
    for standard_name, aliases in COL_ALIASES.items():
        print(f"  {standard_name}: {sorted(aliases)}")

    # Create test data with different naming conventions
    test_cases = [
        {
            "name": "Standard naming",
            "columns": ["channel", "mz", "intensity"],
        },
        {
            "name": "Capitalized naming",
            "columns": ["Channel", "Mz", "Intensity"],
        },
        {
            "name": "Alternative short names",
            "columns": ["ch", "m/z", "counts"],
        },
        {
            "name": "Index-based naming",
            "columns": ["Index", "Mass", "Signal"],
        },
        {
            "name": "Idx variant",
            "columns": ["Idx", "M/Z", "Y"],
        },
        {
            "name": "Chan variant",
            "columns": ["chan", "moverz", "ion_counts"],
        },
    ]

    # Generate sample data
    sample_data = {
        "col1": [1, 2, 3, 4, 5],
        "col2": [10.5, 15.2, 20.8, 25.3, 30.1],
        "col3": [100.0, 200.0, 150.0, 300.0, 250.0],
    }

    print("\n" + "=" * 70)
    print("Test Results:")
    print("=" * 70)

    all_passed = True

    for test_case in test_cases:
        try:
            # Create temporary file with specific column names
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = Path(f.name)

                # Write CSV with test column names
                f.write('\t'.join(test_case["columns"]) + '\n')
                for i in range(5):
                    f.write(f"{sample_data['col1'][i]}\t{sample_data['col2'][i]}\t{sample_data['col3'][i]}\n")

            # Try to read with our function
            df = read_spectrum_table(temp_path)

            # Check if columns are standardized correctly
            expected_cols = ["channel", "mz", "intensity"]
            if set(df.columns) == set(expected_cols):
                print(f"✓ PASS: {test_case['name']}")
                print(f"    Input columns:  {test_case['columns']}")
                print(f"    Output columns: {df.columns}")
            else:
                print(f"✗ FAIL: {test_case['name']}")
                print(f"    Input columns:  {test_case['columns']}")
                print(f"    Output columns: {df.columns}")
                print(f"    Expected:       {expected_cols}")
                all_passed = False

            # Clean up
            temp_path.unlink()

        except Exception as e:
            print(f"✗ ERROR: {test_case['name']}")
            print(f"    {type(e).__name__}: {e}")
            all_passed = False
            if temp_path.exists():
                temp_path.unlink()

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 70)


if __name__ == "__main__":
    test_column_aliases()
