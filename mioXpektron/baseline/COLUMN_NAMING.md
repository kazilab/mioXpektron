# Column Name Handling in Baseline Evaluation

## Problem Solved

The baseline evaluation code previously had hardcoded column names (`mz`, `intensity`, `channel`), which caused issues when data files used different naming conventions.

## Solution

The code now automatically handles various column naming conventions through an **aliasing system** in [`baseline_base.py`](baseline_base.py).

### Supported Column Names

The system recognizes the following column name variations (case-insensitive):

#### Channel Column
Any of these will be recognized as the "channel" column:
- `channel`, `Channel`, `CHANNEL`
- `chan`, `Chan`, `CHAN`
- `ch`, `Ch`, `CH`
- `index`, `Index`, `INDEX`
- `idx`, `Idx`, `IDX`

#### m/z Column
Any of these will be recognized as the "mz" column:
- `m/z`, `M/Z`, `m/Z`, `M/z`
- `mz`, `Mz`, `MZ`
- `mass`, `Mass`, `MASS`
- `moverz`, `MoverZ`, `MOVERZ`
- `m_over_z`, `M_OVER_Z`

#### Intensity Column
Any of these will be recognized as the "intensity" column:
- `intensity`, `Intensity`, `INTENSITY`
- `counts`, `Counts`, `COUNTS`
- `signal`, `Signal`, `SIGNAL`
- `y`, `Y`
- `ion_counts`, `Ion_Counts`, `ION_COUNTS`

### How It Works

1. **Case-Insensitive Matching**: All column names are converted to lowercase before comparison
2. **Automatic Standardization**: Input columns are automatically renamed to standard names (`channel`, `mz`, `intensity`)
3. **Fallback Behavior**: If no channel column is found, an index is automatically generated

### Code Location

The aliasing system is defined in [`baseline_base.py`](baseline_base.py):

```python
COL_ALIASES = {
    "channel": {"channel", "chan", "ch", "index", "idx"},
    "mz": {"m/z", "mz", "mass", "moverz", "m_over_z"},
    "intensity": {"intensity", "counts", "signal", "y", "ion_counts"},
}
```

### Testing

A test script is provided to verify the column name handling: [`test_column_aliases.py`](test_column_aliases.py)

Run the test:
```bash
python test_column_aliases.py
```

### Adding New Aliases

To add support for new column name variations:

1. Open [`baseline_base.py`](baseline_base.py)
2. Find the `COL_ALIASES` dictionary (around line 198)
3. Add your new alias to the appropriate set (all lowercase)
4. Example: To add "tof" as an alias for channel:
   ```python
   "channel": {"channel", "chan", "ch", "index", "idx", "tof"},
   ```

### Usage Example

Your data files can now use any of these naming conventions:

```python
# Example 1: Traditional naming
# Channel, m/z, Intensity

# Example 2: Short names
# ch, mz, counts

# Example 3: Index-based
# Idx, Mass, Signal

# All will work automatically!
```

## Benefits

✓ No need to manually rename columns in your data files
✓ Works with data from different instruments automatically
✓ Case-insensitive matching (Channel = channel = CHANNEL)
✓ Multiple aliases supported for each column type
✓ Easy to extend with new aliases
