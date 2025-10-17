# Sierra Chart .scid Timestamp Fix

## Problem

The `.scid` file parser was showing incorrect dates because it was trying to interpret Sierra Chart timestamps as Unix epoch timestamps or OLE Automation dates.

## Root Cause

Sierra Chart uses **64-bit integers representing microseconds since 1899-12-30 00:00:00 UTC**:

- **Format**: 8-byte unsigned integer (uint64)
- **Epoch**: 1899-12-30 00:00:00 UTC
- **Value**: Microseconds since the epoch
- **Precision**: Microsecond-level timestamps (1/1,000,000 second)

## Solution

Updated both `load_sierra_chart_scid()` and `stream_sierra_chart_scid()` functions in `market_ml/data.py` to:

1. Read the 8-byte value as `uint64` (using `struct.unpack('<Q', ...)`)
2. Interpret it directly as microseconds since 1899-12-30 00:00:00 UTC
3. Add that number of microseconds to the base date
4. Validate the resulting timestamp is between 1990-2100

## Example

```python
import struct
import pandas as pd

# Raw value from .scid file (as uint64)
raw_uint64 = 3969864000000000  # Microseconds

# Convert to datetime
base_date = pd.Timestamp('1899-12-30 00:00:00')
timestamp = base_date + pd.Timedelta(microseconds=raw_uint64)
# Result: 2025-10-17 12:00:00
```

## Conversion Table

| Microseconds Since Epoch | Date/Time |
|--------------------------|-----------|
| 3,787,032,600,000,000 | 2020-01-01 09:30:00 |
| 3,944,851,200,000,000 | 2025-01-01 00:00:00 |
| 3,969,855,000,000,000 | 2025-10-17 09:30:00 (market open) |
| 3,969,864,000,000,000 | 2025-10-17 12:00:00 (noon) |
| 3,969,894,000,000,000 | 2025-10-17 20:20:00 (market close) |

## Testing

Run the test suite to verify:

```bash
python tests/test_scid_timestamps.py
```

## Usage

The fix is automatic - just use the existing functions:

```python
from market_ml.data import load_sierra_chart_scid

# Load historical data
df = load_sierra_chart_scid('path/to/file.scid')
print(df.head())

# Dates will now be correct!
```

## Files Changed

- `market_ml/data.py` - Updated `load_sierra_chart_scid()` function (lines ~115-135)
- `market_ml/data.py` - Updated `stream_sierra_chart_scid()` function (lines ~50-65)
- `tests/test_scid_timestamps.py` - New test file to validate conversion

## References

- [Sierra Chart File Format Documentation](https://www.sierrachart.com/index.php?page=doc/IntradayDataFileFormat.html)
- Sierra Chart uses microseconds since 1899-12-30 00:00:00 UTC
- This aligns with legacy Windows/Excel serial-date conventions but at microsecond precision

## Validation

✅ Timestamps now correctly show dates in the 1990-2025 range
✅ Microsecond precision preserved for high-frequency data
✅ Invalid timestamps (year < 1990 or > 2100) are filtered out
✅ Test suite passes all validation checks

## Real-World Example

Successfully loaded MNQ and NQ futures data:

```bash
python -m market_ml.load_scid_data "/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid"
```

Output:
```
✅ Loaded 1,649,350 records
Date range: 2025-03-13 14:13:25.477000 to 2025-10-17 07:46:11
Duration: 217 days 17:32:45.523000
Total volume: 35,943,679
Cumulative delta: -109,141
```
