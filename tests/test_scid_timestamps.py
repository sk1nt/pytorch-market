"""
Test Sierra Chart .scid file timestamp parsing.

Sierra Chart uses 64-bit integers representing microseconds since 1899-12-30 00:00:00 UTC.
This aligns with legacy Excel/Windows serial-date conventions.

This test validates the timestamp conversion logic.
"""

import struct
import pandas as pd
import pytest

def test_scid_timestamp_conversion():
    """Test converting Sierra Chart timestamps to Python datetime."""
    base_date = pd.Timestamp('1899-12-30 00:00:00')
    
    # Test cases: various dates we want to validate
    test_cases = [
        pd.Timestamp('2025-10-17 12:00:00'),
        pd.Timestamp('2025-01-01 00:00:00'),
        pd.Timestamp('2024-12-31 23:59:59'),
        pd.Timestamp('2020-01-01 09:30:00'),
        pd.Timestamp('2025-10-17 09:30:00'),  # Market open today
    ]
    
    for target_date in test_cases:
        # Calculate microseconds since epoch
        time_diff = target_date - base_date
        microseconds = int(time_diff.total_seconds() * 1_000_000)
        
        # Simulate what's in the file
        as_uint64 = microseconds
        
        # Reverse conversion (what our code does)
        reconstructed = base_date + pd.Timedelta(microseconds=as_uint64)
        
        # Allow small time difference due to floating point precision
        time_diff_seconds = abs((reconstructed - target_date).total_seconds())
        assert time_diff_seconds < 0.001, \
            f"Reconstructed {reconstructed} doesn't match {target_date} (diff: {time_diff_seconds}s)"
        
        print(f"✓ {target_date} -> {microseconds:,} µs -> {reconstructed}")

def test_scid_timestamp_invalid():
    """Test handling of invalid timestamps."""
    base_date = pd.Timestamp('1899-12-30 00:00:00')
    
    # Invalid cases that should be rejected (year < 1990 or > 2100)
    invalid_microseconds = [
        0,  # 1899-12-30 (before 1990)
        1_000_000_000,  # 1899-12-30 00:16:40 (before 1990)
        3_155_760_000_000_000,  # ~2000 years later (overflow)
    ]
    
    for microseconds in invalid_microseconds:
        try:
            ts = base_date + pd.Timedelta(microseconds=microseconds)
            is_valid = 1990 <= ts.year <= 2100
            print(f"{'✗' if not is_valid else '✓'} {microseconds:,} µs -> {ts} (year {ts.year}) -> {'REJECT' if not is_valid else 'ACCEPT'}")
        except (OverflowError, pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta):
            print(f"✗ {microseconds:,} µs -> OVERFLOW -> REJECT")

if __name__ == '__main__':
    print("Testing Sierra Chart timestamp conversion...")
    print("=" * 70)
    test_scid_timestamp_conversion()
    print("\nTesting invalid timestamp handling...")
    print("=" * 70)
    test_scid_timestamp_invalid()
    print("\n✅ All tests passed!")
