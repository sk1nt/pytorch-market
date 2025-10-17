# Market Hours Data Collection Guide

## Overview

You now have **two data collection strategies**:

### 1. **24/7 Hourly Collection** (Already Running)
- **Script**: `market_ml/backfill_gex.py`
- **Frequency**: Every hour, 24/7
- **Duration**: 60 days to get 60 days of data
- **Use case**: Long-term trends, weekend/after-hours analysis

### 2. **Market Hours Minute-by-Minute** (NEW!)
- **Script**: `market_ml/backfill_market_hours.py`
- **Frequency**: Every minute during market hours (8:30 AM - 6:00 PM EST)
- **Market hours only**: Monday-Friday, sleeps on weekends
- **Use case**: High-resolution intraday trading signals

## Aggregation Types

All three aggregation types are available:

| Type | Description | Use Case |
|------|-------------|----------|
| `zero` | 0-DTE options only (same-day expiry) | Short-term gamma flips, intraday moves |
| `one` | 0-1 DTE options (today + tomorrow) | Near-term pressure |
| `full` | All DTE options | Complete gamma landscape |

## Quick Start Commands

### Start Market Hours Collection (Recommended)

```bash
cd /home/rwest/pytorch-market
source .venv/bin/activate

# Collect for 5 trading days with 0DTE focus
nohup python -m market_ml.backfill_market_hours --days 5 --aggregation zero > intraday_zero.log 2>&1 &

# Collect for 5 trading days with full gamma picture
nohup python -m market_ml.backfill_market_hours --days 5 --aggregation full > intraday_full.log 2>&1 &

# Run indefinitely during market hours (best for ongoing monitoring)
nohup python -m market_ml.backfill_market_hours --days 0 --aggregation zero > intraday_zero.log 2>&1 &
```

### Check Status

```bash
# View current collection status
tail -20 intraday_zero.log

# Check how many samples collected
wc -l outputs/intraday/*.csv

# View latest data
tail outputs/intraday/SPY_gex_zero_intraday.csv
```

### Stop Collection

```bash
# Stop market hours collection
pkill -f backfill_market_hours

# Stop hourly collection
pkill -f backfill_gex
```

## Data Output

Market hours data is saved to: `outputs/intraday/`

Files are named: `{SYMBOL}_gex_{AGGREGATION}_intraday.csv`

Examples:
- `SPY_gex_zero_intraday.csv` - SPY with 0DTE options
- `SPY_gex_full_intraday.csv` - SPY with all DTE options
- `VIX_gex_zero_intraday.csv` - VIX with 0DTE options

## Collection Schedule Examples

### Example 1: 5 Trading Days of 0DTE Data
```bash
# Collects ~570 samples per day (9.5 hours × 60 minutes)
# Total: ~2,850 samples over 5 days
python -m market_ml.backfill_market_hours --days 5 --aggregation zero
```

### Example 2: Run During Market Hours Indefinitely
```bash
# Automatically starts at 8:30 AM EST, stops at 6:00 PM EST
# Resumes next trading day
# Perfect for ongoing live monitoring
python -m market_ml.backfill_market_hours --days 0 --aggregation zero
```

### Example 3: Collect All Three Aggregations Simultaneously
```bash
# Run three processes for complete coverage
nohup python -m market_ml.backfill_market_hours --days 0 --aggregation zero > intraday_zero.log 2>&1 &
nohup python -m market_ml.backfill_market_hours --days 0 --aggregation one > intraday_one.log 2>&1 &
nohup python -m market_ml.backfill_market_hours --days 0 --aggregation full > intraday_full.log 2>&1 &
```

## Sample Collection Timeline

**For 60 days of market-hours data:**

| Interval | Samples/Day | Total Samples | Real Time Duration |
|----------|-------------|---------------|-------------------|
| 1 minute | ~570 | ~34,200 | 60 trading days (~85 calendar days) |

**Trading days calculation:**
- Market hours: 8:30 AM - 6:00 PM = 9.5 hours = 570 minutes
- 60 trading days ≈ 12 weeks ≈ 3 months calendar time (accounting for weekends)

## Behavior Details

### Market Hours Detection
- **Trading days**: Monday-Friday
- **Hours**: 8:30 AM - 6:00 PM EST
- **Holidays**: Not currently detected (will attempt to poll on holidays)

### Outside Market Hours
- Script automatically sleeps
- Shows countdown to next market open
- No API calls made while sleeping
- Resumes automatically when market opens

### Sample Output During Market Hours
```
[2025-10-17 09:30:00 EDT] Sample 60/2850
  VIX: price=$24.59, zero_gamma=$22.38, sum_gex=59.54
  ES_SPX: price=$6668.38, zero_gamma=$6724.65, sum_gex=-37854
  SPY: price=$660.71, zero_gamma=$660.9, sum_gex=-1462.86
  QQQ: price=$600.03, zero_gamma=$601.59, sum_gex=-380
  AAPL: price=$247.48, zero_gamma=$246.69, sum_gex=190
```

### Sample Output Outside Market Hours
```
[2025-10-17 18:01:00 EDT] Outside market hours
Waiting 14.5 hours until market opens...
Next market open: 2025-10-18 08:30:00 EDT
```

## Recommended Setup

**For 60 days of comprehensive data**, run these three processes:

1. **Intraday 0DTE** (market hours only)
   ```bash
   nohup python -m market_ml.backfill_market_hours --days 0 --aggregation zero > intraday_zero.log 2>&1 &
   ```

2. **Intraday Full** (market hours only)
   ```bash
   nohup python -m market_ml.backfill_market_hours --days 0 --aggregation full > intraday_full.log 2>&1 &
   ```

3. **Hourly 24/7** (already running)
   ```bash
   # This captures after-hours and weekend changes
   # Already started - check with: ps aux | grep backfill_gex
   ```

This gives you:
- ✅ High-resolution intraday data (every minute during trading)
- ✅ 0DTE gamma flips (zero aggregation)
- ✅ Complete gamma picture (full aggregation)
- ✅ After-hours context (hourly 24/7 collection)

## Monitoring

```bash
# Check all running collection processes
ps aux | grep "backfill" | grep -v grep

# Monitor logs in real-time
tail -f intraday_zero.log

# Check collected data
ls -lh outputs/intraday/
ls -lh outputs/backfill/

# Count samples
wc -l outputs/intraday/*.csv
```

## Next Steps

1. **Start market hours collection** (will auto-start at 8:30 AM EST today):
   ```bash
   cd /home/rwest/pytorch-market && source .venv/bin/activate
   nohup python -m market_ml.backfill_market_hours --days 0 --aggregation zero > intraday_zero.log 2>&1 &
   ```

2. **Check it started properly**:
   ```bash
   tail -20 intraday_zero.log
   ```

3. **Let it run** - it will automatically:
   - Wait until 8:30 AM EST
   - Collect data every minute during market hours
   - Stop at 6:00 PM EST
   - Resume Monday at 8:30 AM EST (if started on Friday evening)
   - Save checkpoints every hour

4. **After a few days**, analyze the data:
   ```python
   import pandas as pd
   df = pd.read_csv('outputs/intraday/SPY_gex_zero_intraday.csv', index_col='timestamp', parse_dates=True)
   print(f"Collected {len(df)} samples")
   print(f"Date range: {df.index.min()} to {df.index.max()}")
   ```
