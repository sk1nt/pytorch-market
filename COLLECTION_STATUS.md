# ✅ GEX Data Collection - Running Status

## 🎯 What's Running Now

You have **3 active data collectors** running in the background:

### 1. Hourly 24/7 Collection (PID: 30986)
- **Frequency**: Every hour, around the clock
- **Aggregation**: `zero` (0-DTE options)
- **Target**: 1,440 samples = 60 days
- **Output**: `outputs/backfill/{SYMBOL}_gex_historical.csv`
- **Log**: `backfill.log`
- **Status**: ✅ Running
- **Will complete**: ~60 days from start

### 2. Market Hours - Zero Aggregation (PID: 35224)
- **Frequency**: Every minute during market hours (8:30 AM - 6:00 PM EST)
- **Aggregation**: `zero` (0-DTE options only)
- **Target**: Runs indefinitely during trading hours
- **Output**: `outputs/intraday/SPY_gex_zero_intraday.csv` (and other symbols)
- **Log**: `intraday_zero.log`
- **Status**: ✅ Running (waiting for 8:30 AM market open)
- **Next start**: Today at 8:30 AM EST

### 3. Market Hours - Full Aggregation (PID: 35628)
- **Frequency**: Every minute during market hours (8:30 AM - 6:00 PM EST)
- **Aggregation**: `full` (all DTE options - complete gamma picture)
- **Target**: Runs indefinitely during trading hours
- **Output**: `outputs/intraday/SPY_gex_full_intraday.csv` (and other symbols)
- **Log**: `intraday_full.log`
- **Status**: ✅ Running (waiting for 8:30 AM market open)
- **Next start**: Today at 8:30 AM EST

## 📊 What You're Collecting

### Symbols (all collectors)
- VIX, ES_SPX, NQ_NDX, SPY, QQQ, AAPL

### Metrics (all collectors)
- Price (spot)
- Zero Gamma (inflection point)
- Major Call/Put Levels (OI and volume)
- Sum GEX (net gamma exposure)
- Strikes (full array)
- Delta Risk Reversal
- Max Priors

### Aggregation Differences

| Type | DTE Range | Description | Use Case |
|------|-----------|-------------|----------|
| `zero` | 0 DTE | Same-day expiring options only | Intraday gamma flips, day trading |
| `one` | 0-1 DTE | Today + tomorrow expiry | Near-term positioning |
| `full` | All DTE | Complete options chain | Full market gamma landscape |

## 📁 Output Files

### Hourly 24/7 Data
```
outputs/backfill/
├── VIX_gex_historical.csv
├── ES_SPX_gex_historical.csv
├── NQ_NDX_gex_historical.csv
├── SPY_gex_historical.csv
├── QQQ_gex_historical.csv
└── AAPL_gex_historical.csv
```

### Market Hours Intraday Data
```
outputs/intraday/
├── VIX_gex_zero_intraday.csv    (0-DTE, minute-by-minute)
├── VIX_gex_full_intraday.csv    (All DTE, minute-by-minute)
├── ES_SPX_gex_zero_intraday.csv
├── ES_SPX_gex_full_intraday.csv
├── ... (all symbols × 2 aggregations)
```

## 🕐 Collection Schedule

### Current Time: Friday 5:27 AM EST

**Today (Friday Oct 17)**
- 5:27 AM → Collectors waiting for market open
- 8:30 AM → Market hours collectors start (570 samples today)
- 6:00 PM → Market hours collectors stop
- Hourly collector continues overnight

**Weekend (Oct 18-19)**
- Market hours collectors sleep (no trading)
- Hourly collector continues every hour

**Monday (Oct 20)**
- 8:30 AM → Market hours collectors resume automatically

## 📈 Expected Data Volume

### After 1 Trading Day
- Hourly: ~24 samples (1 per hour)
- Market hours zero: ~570 samples (1 per minute × 9.5 hours)
- Market hours full: ~570 samples

### After 1 Week (5 Trading Days)
- Hourly: ~168 samples
- Market hours zero: ~2,850 samples
- Market hours full: ~2,850 samples

### After 60 Days
- Hourly: ~1,440 samples (completes, then stops)
- Market hours zero: ~34,200 samples (continues running)
- Market hours full: ~34,200 samples (continues running)

## 🔍 Monitoring Commands

### Check Status Anytime
```bash
cd /home/rwest/pytorch-market
./check_status.sh
```

### View Live Logs
```bash
# Market hours zero collection
tail -f intraday_zero.log

# Market hours full collection
tail -f intraday_full.log

# Hourly collection
tail -f backfill.log
```

### Check Process Health
```bash
ps aux | grep backfill | grep -v grep
```

### View Collected Data
```bash
# Count samples
wc -l outputs/intraday/*.csv
wc -l outputs/backfill/*.csv

# View latest samples
tail outputs/intraday/SPY_gex_zero_intraday.csv
tail outputs/backfill/SPY_gex_historical.csv
```

### Check Disk Usage
```bash
du -sh outputs/
ls -lh outputs/intraday/
ls -lh outputs/backfill/
```

## 🛑 Stop/Start Commands

### Stop All Collectors
```bash
pkill -f backfill_market_hours  # Stop market hours
pkill -f backfill_gex          # Stop hourly
```

### Stop Specific Collector
```bash
kill 35224  # Stop market hours zero
kill 35628  # Stop market hours full
kill 30986  # Stop hourly
```

### Restart Market Hours Collection
```bash
cd /home/rwest/pytorch-market && source .venv/bin/activate

# Zero aggregation (0-DTE)
nohup python -m market_ml.backfill_market_hours --days 0 --aggregation zero > intraday_zero.log 2>&1 &

# Full aggregation (all DTE)
nohup python -m market_ml.backfill_market_hours --days 0 --aggregation full > intraday_full.log 2>&1 &
```

### Restart Hourly Collection
```bash
cd /home/rwest/pytorch-market && source .venv/bin/activate
nohup python -m market_ml.backfill_gex --samples 1440 --interval 3600 > backfill.log 2>&1 &
```

## 📖 Documentation

- **Market hours guide**: `docs/market_hours_collection.md`
- **Historical collection**: `docs/gex_historical_collection.md`
- **Integration guide**: `docs/gexbot_integration.md`
- **Quick start**: `GEXBOT_QUICKSTART.md`

## ✨ What Happens Next

1. **Today at 8:30 AM EST** - Market hours collectors will automatically start collecting minute-by-minute data
2. **Every hour** - Hourly collector takes a snapshot
3. **6:00 PM EST today** - Market hours collectors stop for the day
4. **Weekend** - Market hours collectors sleep, hourly continues
5. **Monday 8:30 AM** - Market hours collectors resume automatically

## 🎯 When You Have Data

After a few days of collection, you can:

```python
import pandas as pd

# Load intraday data
df_zero = pd.read_csv('outputs/intraday/SPY_gex_zero_intraday.csv', 
                      index_col='timestamp', parse_dates=True)

df_full = pd.read_csv('outputs/intraday/SPY_gex_full_intraday.csv',
                      index_col='timestamp', parse_dates=True)

# Compare 0-DTE vs Full gamma
print(f"Zero (0-DTE): {len(df_zero)} samples")
print(f"Full (All DTE): {len(df_full)} samples")

# Analyze gamma flips
df_zero['gamma_flip'] = (df_zero['price'] > df_zero['zero_gamma']).astype(int).diff()
flips = df_zero[df_zero['gamma_flip'] != 0]
print(f"Gamma flips detected: {len(flips)}")

# Use in trading strategy
# See: market_ml/strategy.py for integration examples
```

## 🚀 Summary

You're now collecting:
- ✅ **High-resolution intraday data** (every minute during trading hours)
- ✅ **Two gamma perspectives** (0-DTE and full options chain)
- ✅ **Long-term context** (hourly snapshots 24/7)
- ✅ **Six key symbols** (VIX, ES_SPX, NQ_NDX, SPY, QQQ, AAPL)

All collectors will run automatically and save checkpoints regularly. Just let them run and check back in a few days to analyze the data!
