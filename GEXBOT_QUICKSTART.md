# GEXBOT Integration - Quick Start

## What You Have Now

✅ **Working GEXBOT Integration** - Fetches gamma exposure data for: VIX, ES_SPX, NQ_NDX, SPY, QQQ, AAPL

✅ **All Key Metrics Captured:**
- Zero Gamma (inflection point)
- Sum GEX (net exposure)
- Major Call/Put Levels (OI and volume)
- Price (spot)
- Additional: strikes, delta risk reversal, priors

✅ **Three Ways to Collect Data:**
1. One-time backfill script
2. Continuous polling service  
3. Real-time dashboard

## Quick Start (3 Commands)

```bash
cd /home/rwest/pytorch-market
source .venv/bin/activate

# 1. Collect initial snapshot
python -m market_ml.gexbot_service &

# 2. View real-time dashboard
python -m market_ml.status_page &

# 3. Open browser to: http://localhost:5000
```

## Build Historical Data (2 Months)

### Option A: Fast Collection (Every 5 minutes)
```bash
# Will take ~60 days to collect 2 months of 5-minute data
# Run in screen/tmux or with nohup
nohup python -m market_ml.backfill_gex --samples 17280 --interval 300 > backfill.log 2>&1 &
```

### Option B: Moderate Collection (Every hour)
```bash
# Will take ~60 days to collect 2 months of hourly data  
# More manageable, still captures major moves
nohup python -m market_ml.backfill_gex --samples 1440 --interval 3600 > backfill.log 2>&1 &
```

### Option C: Use Continuous Service
```bash
# Service polls every 5 minutes automatically
# Just let it run and check back in a few days/weeks
python -m market_ml.gexbot_service
```

## Check Your Data

```bash
# View summary
cat outputs/gex_summary.csv

# Check per-symbol snapshots
ls -lh outputs/*.csv

# View backfill progress (if using backfill script)
ls -lh outputs/backfill/
tail outputs/backfill/SPY_gex_historical.csv
```

## Current Live Data

Your last fetch showed:

| Symbol | Price | Zero Gamma | Position | Net GEX (OI) |
|--------|-------|------------|----------|--------------|
| VIX | $24.59 | $22.38 | Above ↑ | +59.54 |
| ES_SPX | $6,668 | $6,724 | Below ↓ | -37,854 |
| SPY | $660.71 | $660.90 | Near → | -1,462 |
| QQQ | $600.03 | $601.59 | Below ↓ | -380 |
| AAPL | $247.48 | $246.69 | Above ↑ | +190 |

## Files & Documentation

| File | Purpose |
|------|---------|
| `docs/gexbot_integration.md` | Complete integration guide |
| `docs/gex_historical_collection.md` | Historical data collection guide |
| `market_ml/gexbot.py` | Data client |
| `market_ml/gexbot_service.py` | Polling service |
| `market_ml/backfill_gex.py` | Backfill script |
| `market_ml/status_page.py` | Web dashboard |
| `outputs/gex_summary.csv` | Latest levels (all symbols) |
| `outputs/{SYMBOL}_gex_snapshot.csv` | Per-symbol data |

## Configuration

### Change Symbols
Edit `market_ml/gexbot_service.py` line 13:
```python
SYMBOLS = ['VIX', 'ES_SPX', 'NQ_NDX', 'SPY', 'QQQ', 'AAPL']
```

### Change Polling Interval
Edit `market_ml/gexbot_service.py` line ~150:
```python
POLL_INTERVAL_SECONDS = 300  # 5 minutes (default)
```

### API Key
Already configured in `market_ml/.config.py` (gitignored)

## What's Working

✅ `classic/zero` endpoint - Current gamma levels  
✅ Unified field parsing (handles multiple schema formats)  
✅ Time-series building via periodic polling  
✅ Real-time dashboard  
✅ Summary outputs  

## What's Not Available (Tier Restricted)

❌ `classic/1d` - Historical daily data  
❌ `classic/max` - Max change windows  
❌ Websocket streaming  

**Workaround:** Build historical data by polling `classic/zero` over time (implemented)

## Common Tasks

### View Dashboard
```bash
python -m market_ml.status_page
# Open: http://localhost:5000
```

### Test API Connection
```bash
python -c "from market_ml.gexbot import load_historical_gex; from market_ml.config import Config; cfg=Config(); print(load_historical_gex('SPY','2025-10-01','2025-10-17',cfg.GEXBOT_API_KEY,'zero'))"
```

### Quick 1-Hour Collection (for testing)
```bash
# Collect 12 samples at 5-minute intervals
python -m market_ml.backfill_gex --samples 12 --interval 300
```

### Stop Services
```bash
pkill -f gexbot_service
pkill -f status_page
# Or: Ctrl+C in the terminal where they're running
```

## Next Steps

1. **Start collecting data** using one of the methods above
2. **Let it run** for several days/weeks to build history
3. **Integrate with strategy** - Use GEX metrics in `market_ml/strategy.py`:
   - Signal when price crosses zero gamma
   - Weight positions by net GEX
   - Avoid trades when near major levels
4. **Backtest** - Once you have historical data:
   - Test strategies against past GEX patterns
   - Validate edge before live trading
5. **Monitor** - Use dashboard to track current levels

## Support

- Full integration guide: `docs/gexbot_integration.md`
- Historical collection: `docs/gex_historical_collection.md`
- GEXBOT API docs: https://www.gexbot.com/apidocs

## Summary

You now have a complete system to:
1. ✅ Fetch current GEX levels for 6 symbols
2. ✅ Track zero gamma, net GEX, and major levels
3. ✅ Build historical time-series via polling
4. ✅ Monitor in real-time via web dashboard
5. ✅ Export data to CSV for analysis

**The system is ready to run!** Just start one of the collection methods and let it accumulate data over time.
