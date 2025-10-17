# GEXBOT Integration Guide

## Overview
Successfully integrated GEXBOT gamma exposure data into the trading pipeline with comprehensive metric tracking and real-time dashboard.

## What We Built

### 1. GEXBOT Data Client (`market_ml/gexbot.py`)
**Supported Endpoints:**
- ✅ `classic/zero` - Current gamma levels (working with your API tier)
- ⚠️ `classic/1d` - Historical daily data (tier-restricted)
- ⚠️ `classic/max` - Max change windows (tier-restricted)
- 📝 `classic/majors` - Major levels endpoint (ready when available)

**Key Functions:**
- `load_historical_gex(symbol, start, end, api_key, aggregation)` - Fetch GEX data
- `load_max_change(symbol, api_key)` - Fetch max change arrays
- `extract_gamma_levels(df_or_dict)` - Extract key metrics from data
- `probe_gexbot_key(api_key)` - Validate API access

**Parsed Metrics:**
- `price` (spot price)
- `zero_gamma` (inflection point where gamma flips)
- `sum_gex_oi` / `sum_gex_vol` (net gamma exposure)
- `major_call_oi` / `major_call_vol` (major positive gamma level)
- `major_put_oi` / `major_put_vol` (major negative gamma level)
- Additional: `strikes`, `delta_risk_reversal`, `max_priors`

### 2. GEXBOT Service (`market_ml/gexbot_service.py`)
**Features:**
- Validates API key on startup
- Fetches current GEX levels for: VIX, ES_SPX, NQ_NDX, SPY, QQQ, AAPL
- Saves per-symbol snapshots: `outputs/{SYMBOL}_gex_snapshot.csv`
- Generates summary: `outputs/gex_summary.csv`
- Attempts real-time streaming (websocket currently unavailable)
- Periodic refresh every 10 seconds

**Usage:**
```bash
python -m market_ml.gexbot_service
```

### 3. Status Dashboard (`market_ml/status_page.py`)
**Features:**
- Real-time GEX metrics table showing:
  - Price vs Zero Gamma
  - Position classification (Above/Below/Near Zero)
  - Major call/put levels
  - Net GEX exposure
- Auto-refreshes every 5 seconds
- Trade ideas section (ready for strategy integration)

**Access:**
- URL: http://localhost:5000
- Auto-refresh: 5 seconds

**Usage:**
```bash
python -m market_ml.status_page
```

## Current Output Example

### GEX Summary (`outputs/gex_summary.csv`)
```csv
symbol,timestamp,price,zero_gamma,sum_gex_oi,sum_gex_vol,major_call_oi,major_call_vol,major_put_oi,major_put_vol
VIX,2025-10-16 20:00:00,24.59,22.38,59.5406,4.54297,25.0,25.0,16.92,21.0
ES_SPX,2025-10-16 20:00:00,6668.38,6724.65,-37854.72214,-180715.41217,6739.65,6739.65,6669.65,6669.65
SPY,2025-10-16 20:00:00,660.71,660.9,-1462.86493,-53115.461,663.0,661.0,660.0,660.0
```

## Key Insights from Current Data

**VIX:**
- Price: 24.59 vs Zero Gamma: 22.38 → **Above zero** (bullish pressure)
- Net GEX: +59.54 (positive gamma, stabilizing)
- Major Call: 25.0 | Major Put: 16.92

**ES_SPX:**
- Price: 6668.38 vs Zero Gamma: 6724.65 → **Below zero** (bearish pressure)
- Net GEX: -37,854 (negative gamma, destabilizing)
- Major Call: 6739.65 | Major Put: 6669.65

**SPY:**
- Price: 660.71 vs Zero Gamma: 660.9 → **Near zero** (neutral/indecision zone)
- Net GEX: -1,462 (slight negative gamma)
- Major Call: 663.0 | Major Put: 660.0

## Configuration

### API Key Setup
**Option 1: Environment Variable (Recommended)**
```bash
export GEXBOT_API_KEY="your_key_here"
```

**Option 2: Config File**
Edit `market_ml/.config.py`:
```python
GEXBOT_API_KEY = "your_key_here"
```

### Symbols
Edit `market_ml/gexbot_service.py` line 15:
```python
SYMBOLS = ['VIX', 'ES_SPX', 'NQ_NDX', 'SPY', 'QQQ', 'AAPL']
```

## Schema Support

### Classic Zero Schema
```json
{
  "timestamp": 1760644800,
  "ticker": "SPX",
  "spot": 6628.73,
  "zero_gamma": 6685.0,
  "major_pos_vol": 6700.0,
  "major_pos_oi": 6700,
  "major_neg_vol": 6630,
  "major_neg_oi": 6630,
  "sum_gex_vol": -180715.41,
  "sum_gex_oi": -37854.72,
  "strikes": [...],
  "max_priors": [...]
}
```

### Major Levels Schema (Also Supported)
```json
{
  "timestamp": 1760644800,
  "ticker": "SPX",
  "spot": 6628.73,
  "mpos_vol": 6700.0,
  "mpos_oi": 6700,
  "mneg_vol": 6630,
  "mneg_oi": 6630,
  "zero_gamma": 6685.0,
  "net_gex_vol": -180715.41,
  "net_gex_oi": -37854.72
}
```

Both schemas are automatically normalized to standardized field names.

## Next Steps

### Immediate Enhancements
1. **Periodic Polling**: Run service continuously to build time-series from zero snapshots
2. **Alerting**: Add notifications when price crosses major levels or zero gamma
3. **Strategy Integration**: Use GEX metrics in `market_ml/strategy.py` for trade signals

### API Tier Upgrades (if needed)
- **classic/1d**: Historical daily GEX time-series
- **classic/max**: Max change windows (1/5/10/15/30 min)
- **Websocket**: Real-time streaming (currently returning DNS error)

### Code is Ready For
- Historical backtesting when `classic/1d` becomes available
- Max change analysis when `classic/max` becomes available
- Real-time streaming when websocket endpoint is accessible

## Files Modified

- ✅ `market_ml/gexbot.py` - Data client with unified schema parsing
- ✅ `market_ml/gexbot_service.py` - Data ingestion service
- ✅ `market_ml/status_page.py` - Real-time dashboard
- ✅ `market_ml/config.py` - Enhanced config loading
- ✅ `market_ml/.config.py` - User secrets (gitignored)

## Testing

Quick test of all components:
```bash
# Test API key and fetch data
python -c "from market_ml.gexbot import load_historical_gex; from market_ml.config import Config; cfg=Config(); print(load_historical_gex('SPY','2025-10-01','2025-10-17',cfg.GEXBOT_API_KEY,'zero').head())"

# Run service (Ctrl+C to stop)
python -m market_ml.gexbot_service

# View status page
python -m market_ml.status_page
# Then visit: http://localhost:5000
```

## Troubleshooting

**"Invalid API Key" errors:**
- Check `market_ml/.config.py` or environment variable
- Verify key at https://gexbot.com account

**Websocket errors (Name or service not known):**
- Expected with current tier/endpoint availability
- Service continues with snapshot polling

**400 errors on classic/1d or classic/max:**
- These endpoints require higher API tier
- Use classic/zero for current levels (working)
