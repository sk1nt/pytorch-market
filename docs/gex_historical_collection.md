# GEX Historical Data Collection Guide

## Overview
Since the GEXBOT `classic/1d` historical endpoint is not available on the current API tier, we collect historical data by polling the `classic/zero` endpoint at intervals.

## Methods

### Method 1: Backfill Script (Recommended for building history)
Use the dedicated backfill script to collect snapshots over time.

**Quick test (30 snapshots over 5 minutes):**
```bash
cd /home/rwest/pytorch-market
source .venv/bin/activate
python -m market_ml.backfill_gex --samples 30 --interval 10
```

**Build 2 months of data (collect every 5 minutes):**
```bash
# 2 months ≈ 60 days ≈ 17,280 5-minute intervals
# At 5-minute intervals, this will take ~60 days to complete
python -m market_ml.backfill_gex --samples 17280 --interval 300
```

**Practical approach (collect hourly for 2 months):**
```bash
# 2 months ≈ 60 days * 24 hours = 1,440 hours
# At hourly intervals, this will take ~60 days to complete
python -m market_ml.backfill_gex --samples 1440 --interval 3600
```

**Output:**
- Data saved to: `outputs/backfill/{SYMBOL}_gex_historical.csv`
- Auto-saves every 10 samples (checkpoint)
- Press Ctrl+C to stop and save current data

### Method 2: Continuous Service (Recommended for ongoing monitoring)
Run the service continuously to poll and build time-series automatically.

**Usage:**
```bash
python -m market_ml.gexbot_service
```

**Behavior:**
- Fetches initial snapshot for all symbols
- Polls every 5 minutes (300 seconds) by default
- Appends new snapshots to historical data
- Saves to: `outputs/{SYMBOL}_gex_snapshot.csv`
- Generates summary: `outputs/gex_summary.csv`
- Runs indefinitely until Ctrl+C

**Configure polling interval:**
Edit `market_ml/gexbot_service.py` line ~150:
```python
POLL_INTERVAL_SECONDS = 300  # Change to desired seconds
```

## Comparison

| Method | Use Case | Duration | Output |
|--------|----------|----------|--------|
| Backfill Script | One-time historical collection | Runs for specified samples | `outputs/backfill/*.csv` |
| Continuous Service | Ongoing monitoring + building history | Runs indefinitely | `outputs/*.csv` + summary |

## Recommended Workflow

### For New Setup (Building Historical Data)
1. **Start backfill in background** (nohup or screen):
   ```bash
   nohup python -m market_ml.backfill_gex --samples 1440 --interval 3600 > backfill.log 2>&1 &
   ```

2. **Monitor progress**:
   ```bash
   tail -f backfill.log
   ```

3. **Check collected data**:
   ```bash
   ls -lh outputs/backfill/
   head outputs/backfill/SPY_gex_historical.csv
   ```

### For Ongoing Monitoring
1. **Run service in background**:
   ```bash
   nohup python -m market_ml.gexbot_service > gexbot_service.log 2>&1 &
   ```

2. **Run status dashboard**:
   ```bash
   python -m market_ml.status_page
   # Visit: http://localhost:5000
   ```

3. **Monitor logs**:
   ```bash
   tail -f gexbot_service.log
   ```

## Data Collection Math

### Polling Frequency vs Storage

**Every 5 minutes (288 samples/day):**
- 1 day = 288 snapshots
- 1 week = 2,016 snapshots
- 1 month = ~8,640 snapshots
- 2 months = ~17,280 snapshots

**Every 1 hour (24 samples/day):**
- 1 day = 24 snapshots
- 1 week = 168 snapshots
- 1 month = ~720 snapshots
- 2 months = ~1,440 snapshots

**Every 15 minutes (96 samples/day):**
- 1 day = 96 snapshots
- 1 week = 672 snapshots
- 1 month = ~2,880 snapshots
- 2 months = ~5,760 snapshots

### Recommended Intervals by Use Case

| Use Case | Interval | Samples (2mo) | Duration to Collect |
|----------|----------|---------------|---------------------|
| Intraday trading | 5 min | 17,280 | 60 days |
| Swing trading | 15 min | 5,760 | 60 days |
| Position trading | 1 hour | 1,440 | 60 days |
| Long-term analysis | 4 hours | 360 | 60 days |

## Output Data Format

Each CSV contains timestamped snapshots with all GEX metrics:

```csv
timestamp,ticker,price,zero_gamma,sum_gex_oi,sum_gex_vol,major_call_oi,major_call_vol,major_put_oi,major_put_vol,...
2025-10-17 04:00:00,SPY,660.71,660.9,-1462.86493,-53115.461,663.0,661.0,660.0,660.0,...
2025-10-17 04:05:00,SPY,660.85,660.95,-1455.23,-52980.12,663.0,661.0,660.0,660.0,...
...
```

## Systemd Service (Optional - Linux)

Create a systemd service for automatic startup:

**Create `/etc/systemd/system/gexbot.service`:**
```ini
[Unit]
Description=GEXBOT Data Collection Service
After=network.target

[Service]
Type=simple
User=rwest
WorkingDirectory=/home/rwest/pytorch-market
Environment="PATH=/home/rwest/pytorch-market/.venv/bin"
ExecStart=/home/rwest/pytorch-market/.venv/bin/python -m market_ml.gexbot_service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable gexbot
sudo systemctl start gexbot
sudo systemctl status gexbot
```

## Troubleshooting

**High memory usage:**
- Reduce polling frequency
- Limit number of symbols
- Periodically archive old data

**Missed intervals:**
- System was asleep/offline during that time
- Gaps in data are normal for long-running collection
- Timestamps will show actual collection times

**API rate limits:**
- Keep interval >= 60 seconds to avoid rate limits
- Current tier appears to allow ~1 request/symbol/minute

## Analysis Examples

Once you have historical data, you can analyze it:

```python
import pandas as pd

# Load collected data
df = pd.read_csv('outputs/backfill/SPY_gex_historical.csv', 
                 index_col='timestamp', parse_dates=True)

# Plot price vs zero gamma over time
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price and zero gamma
ax1.plot(df.index, df['price'], label='Price')
ax1.plot(df.index, df['zero_gamma'], label='Zero Gamma', linestyle='--')
ax1.legend()
ax1.set_ylabel('Price')
ax1.set_title('SPY Price vs Zero Gamma')

# Net GEX
ax2.plot(df.index, df['sum_gex_oi'])
ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_ylabel('Sum GEX (OI)')
ax2.set_xlabel('Time')
ax2.set_title('Net Gamma Exposure')

plt.tight_layout()
plt.savefig('outputs/gex_analysis.png')
print("Analysis saved to outputs/gex_analysis.png")
```

## Next Steps

1. Start collecting data using one of the methods above
2. Let it run for several days to build up history
3. Use the collected data for:
   - Backtest trading strategies
   - Identify gamma level patterns
   - Correlate with price movements
   - Build predictive models
