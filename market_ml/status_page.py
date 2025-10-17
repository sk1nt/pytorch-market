import time
import threading
import pandas as pd
import os
from pathlib import Path
from flask import Flask, render_template_string
from market_ml.config import config

# Enhanced template for GEX status page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GEX Status Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .highlight { background-color: #fff3cd; }
        .metric { font-weight: bold; }
        .above-zero { color: #28a745; }
        .below-zero { color: #dc3545; }
        .near-zero { color: #ffc107; }
    </style>
</head>
<body>
    <h2>GEXBOT Real-Time Status</h2>
    <p><em>Last updated: {{ timestamp }}</em></p>
    
    <h3>Current GEX Levels</h3>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Price</th>
            <th>Zero Gamma</th>
            <th>Position</th>
            <th>Major Call (OI)</th>
            <th>Major Put (OI)</th>
            <th>Sum GEX (OI)</th>
        </tr>
        {% for row in gex_data %}
        <tr>
            <td class="metric">{{ row['symbol'] }}</td>
            <td>{{ "%.2f"|format(row['price']) if row['price'] else 'N/A' }}</td>
            <td>{{ "%.2f"|format(row['zero_gamma']) if row['zero_gamma'] else 'N/A' }}</td>
            <td class="{{ row['position_class'] }}">{{ row['position'] }}</td>
            <td>{{ "%.0f"|format(row['major_call_oi']) if row['major_call_oi'] else 'N/A' }}</td>
            <td>{{ "%.0f"|format(row['major_put_oi']) if row['major_put_oi'] else 'N/A' }}</td>
            <td>{{ "%.2f"|format(row['sum_gex_oi']) if row['sum_gex_oi'] else 'N/A' }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h3>Trade Ideas</h3>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Signal</th>
            <th>Status</th>
        </tr>
        {% for trade in trades %}
        <tr>
            <td>{{ trade['timestamp'] }}</td>
            <td>{{ trade['symbol'] }}</td>
            <td>{{ trade['direction'] }}</td>
            <td>{{ trade['signal'] }}</td>
            <td>{{ trade['status'] }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

app = Flask(__name__)

TRADES_DIR = config.OUTPUT_DIR
gex_data_cache = []
trade_ideas = []

def load_gex_summary():
    """Load GEX summary from the service output."""
    summary_path = Path(TRADES_DIR) / 'gex_summary.csv'
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
            # Classify position relative to zero gamma
            rows = []
            for _, row in df.iterrows():
                price = row.get('price')
                zero_gamma = row.get('zero_gamma')
                if pd.notna(price) and pd.notna(zero_gamma):
                    diff = price - zero_gamma
                    if abs(diff) < 5:  # Within 5 points
                        position = 'Near Zero'
                        position_class = 'near-zero'
                    elif diff > 0:
                        position = f'Above (+{diff:.1f})'
                        position_class = 'above-zero'
                    else:
                        position = f'Below ({diff:.1f})'
                        position_class = 'below-zero'
                else:
                    position = 'N/A'
                    position_class = ''
                
                rows.append({
                    'symbol': row.get('symbol', 'N/A'),
                    'price': row.get('price'),
                    'zero_gamma': row.get('zero_gamma'),
                    'position': position,
                    'position_class': position_class,
                    'major_call_oi': row.get('major_call_oi'),
                    'major_put_oi': row.get('major_put_oi'),
                    'sum_gex_oi': row.get('sum_gex_oi'),
                })
            return rows
        except Exception as e:
            print(f"Error loading GEX summary: {e}")
    return []

def load_trade_ideas():
    """Load trade ideas from CSV files in outputs directory."""
    trades = []
    trades_dir = Path(TRADES_DIR)
    if trades_dir.exists():
        for csv_file in trades_dir.glob('strategy_trades*.csv'):
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    trades.append({
                        'timestamp': str(row.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))),
                        'symbol': row.get('symbol', ''),
                        'direction': row.get('direction', ''),
                        'signal': row.get('signal', ''),
                        'status': row.get('status', 'active'),
                    })
            except Exception:
                pass
    return trades if trades else [{'timestamp': 'N/A', 'symbol': 'N/A', 'direction': 'N/A', 'signal': 'No trades yet', 'status': 'N/A'}]

def update_data():
    global gex_data_cache, trade_ideas
    while True:
        try:
            gex_data_cache = load_gex_summary()
            trade_ideas = load_trade_ideas()
        except Exception as e:
            print(f"Error updating data: {e}")
        time.sleep(5)

@app.route('/')
def status():
    return render_template_string(
        HTML_TEMPLATE, 
        gex_data=gex_data_cache,
        trades=trade_ideas,
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

if __name__ == '__main__':
    threading.Thread(target=update_data, daemon=True).start()
    app.run(debug=True, port=5000)
