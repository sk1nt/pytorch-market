from datetime import datetime
import pytest
from market_ml.polygon import load_options_chain, PolygonError


def test_polygon_0dte_highest_oi_all():
    today = datetime.now().strftime('%Y-%m-%d')
    for symbol in ['QQQ', 'SPX', 'NDX', 'SPY']:
        try:
            df = load_options_chain(symbol, expiration_date=today)
        except PolygonError as e:
            msg = str(e).lower()
            if 'too many requests' in msg or '429' in msg or 'unauthorized' in msg or 'invalid' in msg:
                pytest.skip(f"Polygon API unavailable for {symbol} (rate limited or unauthorized); skipping.")
            raise
        assert not df.empty, f'No {symbol} 0DTE options returned.'
        # Highest open interest
        df_oi = df.dropna(subset=["open_interest"])
        assert not df_oi.empty, f'No {symbol} 0DTE options with open interest.'
        highest_oi = df_oi.loc[df_oi['open_interest'].idxmax()]
        fields = [
            'symbol', 'expiry', 'strike_price', 'right', 'open_interest', 'volume'
        ]
        print(f'Highest OI {symbol} 0DTE option:')
        for field in fields:
            val = highest_oi.get(field, '<NA>')
            print(f'  {field}: {val}')
        # Highest volume
        df_vol = df.dropna(subset=["volume"])
        if not df_vol.empty:
            highest_vol = df_vol.loc[df_vol['volume'].idxmax()]
            print(f'Highest volume {symbol} 0DTE option:')
            for field in fields:
                val = highest_vol.get(field, '<NA>')
                print(f'  {field}: {val}')
        else:
            print(f'No {symbol} 0DTE options with volume.')

if __name__ == '__main__':
    test_polygon_0dte_highest_oi_all()
