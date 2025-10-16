"""Data access helpers for the trading strategy tutorial project.

The Reddit post referenced by the user focuses on building a
machine-learning driven strategy on top of a liquid ETF.  To keep the
repository self-contained we only depend on `yfinance`, which is widely
available and easy to install.  The ``download_price_history`` function
wraps the yfinance API and returns a tidy ``pandas.DataFrame`` ready to be
fed into the rest of the pipeline.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

try:  # pragma: no cover - yfinance is optional in some environments
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The yfinance package is required to download market data. "
        "Install it with `pip install yfinance`."
    ) from exc


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration for the historical data download."""

    tickers: Iterable[str]
    start: _dt.date
    end: _dt.date | None = None
    interval: str = "1d"

    @property
    def tickers_list(self) -> List[str]:
        """Return the tickers as a list regardless of input type."""

        return list(self.tickers)


def download_price_history(config: DownloadConfig) -> pd.DataFrame:
    """Download adjusted OHLCV bars for the requested universe.

    Parameters
    ----------
    config:
        :class:`DownloadConfig` instance describing the assets, time span,
        and resolution of the data download.

    Returns
    -------
    pandas.DataFrame
        ``DataFrame`` indexed by ``DatetimeIndex`` containing the adjusted
        OHLCV data for all tickers in the configuration.  Columns follow the
        standard yfinance multi-index layout (``(field, ticker)``).
    """

    if config.end is None:
        end = _dt.date.today()
    else:
        end = config.end

    data = yf.download(
        tickers=config.tickers_list,
        start=config.start,
        end=end,
        interval=config.interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError(
            "No market data was returned. Check the ticker symbols and the date range."
        )

    # Ensure DatetimeIndex and sort just to be safe.
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    return data


__all__ = ["DownloadConfig", "download_price_history"]
