from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import Ticker

from sectors import sectors


@dataclass
class SymbolInfo:
    info: dict
    name: str
    biz_summary: str
    sector: str


# Map yfinance sector names to our sector names
SECTOR_MAP = {
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Industrials": "Industrials",
    "Communication Services": "Communication Services",
    "Utilities": "Utilities",
    "Energy": "Energy",
}


def get_symbol_info(ticker: str) -> SymbolInfo:
    """Get basic information about a stock symbol.

    Args:
        ticker: The stock ticker symbol (e.g. 'AAPL')

    Returns:
        SymbolInfo containing basic company information

    Raises:
        ValueError: If the stock's sector is not recognized
    """
    symbol = yf.Ticker(ticker)
    info = symbol.info
    yf_sector = info["sector"]

    if yf_sector not in SECTOR_MAP:
        raise ValueError(
            f"Unknown sector '{yf_sector}'. Must be one of: {list(SECTOR_MAP.keys())}"
        )

    return SymbolInfo(
        info=info,
        name=info["shortName"],
        biz_summary=info["longBusinessSummary"],
        sector=SECTOR_MAP[yf_sector],
    )


def load_prices(ticker: Ticker, period: str = "10y") -> pd.DataFrame:
    """Load prices for the given ticker

    Args:
        ticker: The stock ticker for the requested company
        period: The requested period, e.g. "1y", "10y", etc.

    Returns:
        A dataframe of historica prices including the columns:
            - Open
            - High
            - Low
            - Close
            - Volume
            - Dividends
            - Stock Splits
            - Capital Gains

    Raises:
        Exception if unable to create ticker and return prices.
    """
    try:
        df = ticker.history(period=period)
        # Store ticker symbol in DataFrame attrs for later use in column naming
        df.attrs["ticker"] = getattr(ticker, "ticker", "UNKNOWN")
        return df
    except Exception as e:
        raise e


def compute_indicators(df: pd.DataFrame, atr_period=14) -> Optional[pd.DataFrame]:
    """Compute indicators (cumulative return and annualized vol) from adjused closing prices.

    args:
        df: A dataframe with the columns High, Low and Close.

    returns:
        If there are fewer than 20 rows then return None.
        Otherwise returns a dataframe with these keys:
            cum_return: Cumulative return of the security.
            volatility: Annualized volatility of the security.

    """
    if any(col not in df for col in ("High", "Low", "Close")):
        raise ValueError("Dataframe must contain High, Low and Close columns.")
    if len(df) < 20:
        return None

    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    # Ensure index is DatetimeIndex for proper joining
    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)

    # 1) Compute daily returns (percent change of the 'Close' column)
    result.loc[:, "return"] = result["Close"].pct_change()

    # 2) Compute cumulative returns
    result.loc[:, "cum_return"] = (1 + result["return"]).cumprod().sub(1)
    result = result.iloc[1:, :]  # Drop first row that is #NA

    # Ensure timezone-naive index for consistent joining
    result.index = result.index.tz_localize(None)

    # 3) Compute the True Range (TR)
    # True Range = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
    high_low = result["High"] - result["Low"]
    high_prevclose = (result["High"] - result["Close"].shift()).abs()
    low_prevclose = (result["Low"] - result["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_prevclose, low_prevclose], axis=1).max(
        axis=1
    )

    # 4) Compute the Annualized Average True Range (ATR) as a rolling mean of the True Range
    result.loc[:, "ATR"] = true_range.rolling(atr_period).mean() * np.sqrt(252)

    # 5) Compute ATR as a percentage of the current Close
    result.loc[:, "atr_pct"] = result["ATR"] / result["Close"]

    # Return only the three requested columns
    return result[["return", "cum_return", "atr_pct"]]


def load_sector_stats() -> dict[str, pd.DataFrame | None]:
    """Loads cumulative returns and annualized ATR% for each sector.

    Returns:
        A dictionary keyed on sector name containing a DataFrame of cumulative returns and ATR%.
    """
    sector_stats: dict[str, pd.DataFrame | None] = {}
    for name, sector_info in sectors.items():
        symbol = sector_info["Ticker Symbol"]
        ticker = yf.Ticker(symbol)
        prices = load_prices(ticker)
        sector_stats[name] = compute_indicators(prices)

    return sector_stats


def combine_stats(
    symbol_indicators: pd.DataFrame,
    sector_indicators: pd.DataFrame,
    index_indicators: pd.DataFrame,
) -> pd.DataFrame:
    """Combines symbol, sector, and index statistics, rebasing cumulative returns to start at the same date.

    Args:
        symbol_indicators: DataFrame containing symbol indicators from compute_indicators()
        sector_indicators: DataFrame containing sector indicators from compute_indicators()
        index_indicators: DataFrame containing index indicators from compute_indicators()

    Returns:
        DataFrame containing rebased cumulative returns and ATR% for symbol, sector, and index,
        aligned by date using an inner join
    """
    # Find earliest common date across all three series
    common_start_date = max(
        symbol_indicators.index[0],
        sector_indicators.index[0],
        index_indicators.index[0],
    )

    def rebase_series(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Helper to rebase a DataFrame's cum_return and include its ATR."""
        rebased = (
            (1 + df["cum_return"])
            .div(1 + df.loc[common_start_date, "cum_return"])
            .sub(1)
        )
        return pd.DataFrame(
            {f"return_{prefix}": rebased, f"vol_{prefix}": df["atr_pct"]}
        )

    # Get ticker symbols
    symbol_ticker = symbol_indicators.attrs.get("ticker", "UNKNOWN")
    sector_ticker = sector_indicators.attrs.get("ticker", "UNKNOWN")
    index_ticker = index_indicators.attrs.get("ticker", "UNKNOWN")

    # Rebase all series with explicit naming
    symbol_df = rebase_series(symbol_indicators, symbol_ticker)
    sector_df = rebase_series(sector_indicators, sector_ticker)
    index_df = rebase_series(index_indicators, index_ticker)

    # Join all dataframes on index (date)
    return symbol_df.join([sector_df, index_df], how="inner")


if __name__ == "__main__":
    print(load_sector_stats())
