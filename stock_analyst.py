import os

import dotenv
import pandas as pd
import requests  # type: ignore
import yfinance as yf  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from loguru import logger
from openai import OpenAI

from utils import (
    combine_stats,
    compute_indicators,
    get_symbol_info,
    load_prices,
    load_sector_stats,
)

# OpenAI API setup (ensure you have your API key configured)
dotenv.load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Cache sector stats at module level
SECTOR_STATS = load_sector_stats()


def fetch_stock_sector_index_stats(
    ticker: str, index_symbol: str = "SPY"
) -> pd.DataFrame | None:
    """Fetch and analyze statistics for a stock, its sector, and market index.

    Args:
        ticker: Stock symbol (e.g. 'AAPL')
        index_symbol: Market index to compare against (default: 'SPY')

    Returns:
        DataFrame containing combined statistics or None if data retrieval fails
    """
    # Get basic info
    symbol_info = get_symbol_info(ticker)
    logger.info(f"Analyzing {symbol_info.name} ({ticker})")
    logger.info(f"Sector: {symbol_info.sector}")

    # Get symbol prices and indicators
    symbol = yf.Ticker(ticker)
    prices = load_prices(symbol)
    symbol_indicators = compute_indicators(prices)
    if symbol_indicators is None:
        logger.error(f"Not enough price data for {ticker}")
        return None

    # Get index data
    logger.info(f"Loading {index_symbol} data...")
    index = yf.Ticker(index_symbol)
    index_prices = load_prices(index)
    index_indicators = compute_indicators(index_prices)
    if index_indicators is None:
        logger.error(f"Not enough price data for {index_symbol}")
        return None

    # Get matching sector stats
    matching_sector_stats = SECTOR_STATS[symbol_info.sector]
    if matching_sector_stats is None:
        logger.error(f"No sector data available for {symbol_info.sector}")
        return None

    # Combine stats
    combined_stats = combine_stats(
        symbol_indicators, matching_sector_stats, index_indicators
    )

    # Log results
    # latest = combined_stats.iloc[-1]
    # start_date = combined_stats.index[0].strftime("%Y-%m-%d")
    # end_date = combined_stats.index[-1].strftime("%Y-%m-%d")

    # Show full dataframe
    logger.info("\nDetailed Statistics:")
    print(combined_stats)

    return combined_stats


def test_openai():
    """Quick test to ensure valid api key to OpenAI."""
    print("Trying completion")
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[{"role": "user", "content": "write a haiku about ai"}],
    )
    print(completion.choices[0].message)


class CompanyAnalystAgent:
    def __init__(self, company_name, ticker):
        self.company_name = company_name
        self.ticker = ticker
        self.report = ""

    def fetch_company_profile(self):
        try:
            company = yf.Ticker(self.ticker)
            profile = company.info
            self.report += "### Company Profile\n\n"
            self.report += f"Name: {profile.get('longName', 'N/A')}\n"
            self.report += f"Sector: {profile.get('sector', 'N/A')}\n"
            self.report += f"Industry: {profile.get('industry', 'N/A')}\n"
            self.report += (
                f"Description: {profile.get('longBusinessSummary', 'N/A')}\n\n"
            )
            print
        except Exception as e:
            self.report += f"Error fetching company profile: {str(e)}\n\n"

    def fetch_financials(self):
        try:
            company = yf.Ticker(self.ticker)
            financials = company.financials
            self.report += "### Financial Highlights\n\n"
            if financials is not None:
                self.report += financials.to_string() + "\n\n"
            else:
                self.report += "Financial data is not available.\n\n"
        except Exception as e:
            self.report += f"Error fetching financial data: {str(e)}\n\n"

    def fetch_latest_news(self):
        try:
            url = f"https://finance.yahoo.com/quote/{self.ticker}?p={self.ticker}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            # Find all links that might be news articles
            all_links = soup.find_all("a")
            news_links = []

            for link in all_links:
                # Check if the link has text and contains '/news/' in href
                if (
                    link.text.strip()
                    and link.get("href", "").strip()
                    and (
                        "/news/" in link.get("href", "")
                        or "finance.yahoo.com/news" in link.get("href", "")
                    )
                ):
                    news_links.append(link)

            self.report += "### Latest News\n\n"
            if not news_links:
                self.report += "No recent news articles found.\n"
            else:
                # Get unique news items by title to avoid duplicates
                seen_titles = set()
                for link in news_links[:10]:  # Check more links to find unique ones
                    title = link.text.strip()
                    if title and title not in seen_titles and len(seen_titles) < 5:
                        seen_titles.add(title)
                        href = link.get("href", "#")
                        full_link = (
                            f"https://finance.yahoo.com{href}"
                            if href.startswith("/")
                            else href
                        )
                        self.report += f"- [{title}]({full_link})\n"
            self.report += "\n"
        except Exception as e:
            self.report += f"Error fetching news: {str(e)}\n\n"

    def generate_analysis(self):
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional financial analyst. Analyze the following company information and provide a detailed investment analysis.",
                },
                {
                    "role": "user",
                    "content": f"Generate a detailed investment analysis for the following company:\n{self.report}\n",
                },
            ]
            response = client.chat.completions.create(
                model="gpt-4", messages=messages, max_tokens=1000
            )
            self.report += "### Investment Analysis\n\n"
            self.report += response.choices[0].message.content.strip() + "\n\n"
        except Exception as e:
            self.report += f"Error generating analysis: {str(e)}\n\n"

    def create_report(self):
        self.fetch_company_profile()
        self.fetch_financials()
        self.fetch_latest_news()
        self.generate_analysis()
        return self.report


def generate_charts(stats: pd.DataFrame, ticker: str, period: str = "10y") -> None:
    """Generate and save charts for cumulative returns and volatility.

    Args:
        stats: DataFrame containing combined statistics
        ticker: Stock symbol for chart titles and directory naming
        period: Time period for the chart (e.g. "10y", "5y", "1y", "6m")
    """
    import os
    import pandas as pd

    import matplotlib.pyplot as plt

    # Calculate start date based on period
    end_date = pd.Timestamp.now()
    if period.endswith('y'):
        years = int(period[:-1])
        start_date = end_date - pd.DateOffset(years=years)
    elif period.endswith('m'):
        months = int(period[:-1])
        start_date = end_date - pd.DateOffset(months=months)
    else:
        raise ValueError("Period must be in format 'Xy' or 'Xm' where X is a number")

    # Trim data to specified period
    stats = stats[stats.index >= start_date]

    # Extract sector and index tickers from column names
    index_ticker = stats.columns[-1].split("_")[1]  # Last column is for index
    sector_ticker = next(
        col.split("_")[1]
        for col in stats.columns
        if col.startswith("return_")
        and col != f"return_{ticker}"
        and col != f"return_{index_ticker}"
    )

    # Create directory if it doesn't exist
    chart_dir = f"my-docs/charts/{ticker}"
    os.makedirs(chart_dir, exist_ok=True)

    # Set style for better-looking charts
    plt.style.use("bmh")  # Using a built-in style

    # 1. Cumulative Returns Chart
    plt.figure(figsize=(12, 6))
    plt.plot(stats.index, stats[f"return_{ticker}"], label=f"{ticker} Returns")
    plt.plot(stats.index, stats[f"return_{sector_ticker}"], label="Sector Returns")
    plt.plot(stats.index, stats[f"return_{index_ticker}"], label="Market Returns")
    plt.title(
        f"Cumulative Returns ({period}): {ticker} vs Sector ({sector_ticker}) vs Market ({index_ticker})"
    )
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/cumulative_returns_{period}.jpg")
    plt.close()

    # 2. Volatility Chart
    plt.figure(figsize=(12, 6))
    plt.plot(stats.index, stats[f"vol_{ticker}"], label=f"{ticker} Volatility")
    plt.plot(stats.index, stats[f"vol_{sector_ticker}"], label="Sector Volatility")
    plt.plot(stats.index, stats[f"vol_{index_ticker}"], label="Market Volatility")
    plt.title(
        f"Volatility ({period}): {ticker} vs Sector ({sector_ticker}) vs Market ({index_ticker})"
    )
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{chart_dir}/volatility_{period}.jpg")
    plt.close()


def generate_all_charts(ticker: str, periods: list[str] = ["10y", "3m"]) -> None:
    """Generate charts for multiple time periods.

    Args:
        ticker: Stock symbol for chart generation
        periods: List of time periods to generate charts for (e.g. ["10y", "3m"])
    """
    stats = fetch_stock_sector_index_stats(ticker)
    if stats is not None:
        for period in periods:
            generate_charts(stats, ticker, period=period)
            logger.info(f"Generated charts for {ticker} ({period})")


if __name__ == "__main__":
    ticker = "AAPL"
    generate_all_charts(ticker)  # Will generate charts for 10y and 3m periods
