import os

import dotenv
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
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


def main(ticker: str, index_symbol: str = "SPY") -> pd.DataFrame | None:
    """Main function to analyze a stock.

    Args:
        ticker: Stock symbol (e.g. 'AAPL')
        index_symbol: Market index to compare against (default: 'SPY')
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


if __name__ == "__main__":
    df = main("AAPL")
