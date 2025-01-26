import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Tuple
from gpt_research import custom_report
from data.alternative import NewsRetrievalAgent
from stock_analyst import CompanyAnalystAgent

class StockSectorDataPresentationAgent:
    def __init__(self, company, ticker, sector, ranking: Tuple):
        self.company = company
        self.ticker = ticker
        self.sector = sector
        self.top_ranking = ranking[0]
        self.bottom_ranking = ranking[1]
        self.stock_analyst = CompanyAnalystAgent(company, ticker)
        self.company_data = None
        self.sector_data = None

    async def find_best_and_worst_stocks(self):
        research_query = f"Analyze the performance of all stocks in the {self.sector} sector over the past year."
        custom_query = f"Rank the top {self.top_ranking} best-performing stocks and the bottom {self.bottom_ranking} worst-performing stocks based on their price returns over the past year. Provide the results in a clear format with the stock ticker, company name, and percentage return."
        report = await custom_report(
            research_query=research_query,
            custom_query=custom_query
        )
        print(f"Best {self.top_ranking} Stocks:")
        print(report.content)
        print(f"\nWorst {self.bottom_ranking} Stocks:")
        print(report.content)

    def fetch_data(self, period="1y"):
        self.company_data = yf.Ticker(self.ticker).history(period=period)
        self.sector_data = yf.Ticker(self.sector).history(period=period)
        if self.company_data.empty:
            raise ValueError(f"No historical data found for {self.ticker} over {period}")
        if self.sector_data.empty:
            raise ValueError(f"No historical data found for {self.sector} over {period}")

    def beta_correlation_chart(self):
        if self.company_data is None or self.sector_data is None:
            self.fetch_data()
        company_returns = self.company_data['Close'].pct_change().dropna()
        sector_returns = self.sector_data['Close'].pct_change().dropna()
        company_returns, sector_returns = company_returns.align(sector_returns, join='inner')
        if company_returns.empty or sector_returns.empty:
            raise ValueError("No overlapping dates.")
        covariance_matrix = np.cov(company_returns, sector_returns)
        covariance = covariance_matrix[0, 1]
        variance = np.var(sector_returns)
        beta = covariance / variance
        plt.figure(figsize=(10, 6))
        plt.scatter(company_returns, sector_returns, alpha=0.5)
        plt.title(f'{self.ticker} vs {self.sector} (Beta: {beta:.2f})')
        plt.xlabel(f'{self.ticker} Returns')
        plt.ylabel(f'{self.sector} Returns')
        m, b = np.polyfit(company_returns, sector_returns, 1)
        plt.plot(company_returns, m * company_returns + b, color='red')
        plt.grid()
        plt.show()
        plt.close()

    def plot_historical_price(self):
        if self.company_data is None:
            self.fetch_data()
        plt.figure(figsize=(10, 6))
        plt.plot(self.company_data.index, self.company_data['Close'])
        plt.title(f'Historical Closing Price of {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.show()
        plt.close()

    def plot_candlestick_chart(self):
        if self.company_data is None:
            self.fetch_data()
        mpf.plot(
            self.company_data,
            type='candle',
            style='yahoo',
            title=f'{self.ticker} Candlestick Chart',
            volume=True,
            mav=(20, 50),
            figsize=(10, 6)
        )
        plt.close()

    def plot_volume(self):
        if self.company_data is None:
            self.fetch_data()
        plt.figure(figsize=(10, 4))
        plt.bar(self.company_data.index, self.company_data['Volume'], color='blue', alpha=0.6)
        plt.title(f'{self.ticker} Daily Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.grid(True)
        plt.show()
        plt.close()

    def plot_moving_averages(self, short_window=20, long_window=50):
        if self.company_data is None:
            self.fetch_data()
        df = self.company_data.copy()
        df['SMA'] = df['Close'].rolling(window=short_window).mean()
        df['LMA'] = df['Close'].rolling(window=long_window).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Close'], alpha=0.5, label='Close')
        plt.plot(df.index, df['SMA'], label=f'{short_window}-Day SMA', color='green')
        plt.plot(df.index, df['LMA'], label=f'{long_window}-Day SMA', color='red')
        plt.title(f'{self.ticker} Price with MAs')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()

    def monthly_returns_comparison_chart(self):
        if self.company_data is None or self.sector_data is None:
            self.fetch_data()
        company_monthly = self.company_data['Close'].resample('M').last().pct_change().dropna()
        sector_monthly = self.sector_data['Close'].resample('M').last().pct_change().dropna()
        company_monthly, sector_monthly = company_monthly.align(sector_monthly, join='inner')
        df_monthly = pd.DataFrame({
            f'{self.ticker}': company_monthly,
            f'{self.sector}': sector_monthly
        })
        ax = df_monthly.plot(kind='bar', figsize=(10, 6))
        ax.set_title(f'Monthly Returns: {self.ticker} vs {self.sector}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Monthly Returns')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        plt.close()

    def daily_returns_distribution_chart(self):
        if self.company_data is None or self.sector_data is None:
            self.fetch_data()
        company_returns = self.company_data['Close'].pct_change().dropna()
        sector_returns = self.sector_data['Close'].pct_change().dropna()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(company_returns, bins=50, alpha=0.7, color='green')
        plt.title(f'{self.ticker} Daily Returns')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        plt.hist(sector_returns, bins=50, alpha=0.7, color='blue')
        plt.title(f'{self.sector} Daily Returns')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        plt.close()

    def rolling_volatility_chart(self, window=20):
        if self.company_data is None or self.sector_data is None:
            self.fetch_data()
        company_returns = self.company_data['Close'].pct_change()
        sector_returns = self.sector_data['Close'].pct_change()
        company_rolling_vol = company_returns.rolling(window).std() * np.sqrt(252)
        sector_rolling_vol = sector_returns.rolling(window).std() * np.sqrt(252)
        plt.figure(figsize=(10, 6))
        plt.plot(company_rolling_vol.index, company_rolling_vol, label=f'{self.ticker} {window}-day Vol')
        plt.plot(sector_rolling_vol.index, sector_rolling_vol, label=f'{self.sector} {window}-day Vol')
        plt.title(f'{window}-Day Rolling Volatility Comparison')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()

    def stock_perfomer_comp_charts(self):
        pass

async def main():
    company = "Apple Inc."
    ticker = "AAPL"
    sector = "XLK"
    ranking = (25, 25)
    agent = StockSectorDataPresentationAgent(company, ticker, sector, ranking)
    agent.fetch_data("1y")
    # agent.beta_correlation_chart()
    # agent.plot_historical_price()
    # agent.plot_candlestick_chart()
    # agent.plot_volume()
    # agent.plot_moving_averages()
    # agent.monthly_returns_comparison_chart()
    agent.daily_returns_distribution_chart()
    agent.rolling_volatility_chart(20)

asyncio.run(main())
