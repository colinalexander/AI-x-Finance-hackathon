from typing import Tuple
from gpt_research import custom_report, get_report
from data.alternative import NewsRetrievalAgent
from stock_analyst import CompanyAnalystAgent
import yfinance as yf
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class StockSectorDataPresentationAgent:
    """
    Be able to generate data visualizations for the analyst report that allows 
    the comparison of stock performances in the sector, and choose the top 25% and bottom 
    25% of stocks.
    """

    def __init__(self, company, ticker, sector, ranking: Tuple):
        self.company = company
        self.ticker = ticker
        self.sector = sector
        self.stock_analyst = CompanyAnalystAgent(company, ticker)
        self.top_ranking = ranking[0]
        self.bottom_ranking = ranking[1]

    async def find_best_and_worst_stocks(self):
        research_query = f"Analyze the performance of all stocks in the {self.sector} sector over the past year."
        custom_query = f"""
        Rank the top {self.top_ranking} best-performing stocks and the bottom {self.bottom_ranking} worst-performing stocks based on their price returns over the past year.
        Provide the results in a clear format with the stock ticker, company name, and percentage return.
        """

        # Generate the custom report
        report = await custom_report(
            research_query=research_query,
            custom_query=custom_query,
        )

        # Output the results
        print(f"Best {self.top_ranking} Stocks:")
        print(report.content)  # Adjust based on the output format
        print(f"\nWorst {self.bottom_ranking} Stocks:")
        print(report.content)  # Adjust based on the output format

    def beta_correlation_chart(self):
        """Generate a beta correlation chart between the stock and its sector index."""
        # Fetch stock and sector data
        company = yf.Ticker(self.ticker)
        sector_ticker_data = yf.Ticker(self.sector)

        # Fetch historical data for the stock and sector index
        company_data = company.history(period="1y")
        sector_data = sector_ticker_data.history(period="1y")

        if company_data.empty or sector_data.empty:
            raise ValueError("Historical data for the company or sector is unavailable.")

        # Calculate daily returns
        company_returns = company_data['Close'].pct_change().dropna()
        sector_returns = sector_data['Close'].pct_change().dropna()

        # Align data to handle missing dates
        aligned_data = company_returns.align(sector_returns, join='inner')
        if aligned_data[0].empty or aligned_data[1].empty:
            raise ValueError("No overlapping dates in the historical data.")

        company_returns = aligned_data[0]
        sector_returns = aligned_data[1]

        # Calculate beta
        covariance_matrix = np.cov(company_returns, sector_returns)
        covariance = covariance_matrix[0, 1]  
        variance = np.var(sector_returns)  
        beta = covariance / variance

        # Plot the correlation chart
        plt.figure(figsize=(10, 6))
        plt.scatter(company_returns, sector_returns, alpha=0.5, label='Daily Returns')
        plt.title(f'{self.ticker} vs {self.sector} (Beta: {beta:.2f})')
        plt.xlabel(f'{self.ticker} Returns')
        plt.ylabel(f'{self.sector} Returns')

        # Plot the regression line
        m, b = np.polyfit(company_returns, sector_returns, 1)
        plt.plot(company_returns, m * company_returns + b, color='red', label='Regression Line')
        plt.legend()
        plt.grid()
        plt.show()


    def stock_perfomer_comp_charts(self):
        ...


# Example usage

async def main():
    company = "Apple Inc."
    ticker = "AAPL"
    sector = "XLK"  # Technology sector ETF
    ranking = (25, 25)  # Top 25 and bottom 25 stocks

    agent = StockSectorDataPresentationAgent(company, ticker, sector, ranking)
    # await agent.find_best_and_worst_stocks()
    agent.beta_correlation_chart()


# Run the async function
asyncio.run(main())