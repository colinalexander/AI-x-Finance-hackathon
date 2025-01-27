import os
import asyncio
import json
from typing import Tuple, List, Dict, Iterator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm import tqdm
import yfinance as yf

from gpt_research import custom_report
from data.alternative import NewsRetrievalAgent
from stock_analyst import CompanyAnalystAgent
from phi.agent import Agent
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger
from phi.model.openai import OpenAIChat
import base64


import base64
import os

with open(".env", "r") as env_file:
    openai_api_key = env_file.read()
    os.environ["OPENAI_API_KEY"] = openai_api_key

class FinancialImageAnalyzer:
    def __init__(self, model_id="gpt-4o", tools=None):
        if tools is None:
            tools = [DuckDuckGo()]
        self.agent = Agent(
            model=OpenAIChat(id=model_id),
            tools=tools,
            markdown=True,
        )

    def analyze_image(self, image_path, prompt=None):
        if prompt is None:
            prompt = (
                "As a financial analyst, analyze this graph and provide a detailed explanation of the underlying fundamentals."
            )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Get analysis from the agent
            response = self.agent.run(
                prompt,
                images=[image_path],
            )
            return response.content
        except Exception as e:
            raise RuntimeError(f"Error analyzing image: {e}")

    def analyze_images_in_directory(self, image_directory, prompt=None):
        if not os.path.isdir(image_directory):
            raise NotADirectoryError(f"Directory not found: {image_directory}")

        # One item for each ticker symbol
        results = {}
        ticker_symbols = os.listdir(image_directory)
        for ticker_symbol in tqdm(ticker_symbols):
            ticker_subdir = os.path.join(image_directory, ticker_symbol)
            image_files = os.listdir(ticker_subdir)
            
            # One item for each image related to a ticker symbol
            sub_results = {}
            for image_name in tqdm(image_files):
                image_path = os.path.join(ticker_subdir, image_name)
                try:
                    image_analysis = self.analyze_image(image_path, prompt)
                    sub_results[image_path] = image_analysis
                except Exception as e:
                    sub_results[image_path] = f"Error: {e}"

            results[ticker_symbol] = sub_results

        return results



class StockSectorDataPresentationAgent:
    def __init__(self, company, ticker, sector, ranking, save_directory):
        self.company = company
        self.ticker = ticker
        self.sector = sector
        self.top_ranking = ranking[0]
        self.bottom_ranking = ranking[1]
        self.stock_analyst = CompanyAnalystAgent(company, ticker)
        self.company_data = None
        self.sector_data = None
        self.directory = save_directory

    async def find_best_and_worst_stocks(self):
        research_query = f"Analyze the performance of all stocks in the {self.sector} sector over the past year."
        custom_query = f"Rank the top {self.top_ranking} best-performing stocks and the bottom {self.bottom_ranking} worst-performing stocks based on their price returns over the past year. Provide the results in a clear format with the stock ticker, company name, and percentage return."
        report = await custom_report(
            research_query=research_query,
            custom_query=custom_query,
            save_path=self.directory
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


class SectorLSStrategy:
    def __init__(self, report_path):
        self.report_path = report_path

    async def generate_LS_strategy_XLF(self):
        """
        First first the top N and bottom N performers and long top N and short
        the bottom N - present the results clearly with technical indicators and 
        quantitative factors
        """

        name_ticker_list = [
        ("NVIDIA", "NVDA"),
        ("Apple", "AAPL"),
        ("Microsoft", "MSFT"),
        ("Broadcom", "AVGO"),
        ("Salesforce", "CRM"),
        ("Oracle", "ORCL"),
        ("Cisco Systems", "CSCO"),
        ("ServiceNow", "NOW"),
        ("Accenture", "ACN"),
        ("IBM", "IBM"),
        ("Advanced Micro Devices", "AMD"),
        ("Qualcomm", "QCOM"),
        ("Adobe", "ADBE"),
        ("Texas Instruments", "TXN"),
        ("Intuit", "INTU"),
        ("Palantir", "PLTR"),
        ("Applied Materials", "AMAT"),
        ("Arista Networks", "ANET"),
        ("Palo Alto Networks", "PANW"),
        ("Micron Technology", "MU"),
        ("Analog Devices", "ADI"),
        ("Lam Research", "LRCX"),
        ("KLA Corporation", "KLAC"),
        ("Amphenol", "APH"),
        ("Intel", "INTC"),
        ("Cadence Design Systems", "CDNS"),
        ("CrowdStrike", "CRWD"),
        ("Synopsys", "SNPS"),
        ("Motorola Solutions", "MSI"),
        ("Autodesk", "ADSK"),
        ("Fortinet", "FTNT"),
        ("Roper Technologies", "ROP"),
        ("NXP Semiconductors", "NXPI"),
        ("Workday", "WDAY"),
        ("TE Connectivity", "TEL"),
        ("Fair Isaac Corporation (FICO)", "FICO"),
        ("Corning", "GLW"),
        ("Gartner", "IT"),
        ("Cognizant", "CTSH"),
        ("Dell Technologies", "DELL"),
        ("Monolithic Power Systems", "MPWR"),
        ("HP", "HPQ"),
        ("Microchip Technology", "MCHP"),
        ("Hewlett Packard Enterprise", "HPE"),
        ("ANSYS", "ANSS"),
        ("Keysight Technologies", "KEYS"),
        ("GoDaddy", "GDDY"),
        ("CDW Corporation", "CDW"),
        ("NetApp", "NTAP"),
        ("Tyler Technologies", "TYL"),
        ("Teledyne Technologies", "TDY"),
        ("ON Semiconductor", "ON"),
        ("Western Digital", "WDC"),
        ("Seagate Technology", "STX"),
        ("PTC", "PTC"),
        ("Zebra Technologies", "ZBRA"),
        ("Teradyne", "TER"),
        ("Jabil", "JBL"),
        ("Trimble", "TRMB"),
        ("First Solar", "FSLR"),
        ("VeriSign", "VRSN"),
        ("Super Micro Computer", "SMCI"),
        ("F5 Networks", "FFIV"),
        ("Genesis Energy", "GEN"),
        ("Skyworks Solutions", "SWKS"),
        ("Akamai Technologies", "AKAM"),
        ("EPAM Systems", "EPAM"),
        ("Juniper Networks", "JNPR"),
        ("Enphase Energy", "ENPH"),
        ("Xilinx", "XAKH25")
    ]

        for ticker in name_ticker_list:
            company_name = ticker[0]
            ticker_val = ticker[1]

            research_query = f"Analyze the performance of the stock in the f{company_name} over the past year."
            custom_query = f"Rank the performance of the stock and a clear analyst recommendation for the "
            
            report = await custom_report(
                research_query=research_query,
                custom_query=custom_query,
                save_path=self.report_path
            )

            print(f"Company {company_name} Analyst Recommendation:")
            print(report.content)
    
           
        


class GenerateLSFromFiles(Workflow):
    """
    A workflow that generates a financial analysis report using .txt files from a directory.
    """

    # Financial Analyst Agent
    financial_analyst: Agent = Agent(
        description="You are a professional L/S equity analyst working for a hedge fund.",
        instructions=[
            "You will be provided with relevant news articles and their contents about a specific company or topic.",
            "Analyze the data and decide the top 5 stocks to long and 5 stocks to short based on the file data. "
            "Decide if the stock merits a 'long' (buy), 'short' (sell), or 'neutral' position.",
            "Structure your analysis using the following professional report format tailored for L/S hedge funds:",
            "",
            "### Stock Analysis Report",
            "#### Recommendation",
            "- [Long / Short / Neutral]",
            "",
            "#### Investment Summary",
            "- Provide a concise, compelling investment thesis explaining why the stock is a long or short position.",
            "- Summarize key valuation metrics, growth drivers, risks, and catalysts.",
            "",
            "#### Key Details",
            "- Ticker / Company Name: [Ticker Symbol and Company Name]",
            "- Industry / Sector: [Industry and Sector of the company]",
            "- Current Price: [$X.XX]",
            "- Target Price: [$X.XX]",
            "- Market Cap: [$XX Billion]",
            "- 52-Week Range: [$Low – $High]",
            "- Dividend Yield: [X.XX%]",
            "- Analyst Coverage: [List major analysts covering the stock and their consensus ratings, if applicable.]",
            "",
            "#### Investment Thesis",
            "- Growth Drivers: [Opportunities, new markets, or innovations.]",
            "- Valuation: [Comparison to peers and historical averages.]",
            "- Competitive Position: [Market position, competitors, and moat.]",
            "- Macro/Industry Trends: [Relevant sector and economic trends.]",
            "- Catalysts: [Upcoming events or news impacting valuation.]",
            "",
            "#### Valuation Analysis",
            "- Valuation Metrics: [P/E, EV/EBITDA, Price/Sales, DCF valuation, etc.]",
            "- Comparison to Peers: [Stock performance relative to competitors.]",
            "- Upside/Downside Potential: [Expected price change and rationale.]",
            "",
            "#### Financial Overview",
            "- Revenue Growth (YoY): [XX%]",
            "- Earnings Growth (YoY): [XX%]",
            "- Debt/Equity Ratio: [X.XX]",
            "- Cash Flow Metrics: [Free cash flow trends, etc.]",
            "- Margins: [Gross, operating, and net margins.]",
            "",
            "#### Risks and Concerns",
            "- Operational Risks: [Execution risks, supply chain issues, etc.]",
            "- Valuation Risks: [Sensitivity to assumptions or overvaluation.]",
            "- Macroeconomic Risks: [Regulatory concerns, interest rates, etc.]",
            "- Competitive Risks: [Disruption, pricing pressures, new entrants.]",
            "",
            "#### Technical Analysis",
            "- Trend Analysis: [Uptrend, downtrend, or range-bound?]",
            "- Support/Resistance Levels: [$X.XX / $Y.YY]",
            "- Volume Trends: [Unusual volume activity?]",
            "- Momentum Indicators: [RSI, MACD, etc.]",
            "",
            "#### Recent Developments",
            "- Summarize notable news, earnings, acquisitions, or regulatory updates.",
            "",
            "#### Recommendation Justification",
            "- Provide a robust rationale for the recommendation based on all preceding analysis.",
            "",
            "Ensure that your report is concise, professional, and objective. All data must be accurately cited from the provided sources. Avoid fabricating any data."
        ],
    )

    def run(
        self,
        directory: str,
        topic: str,
        use_cached_report: bool = False
    ) -> Iterator[RunResponse]:
        """
        Generate a financial analysis report using .md files from a directory.

        Args:
            directory (str): Path to the directory containing performance report files.
            topic (str): The topic for which to generate the financial analysis.
            use_cached_report (bool, optional): Whether to return a previously generated report. Defaults to False.

        Returns:
            Iterator[RunResponse]: A stream of objects containing the generated report or status updates.
        """
        logger.info(f"Generating a financial analysis on: {topic}")

        if use_cached_report and "reports" in self.session_state:
            logger.info("Checking if cached financial analysis exists")
            for cached_report in self.session_state["reports"]:
                if cached_report["topic"] == topic:
                    yield RunResponse(
                        run_id=self.run_id,
                        event=RunEvent.workflow_completed,
                        content=cached_report["report"],
                    )
                    return

        logger.info(f"Reading .md files from directory: {directory}")
        scraped_articles = []
        for filename in os.listdir(directory):
            if filename.endswith(".md"):
                file_path = os.path.join(directory, filename)
                with open(file_path, "r") as file:
                    content = file.read()
                scraped_articles.append({
                    "title": filename,
                    "url": file_path,
                    "content": content
                })

        if not scraped_articles:
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"No .md files found in directory: {directory}",
            )
            return

        # Step 3: Generate the final financial analysis
        logger.info("Generating final financial analysis report")
        financial_analyst_input = {
            "topic": topic,
            "articles": scraped_articles,
        }

        # Stream responses from the financial_analyst agent
        yield from self.financial_analyst.run(json.dumps(financial_analyst_input, indent=4), stream=True)

        # Cache the final report
        if "reports" not in self.session_state:
            self.session_state["reports"] = []
        self.session_state["reports"].append(
            {"topic": topic, "report": self.financial_analyst.run_response.content}
        )

    def image_analysis_agent(self, image_directory):
        agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

        with open(image_directory, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        agent.print_response(
            "As a financial analyst, tell me about this graph and give me detailed explanation about the underlying fundamentals.",
            images=[encoded_image],
            stream=True,
        )
    

# if __name__ == "__main__":
#     directory = "./research_reports"  
#     topic = "Sector Performance Analysis"

#     generate_financial_analysis = GenerateLSFromFiles(
#         session_id=f"financial-analysis-on-{topic}",
#         storage=SqlWorkflowStorage(
#             table_name="financial_analysis_workflows",
#             db_file="tmp/workflows.db",
#         ),
#     )

#     report_stream: Iterator[RunResponse] = generate_financial_analysis.run(
#         directory=directory, 
#         topic=topic, 
#         use_cached_report=False
#     )

#     pprint_run_response(report_stream, markdown=True)



def image_analysis_agent(image_directory):
        agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[DuckDuckGo()],
        markdown=True,
    )


        with open(image_directory, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        agent.print_response(
            "As a financial analyst, tell me about this graph and give me detailed explanation about the underlying fundamentals.",
            images=[encoded_image],
            stream=True,
        )


if __name__ == "__main__":
    # image_analysis_agent('research_reports/charts/AAPL')
    f = FinancialImageAnalyzer()

    # results = f.analyze_image("research_reports/charts/AAPL/cumulative_returns_10y.jpg")
    results = f.analyze_images_in_directory("research_reports/charts")

    print(results)