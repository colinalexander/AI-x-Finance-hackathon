import json
from textwrap import dedent
from typing import Optional, Dict, Iterator
from pydantic import BaseModel, Field

from phi.agent import Agent
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
from phi.utils.pprint import pprint_run_response
from phi.utils.log import logger


class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")


class SearchResults(BaseModel):
    articles: list[NewsArticle]


class ScrapedArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(..., description="Summary of the article if available.")
    content: Optional[str] = Field(
        ...,
        description="Content in markdown format if available. Return None if the content is not available or does not make sense.",
    )


class GenerateFinancialAnalysis(Workflow):
    """
    A workflow that focuses on generating a financial analysis report for a given company/topic.
    """

    # 1. Searching the web for articles
    web_searcher: Agent = Agent(
        tools=[DuckDuckGo()],
        instructions=[
            "Given a topic, search for 15 articles and return all the articles in order of relevancy.",
        ],
        response_model=SearchResults,
    )

    # 2. Scraping each article for content
    article_scraper: Agent = Agent(
        tools=[Newspaper4k()],
        instructions=[
            "Given a URL, scrape the article and return the title, url, and markdown-formatted content.",
            "If the content is not available or does not make sense, return None as the content.",
        ],
        response_model=ScrapedArticle,
    )

    # 3. Generating the final financial analysis
    # financial_analyst: Agent = Agent(
    #     description="You are a professional financial analyst providing insights on the given topic.",
    #     instructions=[
    #         "You will be provided with relevant news articles and their contents.",
    #         "For each article, you must assess how relevant it is to the company's financial performance on a scale of 1-5 (where 1 = not relevant, 5 = highly relevant).",
    #         "Then, you must analyze these stories collectively and, based on their overall direction (positive, negative, neutral), provide a final recommendation: "
    #         "'Strong Buy', 'Buy', 'Neutral', 'Sell', or 'Strong Sell'.",
    #         "Focus on business fundamentals, market impact, potential future performance, and any indicated risk factors.",
    #         "Structure your final output in a professional financial analyst report format with sections like:",
    #         "    - Executive Summary",
    #         "    - Background",
    #         "    - Market/Industry Perspective",
    #         "    - Detailed Analysis & Metrics",
    #         "    - Risks & Opportunities",
    #         "    - Overall Recommendation",
    #         "    - Sources",
    #         "Make sure to cite sources accurately and do not fabricate data. Keep the tone analytical and objective.",
    #     ],
    #     expected_output=dedent("""\
    #     <report_format>
    #     ## {Engaging Report Title Reflecting the Financial Analysis}

    #     ### Executive Summary
    #     {High-level overview of the situation or event, highlighting key data points and findings in concise form.}

    #     ### Background
    #     {Context regarding the company, industry, or topic. Include relevant historical or strategic info.}

    #     ### Market or Industry Perspective
    #     {Discuss the broader market implications, industry trends, or competitor considerations.}

    #     ### Detailed Analysis & Metrics
    #     {Dive deeper into the financials, metrics, and any relevant analytical perspective gleaned from articles.}

    #     #### Article Relevance Scores
    #     {For each article, provide a rating (1–5) indicating how closely it's tied to financial performance. 
    #      E.g.:
    #      - [Article Title](url): Relevance Score = X
    #      - [Article Title](url): Relevance Score = Y}

    #     ### Risks & Opportunities
    #     {Identify key risks that might affect the outcome and highlight any growth or upside opportunities.}

    #     ### Overall Recommendation
    #     {After analyzing all relevant articles and data, provide a final recommendation. 
    #      Choose from: 'Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell'. 
    #      Provide a rationale for this choice.}

    #     ### Sources
    #     - [Article Title](article_url)
    #     - [Article Title](article_url)
    #     ...
    #     </report_format>
    #     """),
    # )

    financial_analyst: Agent = Agent(
    description="You are a professional L/S equity analyst working for a hedge fund.",
    instructions=[
        "You will be provided with relevant news articles and their contents about a specific company or topic.",
        "Analyze the data and decide if the stock merits a 'long' (buy) or 'short' (sell) position, or if it should be neutral.",
        "Structure your analysis using the following professional report format tailored for L/S hedge funds:",
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
    expected_output=dedent("""\
        ### Stock Analysis Report
        
        #### Recommendation
        - [Long / Short / Neutral]
        
        #### Investment Summary
        {Concise thesis summarizing the overall investment case.}
        
        #### Key Details
        - Ticker / Company Name: {Ticker Symbol and Company Name}
        - Industry / Sector: {Industry and Sector}
        - Current Price: ${X.XX}
        - Target Price: ${X.XX}
        - Market Cap: ${XX Billion}
        - 52-Week Range: ${Low} – ${High}
        - Dividend Yield: {X.XX%}
        - Analyst Coverage: {List of analysts, if available.}
        
        #### Investment Thesis
        - Growth Drivers: {Opportunities and growth catalysts.}
        - Valuation: {Comparison to peers and metrics.}
        - Competitive Position: {Company's market strength.}
        - Macro/Industry Trends: {Sector and economic trends.}
        - Catalysts: {Upcoming events affecting valuation.}
        
        #### Valuation Analysis
        - Valuation Metrics: {Key metrics like P/E, EV/EBITDA, etc.}
        - Comparison to Peers: {Performance relative to competitors.}
        - Upside/Downside Potential: {Projected price change and justification.}
        
        #### Financial Overview
        - Revenue Growth (YoY): {XX%}
        - Earnings Growth (YoY): {XX%}
        - Debt/Equity Ratio: {X.XX}
        - Cash Flow Metrics: {Trends in cash flow.}
        - Margins: {Gross, operating, net margins.}
        
        #### Risks and Concerns
        - Operational Risks: {Execution or operational challenges.}
        - Valuation Risks: {Potential overvaluation risks.}
        - Macroeconomic Risks: {Geopolitical or economic concerns.}
        - Competitive Risks: {Market or industry competition.}
        
        #### Technical Analysis
        - Trend Analysis: {Stock trend direction.}
        - Support/Resistance Levels: ${X.XX / Y.YY}
        - Volume Trends: {Patterns in trading volume.}
        - Momentum Indicators: {Indicators like RSI or MACD.}
        
        #### Recent Developments
        - {Key news and updates affecting the stock.}
        
        #### Recommendation Justification
        {Detailed reasoning behind the long/short/neutral recommendation.}
        """),
)

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
        use_cached_report: bool = False
    ) -> Iterator[RunResponse]:
        """
        Generate a financial analysis report for a given topic.

        Args:
            topic (str): The topic for which to generate the financial analysis.
            use_search_cache (bool, optional): Whether to use cached search results. Defaults to True.
            use_scrape_cache (bool, optional): Whether to use cached scraped articles. Defaults to True.
            use_cached_report (bool, optional): Whether to return a previously generated report on the same topic. Defaults to False.

        Returns:
            Iterator[RunResponse]: A stream of objects containing the generated report or status updates.

        Workflow Steps:
        1. Check for a cached report if `use_cached_report` is True.
        2. Search the web for articles on the topic (use cache if enabled).
        3. Scrape the article contents (use cache if enabled).
        4. Generate the final financial analysis using the scraped article contents.
        5. Include relevance scores (1–5) and an overall recommendation.
        """
        logger.info(f"Generating a financial analysis on: {topic}")

        # Step 1: Return a cached report if available
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

        # Step 2: Retrieve or perform new web search
        search_results: Optional[SearchResults] = None
        try:
            if use_search_cache and "search_results" in self.session_state:
                search_results = SearchResults.model_validate(self.session_state["search_results"])
                logger.info(f"Found {len(search_results.articles)} articles in cache.")
        except Exception as e:
            logger.warning(f"Could not read search results from cache: {e}")

        if search_results is None:
            web_searcher_response: RunResponse = self.web_searcher.run(topic)
            if (
                web_searcher_response
                and web_searcher_response.content
                and isinstance(web_searcher_response.content, SearchResults)
            ):
                logger.info(f"WebSearcher identified {len(web_searcher_response.content.articles)} articles.")
                search_results = web_searcher_response.content
                self.session_state["search_results"] = search_results.model_dump()

        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content=f"No articles found on the topic: {topic}",
            )
            return

        # Step 3: Scrape articles
        scraped_articles: Dict[str, ScrapedArticle] = {}
        if (
            use_scrape_cache
            and "scraped_articles" in self.session_state
            and isinstance(self.session_state["scraped_articles"], dict)
        ):
            for url, scraped_article in self.session_state["scraped_articles"].items():
                try:
                    validated_scraped_article = ScrapedArticle.model_validate(scraped_article)
                    scraped_articles[validated_scraped_article.url] = validated_scraped_article
                except Exception as e:
                    logger.warning(f"Could not read scraped article from cache: {e}")
            logger.info(f"Found {len(scraped_articles)} scraped articles in cache.")

        for article in search_results.articles:
            if article.url in scraped_articles:
                logger.info(f"Found scraped article in cache: {article.url}")
                continue

            article_scraper_response: RunResponse = self.article_scraper.run(article.url)
            if (
                article_scraper_response
                and article_scraper_response.content
                and isinstance(article_scraper_response.content, ScrapedArticle)
            ):
                scraped_articles[article_scraper_response.content.url] = article_scraper_response.content.model_dump()
                logger.info(f"Scraped article: {article_scraper_response.content.url}")

        self.session_state["scraped_articles"] = {k: v for k, v in scraped_articles.items()}

        # Step 4: Generate the final financial analysis
        logger.info("Generating final financial analysis report")
        financial_analyst_input = {
            "topic": topic,
            "articles": [ScrapedArticle.model_validate(v).model_dump() for v in scraped_articles.values()],
        }

        # Stream responses from the financial_analyst agent
        yield from self.financial_analyst.run(json.dumps(financial_analyst_input, indent=4), stream=True)

        # Cache the final report
        if "reports" not in self.session_state:
            self.session_state["reports"] = []
        self.session_state["reports"].append(
            {"topic": topic, "report": self.financial_analyst.run_response.content}
        )


if __name__ == "__main__":
    topic = "AAPL Stock"

    generate_financial_analysis = GenerateFinancialAnalysis(
        session_id=f"financial-analysis-on-{topic}",
        storage=SqlWorkflowStorage(
            table_name="financial_analysis_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    report_stream: Iterator[RunResponse] = generate_financial_analysis.run(
        topic=topic, 
        use_search_cache=True, 
        use_scrape_cache=True, 
        use_cached_report=False
    )

    pprint_run_response(report_stream, markdown=True)
