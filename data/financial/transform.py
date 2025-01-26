
from .extract import FinancialDataExtractor, RedditDataExtractor

class DataTransformer:
    def __init__(self):
        ... 
    
    class DataToRAGFormatter:
        def __init__(self, financial_extractor: FinancialDataExtractor, reddit_extractor: RedditDataExtractor):
            self.financial_extractor = financial_extractor
            self.reddit_extractor = reddit_extractor

        def format_to_rag(self, ticker: str, subreddit: str, start_date: str = None, end_date: str = None, reddit_limit: int = 10):
            financial_data_yahoo = self.financial_extractor.extract_ticker_info_yahoo(ticker, start_date, end_date)
            financial_data_polygon = self.financial_extractor.extract_ticker_info_polygon(ticker, start_date, end_date)
            reddit_data = self.reddit_extractor.extract_subreddit_data(subreddit, reddit_limit)

            return {
                "ticker": ticker,
                "yahoo_financial_data": financial_data_yahoo,
                "polygon_financial_data": financial_data_polygon,
                "reddit_data": reddit_data,
            }
    
        