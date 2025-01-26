from typing import List, Dict
import yfinance as yf
from polygon import RESTClient
from bs4 import BeautifulSoup
import requests
from datetime import datetime


class FinancialDataExtractor:
    def __init__(self, *polygon_api: str):
        self.polygon_client = RESTClient(polygon_api)

    def extract_ticker_info_yahoo(self, tickers: str, start_date: str = None, end_date: str = None):
        data = yf.download([tickers], start_date, end_date)

        return data

    def extract_ticker_info_polygon(self, ticker: str, start_date: str = None, end_date: str = None):
        try:
            quote = self.polygon_client.get_last_quote(ticker)
            details = self.polygon_client.get_ticker_details(ticker)
            aggs = self.polygon_client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                # Can adjust the start date to be something but query info
                from_=start_date or "2023-01-01",
                to=end_date or datetime.now().strftime("%Y-%m-%d"),
            )

            return {
                "quote": {
                    "ask_price": quote.askprice,
                    "ask_size": quote.asksize,
                    "bid_price": quote.bidprice,
                    "bid_size": quote.bidsize,
                },
                "details": details,
            }
        except Exception as e:
            return {"error": str(e)}
    


