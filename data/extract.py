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
    
class RedditDataExtractor:
    def extract_subreddit_data(self, subreddit_name: str, limit: int = 10):
        try:
            base_url = f"https://www.reddit.com/r/{subreddit_name}/new/"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(base_url, headers=headers)

            if response.status_code != 200:
                return {"error": f"Failed to fetch subreddit data: {response.status_code}"}

            soup = BeautifulSoup(response.content, "html.parser")
            posts = []

            post_elements = soup.find_all("div", class_="Post", limit=limit)
            for post in post_elements:
                title_element = post.find("h3")
                if not title_element:
                    continue

                title = title_element.text
                link_element = post.find("a", href=True)
                url = link_element["href"] if link_element else None
                score_element = post.find("div", class_="score")
                score = int(score_element.text) if score_element else None
                num_comments_element = post.find("span", class_="num-comments")
                num_comments = int(num_comments_element.text) if num_comments_element else None

                posts.append({
                    "title": title,
                    "url": f"https://www.reddit.com{url}" if url else None,
                    "score": score,
                    "num_comments": num_comments,
                })

            return posts
        except Exception as e:
            return {"error": str(e)}


