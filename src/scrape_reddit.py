import praw
import praw.models
import praw.models.reddit
import praw.models.reddit.submission


REDDIT = praw.Reddit(
    client_id="t4zG3rmVFn8uKDgNwhtE9g",
    client_secret="cH73VMSESPi2tOXVszzJaxFZqL2TxA",
    user_agent="scraping/1.0 by u/Jousboxx",
)


def scrape_reddit(
    search_term: str, count: int, text_only=True
) -> list[praw.models.reddit.submission.Submission]:
    remaining_count = count
    results = []
    while remaining_count > 0:
        for submission in REDDIT.subreddit("all").search(search_term):
            if text_only:
                if len(submission.selftext) == 0:
                    continue
            results.append(submission)
            remaining_count -= 1

    return results


if __name__ == "__main__":
    results = scrape_reddit("NVDA", 1000)
    for item in results:
        print(item.title)
