"""Example usage of GPT Research package."""

import asyncio

from gpt_research import custom_report, get_report


async def main():
    """Run example research queries."""
    # Example 1: Basic research
    print("Running basic research query...")
    report = await get_report(
        query="What are the latest developments in AI and finance?",
        report_type="research_report",
    )
    print("\nBasic Research Report:")
    print(report)

    # Example 2: Research with specific sources
    print("\nRunning research with specific sources...")
    report = await get_report(
        query="What is the impact of AI on stock trading?",
        sources=[
            "https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp",
            "https://www.forbes.com/sites/bernardmarr/2021/07/05/the-best-examples-of-artificial-intelligence-in-use-today/",
        ],
    )
    print("\nResearch Report with Specific Sources:")
    print(report)

    # Example 3: Custom report generation
    print("\nGenerating custom report...")
    report = await custom_report(
        research_query="What are the current applications of AI in financial markets?",
        custom_query="Create a bulleted list of the top 5 most impactful AI applications in finance, including their benefits and potential risks",
    )
    print("\nCustom Report:")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
