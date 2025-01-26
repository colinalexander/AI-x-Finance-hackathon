# GPT Research

A Python package for research automation using GPT Researcher with custom report generation capabilities.

## Features

- Async-based research API
- Custom report generation with LLM flexibility
- Support for specific source URLs
- Langchain integration for custom LLMs

## Usage

### Basic Research

```python
import asyncio
from gpt_research import get_report

async def main():
    # Basic research query
    report = await get_report(
        query="What are the latest advancements in AI?",
        report_type="research_report"
    )
    print(report)

    # Research with specific sources
    report = await get_report(
        query="What is artificial intelligence?",
        sources=["https://en.wikipedia.org/wiki/Artificial_intelligence"]
    )
    print(report)

asyncio.run(main())
```

### Custom Reports

```python
import asyncio
from gpt_research import custom_report
from langchain.chat_models import ChatAnthropic  # Example custom LLM

async def main():
    # Default OpenAI LLM
    report = await custom_report(
        research_query="What are the environmental impacts of electric vehicles?",
        custom_query="Summarize the key findings in bullet points"
    )
    print(report)

    # Custom LLM
    custom_llm = ChatAnthropic()
    report = await custom_report(
        research_query="What are the latest developments in quantum computing?",
        custom_query="Create a timeline of major breakthroughs",
        llm=custom_llm
    )
    print(report)

asyncio.run(main())
```

## Running Examples

To run the example file using uv:

```bash
uv run python -m gpt_research.example
```

This will demonstrate:

1. Basic research queries
2. Research with specific sources
3. Custom report generation with LLM processing
