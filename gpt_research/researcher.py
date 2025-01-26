import os
from typing import Any, List, Optional
from gpt_researcher import GPTResearcher
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI


async def get_report(
    query: str,
    report_type: str = "research_report",
    sources: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> str:
    """Run GPT Researcher and return results.

    Args:
        query: The research query to investigate
        report_type: Type of report to generate (default: research_report)
        sources: Optional list of source URLs to research
        save_path: Optional path to save the report

    Returns:
        The generated research report as a string
    """
    researcher = GPTResearcher(
        query=query, report_type=report_type, source_urls=sources
    )
    await researcher.conduct_research()
    report = await researcher.write_report()

    # Save report to the specified path if provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{query.replace(' ', '_')}_report.txt")
        with open(file_path, "w") as file:
            file.write(report)

    return report


async def custom_report(
    research_query: str,
    custom_query: str,
    llm: Optional[BaseChatModel] = None,
    report_type: str = "research_report",
    sources: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> BaseMessage | Any:
    """Generate a custom report by applying a custom query to research results.

    Args:
        research_query: The initial query to research
        custom_query: The custom query to apply to the research results
        llm: Optional custom LLM (defaults to OpenAI GPT-4)
        report_type: Type of report to generate (default: research_report)
        sources: Optional list of source URLs to research
        save_path: Optional path to save the custom report

    Returns:
        The generated custom report as a string
    """
    # Default to OpenAI if no custom LLM provided
    chat_model = (
        llm
        if llm is not None
        else ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,  # More deterministic outputs
            model_kwargs={"seed": 0},  # For reproducibility
        )
    )

    # Get initial research
    research = await get_report(
        query=research_query, report_type=report_type, sources=sources
    )

    # Format prompt with research context
    prompt = f"""Given ONLY the provided context, {custom_query}
    
    <context>
    {research}
    </context>"""

    # Generate custom report using specified LLM
    response = await chat_model.apredict_messages([HumanMessage(content=prompt)])

    # Save the custom report to the specified path if provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{research_query.replace(' ', '_')}_custom_report.txt")
        with open(file_path, "w") as file:
            file.write(response.content)

    return response
