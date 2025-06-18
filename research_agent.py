# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Overview: This script runs a Haystack research agent that uses MCP tools to create comprehensive research reports.

import pathlib
import asyncio
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.components.generators.utils import print_streaming_chunk

from haystack_integrations.components.connectors.langfuse.langfuse_connector import (
    LangfuseConnector,
)

from haystack_integrations.tools.mcp.mcp_tool import SSEServerInfo
from haystack_integrations.tools.mcp.mcp_toolset import MCPToolset


# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall


async def print_streaming_chunk_async(chunk: StreamingChunk) -> None:
    """
    Default callback function for streaming responses.

    Prints the tokens of the first completion to stdout as soon as they are received
    """
    print(chunk.content, flush=True, end="")

def load_generate_queries_system_message():
    """Load the generate queries system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "generate_queries_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()


def load_web_search_system_message():
    """Load the web search system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "web_search_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()


def load_research_reflection_system_message():
    """Load the research reflection system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "research_reflection_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()


def load_finalize_report_system_message():
    """Load the finalize report system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "finalize_report_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()


def load_research_agent_system_message():
    """Load the main research agent system message from the external file."""
    current_dir = pathlib.Path(__file__).parent
    system_file = current_dir / "research_agent_system_prompt.txt"
    with open(system_file, encoding="utf-8") as f:
        return f.read()


async def main():
    # Use streaming to print the response
    use_streaming = True

    # Initialize LangfuseConnector - it will be active if environment variables are set
    # if you don't want to use Langfuse, just comment out the following line
    # Uncomment this line below to use Langfuse tracing
    tracer = LangfuseConnector("Research Agent")

    
    search_tools = MCPToolset(
        SSEServerInfo(
            url="http://localhost:8106/sse"
        ),
        tool_names=["tavily-search"],
        invocation_timeout=120, # seconds, as tavily takes time to respond
    )

    # Create the LLM instance
    llm = OpenAIChatGenerator(model="gpt-4.1-mini")

    # Create generate queries agent
    generate_queries_agent = Agent(
        system_prompt=load_generate_queries_system_message(),
        chat_generator=llm,
        tools=[],  # No tools needed for query generation
        streaming_callback=print_streaming_chunk if use_streaming else None,
    )

    generate_queries_tool = ComponentTool(
        name="generate_queries_agent",
        description="Analyzes research questions and generates focused search queries. Input: user question.",
        component=generate_queries_agent,
        outputs_to_string={"source": "last_message"}
    )

    # Create web search agent
    web_search_agent = Agent(
        system_prompt=load_web_search_system_message(),
        chat_generator=llm,
        tools=search_tools,
        streaming_callback=print_streaming_chunk if use_streaming else None,
    )

    web_search_tool = ComponentTool(
        name="web_search_agent",
        description="Executes web searches and returns structured results. Input: '[query]'",
        component=web_search_agent,
        # this is needed to store the web_search_results list of messages in the agent state
        # default handler will append messages from all web search tools to the same list of messages
        outputs_to_state={"web_search_results": {"source": "last_message"}}
    )

    # Create research reflection agent
    research_reflection_agent = Agent(
        system_prompt=load_research_reflection_system_message(),
        chat_generator=llm,
        tools=[],  # No tools needed for reflection and synthesis
        streaming_callback=print_streaming_chunk if use_streaming else None,
    )

    research_reflection_tool = ComponentTool(
        name="research_reflection_agent",
        description="Synthesizes search results, evaluates completeness, and determines if additional research is needed. No input is needed.",
        component=research_reflection_agent,        
        inputs_from_state={"web_search_results": "messages"},

        # this is needed to force LLM not to inject some synthesized chat message into the agent run
        # but to actually use the web_search_results list of messages
        parameters={
            "type": "object",
            "properties": {},# no inputs
            "required": []
        },
    )

    # Create finalize report agent
    finalize_report_agent = Agent(
        system_prompt=load_finalize_report_system_message(),
        chat_generator=llm,
        tools=[],  # No tools needed for report generation
        streaming_callback=print_streaming_chunk if use_streaming else None,
    )

    finalize_report_tool = ComponentTool(
        name="finalize_report_agent",
        description="Creates comprehensive, user-friendly research reports. No input is needed.",
        component=finalize_report_agent,
        inputs_from_state={"web_search_results": "messages"},

        # this is needed to force LLM not to inject some synthesized chat message into the agent run
        # but to actually use the web_search_results list of messages
        parameters={
            "type": "object",
            "properties": {},# no inputs
            "required": []
        },
    )

    # Create main research orchestrator agent
    research_agent = Agent(
        system_prompt=load_research_agent_system_message(),
        chat_generator=llm,
        tools=[
            generate_queries_tool,
            web_search_tool,
            research_reflection_tool,
            finalize_report_tool
        ],
        streaming_callback=print_streaming_chunk_async if use_streaming else None,
        state_schema={
            "web_search_results": {"type": list[ChatMessage]}
        }
    )

    try:
        print("Running deep research agent...")
        response = await research_agent.run_async(
            messages=[
                ChatMessage.from_user(text="What's mechanistic interpretability and why is it important in LLMs?")
            ]
        )
        if not use_streaming:
            print(response["messages"][-1].text)
    finally:
        # Clean up MCP toolsets
        search_tools.close()


if __name__ == "__main__":
    asyncio.run(main()) 