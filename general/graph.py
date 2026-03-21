"""General Agent — General-purpose AI assistant powered by a ReAct agent with tools."""
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated
import operator
import os
import subprocess
import json

DESCRIPTION = "General-purpose AI assistant for coding, planning, and analysis tasks"
SKILLS = [
    {"id": "general-coding", "name": "General Coding", "description": "Answer questions, write code, explain concepts"},
    {"id": "planning", "name": "Planning", "description": "Plan and execute multi-step tasks"},
    {"id": "analysis", "name": "Analysis", "description": "Analyze data, code, and systems"},
]


def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "claude-sonnet-oauth"),
        openai_api_base=os.getenv("OPENAI_API_BASE", os.getenv("LITELLM_API_URL", "http://litellm.litellm.svc:4000")),
        openai_api_key=os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_API_KEY", "sk-placeholder")),
    )


@tool
def think(thought: str) -> str:
    """Use this tool to think step by step before answering."""
    return f"Thought: {thought}"


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Web search results for '{query}': [Search functionality requires configuration]"


# Build a ReAct agent with basic tools
# This is functionally similar to Deep Agents but without the dependency issues
TOOLS = [think, web_search]

graph = create_react_agent(
    model=get_llm(),
    tools=TOOLS,
    prompt="""You are a general-purpose AI assistant for the o8s platform.

Your capabilities:
- Answer questions about coding, technology, and software development
- Help with planning and breaking down complex tasks
- Analyze code, systems, and data
- Provide step-by-step guidance

Be helpful, concise, and actionable. Think step by step for complex problems.""",
)
