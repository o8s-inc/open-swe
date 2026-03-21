"""Supervisor Agent — Routes tasks to specialized sub-agents (devops, code-review, ops, docs, general)."""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import operator
import os
import json

DESCRIPTION = "Routes tasks to specialized sub-agents (devops, code-review, ops, docs, general)"
SKILLS = [
    {"id": "route-devops", "name": "Route DevOps", "description": "Detect and route devops/monitoring tasks"},
    {"id": "route-code-review", "name": "Route Code Review", "description": "Detect and route code review tasks"},
    {"id": "route-ops", "name": "Route Ops", "description": "Detect and route ops/cluster health tasks"},
    {"id": "route-docs", "name": "Route Docs", "description": "Detect and route documentation tasks"},
    {"id": "route-general", "name": "Route General", "description": "Detect and route general coding tasks"},
]

class SupervisorState(TypedDict):
    user_message: str
    intent: str
    intent_confidence: float
    selected_agent: str
    agent_input: dict
    agent_output: dict
    final_response: str
    messages: Annotated[list, operator.add]

def get_llm(model: str = None):
    """Get LLM with sonnet for better reasoning in supervisor."""
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "claude-sonnet-oauth"),
        openai_api_base=os.getenv("OPENAI_API_BASE", os.getenv("LITELLM_API_URL", "http://litellm.litellm.svc:4000")),
        openai_api_key=os.getenv("OPENAI_API_KEY", os.getenv("LITELLM_API_KEY", "sk-placeholder")),
    )

def detect_intent(state: SupervisorState) -> SupervisorState:
    """Analyze user message and detect the appropriate agent."""
    llm = get_llm()
    
    agents_info = {
        "devops": "Monitors Sentry errors and Grafana metrics, creates Linear tickets, and proposes fixes",
        "code_review": "Reviews GitHub pull requests with documentation-aware analysis",
        "ops": "Checks Kubernetes cluster health, triages alerts, and suggests operational actions",
        "docs": "Monitors repository changes and keeps documentation up to date",
        "general": "General-purpose AI assistant for coding tasks (uses Deep Agents)",
    }
    
    system_prompt = f"""You are an intelligent routing agent for the o8s platform.

Your job is to analyze the user's request and route it to the appropriate specialized agent.

Available agents:
{json.dumps(agents_info, indent=2)}

For each request, determine:
1. The primary intent (one of: devops, code_review, ops, docs, general)
2. Confidence level (0.0-1.0)
3. Any additional context needed

Return a JSON object with: {{"intent": "agent_name", "confidence": float, "context": {{...}}}}

Guidelines:
- Use "devops" for monitoring, alerts, error tracking, ticket creation
- Use "code_review" for PR reviews, code analysis, pull requests
- Use "ops" for cluster health, Kubernetes issues, infrastructure
- Use "docs" for documentation updates, README changes, knowledge base
- Use "general" for general coding tasks, code generation, refactoring
"""
    
    user_prompt = f"""Analyze this request and determine the appropriate agent:

"{state['user_message']}"

Return only valid JSON."""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    
    # Parse JSON from response
    try:
        content = response.content.strip()
        # Try to extract JSON from markdown or extra text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        
        result = json.loads(content)
        intent = result.get("intent", "general")
        confidence = float(result.get("confidence", 0.5))
        context = result.get("context", {})
    except json.JSONDecodeError:
        intent = "general"
        confidence = 0.3
        context = {}
    
    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "selected_agent": intent,
        "agent_input": {"input": state["user_message"], **context},
        "messages": [response]
    }

def route_to_agent(state: SupervisorState) -> SupervisorState:
    """Prepare input for the selected agent."""
    return {
        **state,
        "messages": [AIMessage(content=f"Routing to {state['selected_agent']} agent...")]
    }

def collect_result(state: SupervisorState) -> SupervisorState:
    """Collect result from the agent and format final response."""
    agent_output = state.get("agent_output", {})
    
    # Format the response
    output_text = agent_output.get("output", "")
    events = agent_output.get("events", [])
    
    if events:
        # Extract text from events
        event_texts = []
        for event in events:
            if isinstance(event, dict) and "output" in event:
                event_texts.append(event["output"])
        output_text = "\n\n".join(event_texts)
    
    # Generate final response with context
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that synthesizes agent results for the user."),
        HumanMessage(content=f"""The {state['selected_agent']} agent processed your request:

Original request: "{state['user_message']}"

Agent output:
{output_text}

Provide a concise summary of what was done and the final result."""),
    ])
    
    return {
        **state,
        "final_response": response.content,
        "messages": [response]
    }

def fallback_to_general(state: SupervisorState) -> SupervisorState:
    """If confidence is low, use general agent."""
    if state.get("intent_confidence", 1.0) < 0.5:
        return {
            **state,
            "selected_agent": "general",
            "agent_input": {"input": state["user_message"]},
        }
    return state

# Build the graph
builder = StateGraph(SupervisorState)

builder.add_node("detect_intent", detect_intent)
builder.add_node("route_to_agent", route_to_agent)
builder.add_node("collect_result", collect_result)
builder.add_node("fallback_to_general", fallback_to_general)

# Flow: detect intent -> check confidence -> route -> collect result
builder.set_entry_point("detect_intent")
builder.add_edge("detect_intent", "fallback_to_general")
builder.add_edge("fallback_to_general", "route_to_agent")
builder.add_edge("route_to_agent", "collect_result")
builder.add_edge("collect_result", END)

graph = builder.compile()
