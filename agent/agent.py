"""
agent/agent.py

Defines the LangChain agent: model + tools + prompt.
The agent itself is stateless — all state lives in the ContextStore.
"""

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from shared.config import AGENT
from shared.search import search_tool
from shared.calculator import calculator_tool

# Register tools here — add your own as needed
TOOLS = [
    search_tool,
    calculator_tool,
]

SYSTEM_PROMPT = """\
You are a capable research and reasoning agent. You have access to tools
to help you complete tasks. Think step by step.

When you have fully completed the task, respond with a clear final answer
prefixed with "FINAL ANSWER:".

If you cannot complete the task in this session, summarize what you have
done so far and what remains, prefixed with "CONTINUATION:".
"""


def build_agent_executor() -> AgentExecutor:
    """
    Build a stateless LangChain agent executor.
    State (message history) is passed in per-invocation from the ContextStore.
    """
    llm = ChatOpenAI(model=AGENT.model, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=AGENT.max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
