"""
agent/context.py

Manages per-task agent context in Redis.

Each task gets a hash key: context:{task_hash}
Fields:
  - messages     JSON list of LangChain message dicts
  - tool_results JSON list of tool call results
  - status       "pending" | "running" | "done" | "failed"
  - step_count   int - total tool calls made so far
  - metadata     JSON dict of task input + type + version
  - created_at   ISO timestamp
  - updated_at   ISO timestamp
"""

import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any

import redis
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_openai import ChatOpenAI

from config import REDIS, CONTEXT_KEY_PREFIX, CONTEXT_TTL, AGENT

logger = logging.getLogger(__name__)


def make_task_hash(task_type: str, task_input: dict, version: str = "v1") -> str:
    """
    Deterministic hash for a task. Same inputs → same hash → cache hit.
    Include version to bust the cache when agent logic changes.
    """
    payload = json.dumps(
        {"type": task_type, "input": task_input, "version": version},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def context_key(task_hash: str) -> str:
    return f"{CONTEXT_KEY_PREFIX}:{task_hash}"


class ContextStore:
    """
    Redis-backed context store for a single agent task.
    Handles serialization, compression, and TTL management.
    """

    def __init__(self, client: redis.Redis, task_hash: str):
        self.client = client
        self.task_hash = task_hash
        self.key = context_key(task_hash)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any] | None:
        """
        Load context from Redis. Returns None if no context exists (cold start).
        """
        raw = self.client.hgetall(self.key)
        if not raw:
            return None

        messages_raw = json.loads(raw[b"messages"])
        messages = messages_from_dict(messages_raw)

        return {
            "messages": messages,
            "tool_results": json.loads(raw[b"tool_results"]),
            "status": raw[b"status"].decode(),
            "step_count": int(raw[b"step_count"]),
            "metadata": json.loads(raw[b"metadata"]),
            "created_at": raw[b"created_at"].decode(),
            "updated_at": raw[b"updated_at"].decode(),
        }

    def save(self, context: dict[str, Any]) -> None:
        """
        Write context to Redis and reset TTL.
        """
        now = datetime.now(timezone.utc).isoformat()

        pipe = self.client.pipeline()
        pipe.hset(
            self.key,
            mapping={
                "messages": json.dumps(messages_to_dict(context["messages"])),
                "tool_results": json.dumps(context["tool_results"]),
                "status": context["status"],
                "step_count": context["step_count"],
                "metadata": json.dumps(context["metadata"]),
                "created_at": context.get("created_at", now),
                "updated_at": now,
            },
        )
        pipe.expire(self.key, CONTEXT_TTL)
        pipe.execute()

        logger.debug(
            "Saved context %s (status=%s, steps=%d, messages=%d)",
            self.task_hash,
            context["status"],
            context["step_count"],
            len(context["messages"]),
        )

    def initialize(self, metadata: dict, system_prompt: str) -> dict[str, Any]:
        """
        Create a fresh context for a new task.
        """
        now = datetime.now(timezone.utc).isoformat()
        context = {
            "messages": [SystemMessage(content=system_prompt)],
            "tool_results": [],
            "status": "pending",
            "step_count": 0,
            "metadata": metadata,
            "created_at": now,
            "updated_at": now,
        }
        self.save(context)
        logger.info("Initialized fresh context for task %s", self.task_hash)
        return context

    def mark_status(self, status: str) -> None:
        """Lightweight status update without rewriting everything."""
        self.client.hset(self.key, "status", status)
        self.client.hset(
            self.key, "updated_at", datetime.now(timezone.utc).isoformat()
        )
        self.client.expire(self.key, CONTEXT_TTL)

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    def maybe_compress(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        If step_count is a multiple of compress_every_n_steps, summarize
        the message history and replace it with a compressed version.

        This keeps Redis lean and prevents context window bloat.
        """
        step_count = context["step_count"]
        if step_count == 0 or step_count % AGENT.compress_every_n_steps != 0:
            return context

        logger.info(
            "Compressing context for task %s at step %d", self.task_hash, step_count
        )

        messages = context["messages"]

        # Keep the system message, compress everything else
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) < 3:
            # Nothing worth compressing yet
            return context

        summary = self._summarize(other_msgs)
        compressed_msg = HumanMessage(
            content=(
                f"[CONTEXT SUMMARY — compressed at step {step_count}]\n\n"
                f"{summary}\n\n"
                f"[End of summary. Continue from here.]"
            )
        )

        context["messages"] = system_msgs + [compressed_msg]
        logger.info(
            "Compressed %d messages → 1 summary for task %s",
            len(other_msgs),
            self.task_hash,
        )
        return context

    def _summarize(self, messages: list[BaseMessage]) -> str:
        """
        Use a cheap model to summarize the message history.
        """
        llm = ChatOpenAI(model=AGENT.summary_model, temperature=0)

        transcript = "\n\n".join(
            f"[{type(m).__name__}]: {m.content}" for m in messages
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a concise summarizer. "
                        "Summarize the following agent conversation history, "
                        "preserving all key facts, tool results, decisions made, "
                        "and what the agent was in the middle of doing. "
                        "Be specific — this summary will be used to resume the task."
                    )
                ),
                HumanMessage(content=transcript),
            ]
        )
        return response.content
