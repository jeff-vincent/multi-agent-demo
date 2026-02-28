"""
orchestrator/orchestrator.py

Decomposes a high-level goal into tasks, seeds context if needed,
and enqueues tasks to the Redis queue.

The orchestrator also subscribes to the results channel to track
completion across all tasks.
"""

import json
import logging
import uuid
from typing import Any

import redis
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import REDIS, QUEUE_KEY, RESULTS_CHANNEL, RESULT_KEY_PREFIX, RESULT_TTL
from agent.context import ContextStore, make_task_hash

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS.host,
            port=REDIS.port,
            db=REDIS.db,
            password=REDIS.password,
        )
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_goal(self, goal: str) -> list[str]:
        """
        Decompose a goal into tasks and enqueue them.
        Returns a list of task_ids.
        """
        tasks = self._decompose(goal)
        task_ids = []

        for task in tasks:
            task_id = self._enqueue_task(
                task_type=task["type"],
                task_input=task["input"],
                shared_context=task.get("shared_context"),
            )
            task_ids.append(task_id)
            logger.info("Enqueued task %s: %s", task_id, task["input"][:80])

        return task_ids

    def submit_single(self, task_type: str, task_input: str) -> str:
        """Submit a single task directly. Returns task_id."""
        return self._enqueue_task(task_type, task_input)

    def wait_for_results(self, task_ids: list[str], timeout: int = 300) -> dict[str, Any]:
        """
        Subscribe to results channel and collect results for all task_ids.
        Returns dict of task_id → result.
        """
        remaining = set(task_ids)
        results = {}

        pubsub = self.client.pubsub()
        pubsub.subscribe(RESULTS_CHANNEL)

        logger.info("Waiting for %d task(s)...", len(remaining))

        for message in pubsub.listen():
            if message["type"] != "message":
                continue

            result = json.loads(message["data"])
            task_id = result["task_id"]

            if task_id in remaining:
                results[task_id] = result
                remaining.discard(task_id)
                logger.info(
                    "Received result for task %s (success=%s). %d remaining.",
                    task_id,
                    result["success"],
                    len(remaining),
                )

            if not remaining:
                break

        pubsub.unsubscribe()
        return results

    def poll_result(self, task_id: str) -> dict | None:
        """
        Poll for a result without blocking (returns None if not ready).
        Useful if you don't want to use pub/sub.
        """
        raw = self.client.get(f"{RESULT_KEY_PREFIX}:{task_id}")
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------
    # Task decomposition
    # ------------------------------------------------------------------

    def _decompose(self, goal: str) -> list[dict]:
        """
        Use an LLM to break a high-level goal into discrete agent tasks.
        Returns a list of {type, input, shared_context?} dicts.
        """
        response = self.llm.invoke(
            [
                SystemMessage(
                    content="""\
You decompose complex goals into a list of discrete agent tasks.
Respond ONLY with a valid JSON array. Each item must have:
  - "type": short snake_case label (e.g. "research", "summarize", "analyze")
  - "input": the specific instruction for the agent
  - "shared_context": optional string of background info to seed the agent with

Example:
[
  {"type": "research", "input": "Find the top 3 open source agent frameworks in 2025"},
  {"type": "analyze", "input": "Compare their GitHub stars, licenses, and community size"}
]
"""
                ),
                HumanMessage(content=f"Goal: {goal}"),
            ]
        )

        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        tasks = json.loads(raw.strip())
        logger.info("Decomposed goal into %d task(s)", len(tasks))
        return tasks

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def _enqueue_task(
        self,
        task_type: str,
        task_input: str,
        shared_context: str | None = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        task_hash = make_task_hash(task_type, {"input": task_input})

        # Optionally seed context with shared background knowledge
        # before the agent even starts — it wakes up with a warm context
        if shared_context:
            self._seed_context(task_hash, task_type, task_input, shared_context)

        task_payload = {
            "task_id": task_id,
            "task_hash": task_hash,
            "type": task_type,
            "input": task_input,
        }

        self.client.rpush(QUEUE_KEY, json.dumps(task_payload))
        return task_id

    def _seed_context(
        self,
        task_hash: str,
        task_type: str,
        task_input: str,
        shared_context: str,
    ):
        """
        Pre-populate Redis context before the worker picks up the task.
        The agent wakes up with this background already in its memory.
        """
        store = ContextStore(self.client, task_hash)
        if store.load() is not None:
            logger.debug("Context already exists for hash %s, skipping seed.", task_hash)
            return

        system_prompt = (
            f"You are a {task_type} agent. You have been given the following "
            f"background context to help you complete your task:\n\n{shared_context}"
        )
        store.initialize(
            metadata={"type": task_type, "input": task_input},
            system_prompt=system_prompt,
        )
        logger.info("Seeded context for task hash %s", task_hash)
