"""
agent/worker.py

Queue consumer. Runs forever, pulling task_ids from Redis and executing them.

Each iteration:
  1. BLPOP task_id from queue
  2. Load context from Redis (or initialize fresh)
  3. Run agent loop
  4. Compress + save context back to Redis
  5. If done → publish result
     If continuing → re-enqueue

Run with:
    python -m agent.worker
"""

import json
import logging
import signal
import sys
import time
from typing import Any

import redis
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shared.config import (
    REDIS,
    QUEUE_KEY,
    RESULTS_CHANNEL,
    RESULT_KEY_PREFIX,
    RESULT_TTL,
    QUEUE_BLOCK_TIMEOUT,
    AGENT,
)
from shared.context import ContextStore
from agent.agent import build_agent_executor, SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AgentWorker:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS.host,
            port=REDIS.port,
            db=REDIS.db,
            password=REDIS.password,
        )
        self.executor = build_agent_executor()
        self.running = True

        # Graceful shutdown on SIGTERM / SIGINT
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, *_):
        logger.info("Shutdown signal received. Finishing current task...")
        self.running = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        logger.info("Worker started. Listening on queue: %s", QUEUE_KEY)

        while self.running:
            try:
                self._poll_once()
            except Exception as e:
                logger.exception("Unhandled error in worker loop: %s", e)
                time.sleep(1)  # Brief backoff before retrying

        logger.info("Worker shut down cleanly.")

    def _poll_once(self):
        """Block until a task arrives, then process it."""
        result = self.client.blpop(QUEUE_KEY, timeout=QUEUE_BLOCK_TIMEOUT)
        if result is None:
            return  # Timeout — loop again

        _, raw = result
        task = json.loads(raw)
        task_id = task["task_id"]
        task_hash = task["task_hash"]
        task_input = task["input"]

        logger.info("Picked up task %s (hash: %s)", task_id, task_hash)
        self._process_task(task_id, task_hash, task_input, task)

    # ------------------------------------------------------------------
    # Task processing
    # ------------------------------------------------------------------

    def _process_task(
        self,
        task_id: str,
        task_hash: str,
        task_input: str,
        task_meta: dict,
    ):
        store = ContextStore(self.client, task_hash)

        # Load existing context or initialize fresh
        context = store.load()
        if context is None:
            logger.info("Cold start for task %s", task_id)
            context = store.initialize(
                metadata=task_meta,
                system_prompt=SYSTEM_PROMPT,
            )
        else:
            logger.info(
                "Warm resume for task %s (step %d, status: %s)",
                task_id,
                context["step_count"],
                context["status"],
            )

        store.mark_status("running")

        try:
            output, intermediate_steps = self._run_agent(context, task_input)
        except Exception as e:
            logger.exception("Agent failed on task %s: %s", task_id, e)
            store.mark_status("failed")
            self._publish_result(task_id, task_hash, success=False, error=str(e))
            return

        # Update context with new messages and tool results
        context = self._update_context(context, task_input, output, intermediate_steps)

        # Compress if needed, then save
        context = store.maybe_compress(context)
        store.save(context)

        # Determine if done or continuing
        if output.startswith("FINAL ANSWER:"):
            final_answer = output.removeprefix("FINAL ANSWER:").strip()
            store.mark_status("done")
            self._publish_result(task_id, task_hash, success=True, answer=final_answer)
            logger.info("Task %s completed: %s", task_id, final_answer[:100])

        elif output.startswith("CONTINUATION:"):
            logger.info("Task %s needs more steps, re-enqueuing.", task_id)
            store.mark_status("pending")
            # Re-enqueue with same task — agent resumes from saved context
            self.client.rpush(QUEUE_KEY, json.dumps(task_meta))

        else:
            # Treat anything else as a completed answer
            store.mark_status("done")
            self._publish_result(task_id, task_hash, success=True, answer=output)

    # ------------------------------------------------------------------
    # Agent invocation
    # ------------------------------------------------------------------

    def _run_agent(
        self, context: dict[str, Any], task_input: str
    ) -> tuple[str, list]:
        """
        Invoke the LangChain agent with the current context as chat history.
        Returns (output_text, intermediate_steps).
        """
        result = self.executor.invoke(
            {
                "input": task_input,
                "system_prompt": SYSTEM_PROMPT,
                "chat_history": context["messages"],
            }
        )
        return result["output"], result.get("intermediate_steps", [])

    # ------------------------------------------------------------------
    # Context update
    # ------------------------------------------------------------------

    def _update_context(
        self,
        context: dict[str, Any],
        task_input: str,
        output: str,
        intermediate_steps: list,
    ) -> dict[str, Any]:
        """
        Append the new turn to the message history and record tool results.
        """
        # Record the human input turn
        context["messages"].append(HumanMessage(content=task_input))

        # Record tool calls and results from intermediate steps
        for action, observation in intermediate_steps:
            context["messages"].append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": action.tool_call_id if hasattr(action, "tool_call_id") else "call_0",
                            "name": action.tool,
                            "args": action.tool_input,
                        }
                    ],
                )
            )
            context["messages"].append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=action.tool_call_id if hasattr(action, "tool_call_id") else "call_0",
                )
            )
            context["tool_results"].append(
                {"tool": action.tool, "input": action.tool_input, "output": str(observation)}
            )

        # Record the final AI response
        context["messages"].append(AIMessage(content=output))
        context["step_count"] += len(intermediate_steps)

        return context

    # ------------------------------------------------------------------
    # Result publishing
    # ------------------------------------------------------------------

    def _publish_result(
        self,
        task_id: str,
        task_hash: str,
        success: bool,
        answer: str | None = None,
        error: str | None = None,
    ):
        result = {
            "task_id": task_id,
            "task_hash": task_hash,
            "success": success,
            "answer": answer,
            "error": error,
        }
        result_json = json.dumps(result)

        # Write to a result key (for polling)
        result_key = f"{RESULT_KEY_PREFIX}:{task_id}"
        self.client.set(result_key, result_json, ex=RESULT_TTL)

        # Publish to channel (for pub/sub listeners e.g. orchestrator)
        self.client.publish(RESULTS_CHANNEL, result_json)

        logger.info(
            "Published result for task %s (success=%s)", task_id, success
        )


if __name__ == "__main__":
    worker = AgentWorker()
    worker.run()
