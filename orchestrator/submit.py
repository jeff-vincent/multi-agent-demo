"""
orchestrator/submit.py

CLI for submitting tasks to the agent queue.

Usage:
    # Submit a high-level goal (auto-decomposes into tasks)
    python -m orchestrator.submit --goal "Research the top agent frameworks in 2025"

    # Submit a single task directly
    python -m orchestrator.submit --task-type research --input "Find recent papers on RAG"

    # Submit and wait for results
    python -m orchestrator.submit --goal "..." --wait
"""

import argparse
import json
import logging

from orchestrator.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Submit tasks to the agent queue")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--goal", help="High-level goal to decompose into tasks")
    group.add_argument("--input", help="Single task input (requires --task-type)")

    parser.add_argument("--task-type", default="general", help="Task type label")
    parser.add_argument(
        "--wait", action="store_true", help="Block until all results are returned"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout in seconds when --wait is set"
    )

    args = parser.parse_args()
    orch = Orchestrator()

    if args.goal:
        task_ids = orch.submit_goal(args.goal)
        print(f"\nSubmitted {len(task_ids)} task(s):")
        for t in task_ids:
            print(f"  {t}")
    else:
        task_id = orch.submit_single(args.task_type, args.input)
        task_ids = [task_id]
        print(f"\nSubmitted task: {task_id}")

    if args.wait:
        print("\nWaiting for results...\n")
        results = orch.wait_for_results(task_ids, timeout=args.timeout)
        for task_id, result in results.items():
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"Success: {result['success']}")
            if result.get("answer"):
                print(f"Answer:\n{result['answer']}")
            if result.get("error"):
                print(f"Error: {result['error']}")
    else:
        print("\nTasks enqueued. Poll for results with:")
        for task_id in task_ids:
            print(f"  redis-cli get result:{task_id}")


if __name__ == "__main__":
    main()
