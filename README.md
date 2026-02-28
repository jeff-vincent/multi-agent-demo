# Resumable Agent Context Cache

A LangChain multi-agent system with Redis-backed context caching and queue-based task dispatch.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Orchestrator                                           │
│  - Receives high-level goal                             │
│  - Decomposes into tasks                                │
│  - hash(task) → check Redis for warm context           │
│  - Enqueues task_ids → Redis queue                      │
└────────────────────┬────────────────────────────────────┘
                     │  RPUSH task_id
                     ▼
              Redis Queue (LIST)
              "agent:queue"
                     │  BLPOP
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Agent Workers (N instances)                            │
│  - Pull task_id from queue                              │
│  - Load context:{task_hash} from Redis                  │
│  - Run LangChain agent loop                             │
│  - Compress + write context back                        │
│  - On done: publish to results channel                  │
│  - On continue: re-enqueue self                         │
└─────────────────────────────────────────────────────────┘
                     │
                     ▼
              Redis Hash
              context:{task_hash}
              - messages (compressed)
              - tool_results
              - status
              - metadata
```

## Key ideas

- **Task hash** = sha256 of (task_type + input + version). Same task → same key → cached context.
- **Resumability** — worker dies mid-task? Another picks it up. Context is in Redis, not in-process.
- **Horizontal scaling** — spin up N workers, they all consume from the same queue.
- **Context compression** — every N tool calls, agent summarizes its own history to keep Redis lean.
- **Pre-loaded context** — orchestrator can seed context before enqueuing (inject shared knowledge).

## Quickstart

```bash
# Start Redis
docker run -d -p 6379:6379 redis:7

# Install deps
pip install -r requirements.txt

# Set your OpenAI key
export OPENAI_API_KEY=sk-...

# Run a worker
python -m agent.worker

# In another terminal, submit a task
python -m orchestrator.submit --goal "Research the latest trends in multi-agent AI systems"
```

## Project layout

```
agent/
  context.py       # Redis context store (serialize, load, compress)
  worker.py        # Queue consumer + agent loop
  agent.py         # LangChain agent definition + tools

orchestrator/
  submit.py        # CLI to submit tasks
  orchestrator.py  # Task decomposition + enqueue logic

tools/
  search.py        # Example tool: web search
  calculator.py    # Example tool: calculator

config.py          # Redis connection, queue names, TTLs
requirements.txt
```
