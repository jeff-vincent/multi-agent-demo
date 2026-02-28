import os
from dataclasses import dataclass


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", 6379))
    db: int = int(os.getenv("REDIS_DB", 0))
    password: str | None = os.getenv("REDIS_PASSWORD")


@dataclass
class AgentConfig:
    model: str = os.getenv("AGENT_MODEL", "gpt-4o")
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", 10))
    # After this many tool calls, compress context to save space
    compress_every_n_steps: int = int(os.getenv("AGENT_COMPRESS_STEPS", 5))
    # Summarization model (cheaper, faster)
    summary_model: str = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")


REDIS = RedisConfig()
AGENT = AgentConfig()

# Redis key patterns
QUEUE_KEY = "agent:queue"
CONTEXT_KEY_PREFIX = "context"
RESULT_KEY_PREFIX = "result"
RESULTS_CHANNEL = "agent:results"

# TTLs (seconds)
CONTEXT_TTL = 60 * 60 * 24      # 24 hours
RESULT_TTL = 60 * 60 * 2        # 2 hours
QUEUE_BLOCK_TIMEOUT = 5          # BLPOP timeout in seconds
