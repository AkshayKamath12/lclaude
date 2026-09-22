"""Smoke test & latency benchmark utility for local Ollama engine.

Usage:
    python -m scripts.check_engine
"""

import time
import sys
from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEngineError,
)


def run_diagnostics(model: str = "qwen2.5:7b-instruct") -> None:
    print(f"=== Ollama Engine Diagnostics [{model}] ===")
    engine = InferenceEngine(model=model)

    # 1. Readiness Check
    sys.stdout.write("[1/2] Verifying server readiness & model availability... ")
    sys.stdout.flush()
    try:
        engine.verify_ready()
        print("PASS")
    except OllamaConnectionError as e:
        print("FAIL")
        print(f"\n[Connection Error]: {e}")
        return
    except ModelNotFoundError as e:
        print("FAIL")
        print(f"\n[Model Error]: {e}")
        return
    except OllamaEngineError as e:
        print("FAIL")
        print(f"\n[Engine Error]: {e}")
        return

    # 2. Live Stream & Benchmark
    print("\n[2/2] Streaming test prompt:")
    prompt = "Count from 1 to 5, one number per line. No extra text."
    print(f"Prompt: '{prompt}'\n---")

    start_time = time.time()
    first_token_time = None
    token_count = 0

    try:
        for token in engine.stream_chat([{"role": "user", "content": prompt}]):
            if first_token_time is None:
                first_token_time = time.time()
            sys.stdout.write(token)
            sys.stdout.flush()
            token_count += 1

        total_time = time.time() - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0.0

        print("\n---")
        print(f"Metrics:")
        print(f"  - Time to First Token (TTFT): {ttft:.2f}s")
        print(f"  - Total Generation Time:     {total_time:.2f}s")
        print(f"  - Status: OK\n")

    except Exception as exc:
        print(f"\n[Stream Failed]: {exc}")


if __name__ == "__main__":
    run_diagnostics()