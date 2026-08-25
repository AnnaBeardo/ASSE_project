import time

from llm_client import LLMHandler
from metrics_tracker import MetricsTracker


TEST_PROMPTS = [
    "What is the capital of Italy?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a Python function that sorts a list of integers and removes duplicates.",
    "Solve step by step: if a train travels 120 miles in 2 hours, what is its average speed?",
]


def main():
    handler = LLMHandler()
    tracker = MetricsTracker(run_name="pipeline_test")

    for i, prompt in enumerate(TEST_PROMPTS, start=1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}")
        print(f"{'=' * 80}")
        print(f"Original prompt: {prompt}")

        start_time = time.time()

        result = handler.invoke(
            prompt,
            enable_caveman=True
        )

        latency = time.time() - start_time

        routed_to = result["routed_to"]
        complexity = "high" if "gemma" in routed_to.lower() else "low"

        tracker.log_call(
            model_name=routed_to,
            complexity=complexity,
            prompt=result["optimized_prompt_text"],
            response=result["response"],
            latency=latency
        )

        stats = result["compression_stats"]

        print(f"Routing logic: {result['route_logic']}")
        print(f"Model: {routed_to}")
        print(
            "Compression: "
            f"{stats['original_tokens']} -> {stats['compressed_tokens']} tokens "
            f"(saved {stats['saved_tokens']})"
        )
        print(f"Optimized prompt: {result['optimized_prompt_text']}")
        print(f"Response: {result['response']}")
        print(f"Latency: {latency:.4f} s")

    tracker.save_to_csv()


if __name__ == "__main__":
    main()
