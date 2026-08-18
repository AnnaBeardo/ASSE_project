"""
Tests the whole system and measures token savings, comparing with a baseline scenario.

For each prompt it measaures:
- Raw input tokens 
- Actual input tokens sent to the model 
- Output tokens 
- Which model handled the request 
- Which routing logic decided the routing (heuristic / semantic)


!!! IMPORTANTE:  AGGIUSTARE LA QUESTIONE PREZZI !!! 
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from llm_client import LLMHandler, PRO_MODEL, FLASH_MODEL

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text))

    _TOKENIZER_INFO = "tiktoken (cl100k_base) "
except ImportError:
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        # Raw estimation
        return max(1, len(text) // 4)

    _TOKENIZER_INFO = "Tiktoken not installed, using rough estimate (1 token ~ 4 chars)"


# Placeholder prices
# !!! DA CAMBIARE !!! 
COST_PER_1K_TOKENS = {
    FLASH_MODEL: {"input": 0.0002, "output": 0.0006},
    PRO_MODEL:   {"input": 0.0035, "output": 0.0105},
}


TEST_PROMPTS = [
    "```python\ndef calculate_fibonacci(n):\n    pass\n```",
    "Write a SQL query: SELECT * FROM users WHERE age > 30",
    "Design a microservices architecture and output the OpenAPI 3.0 schema.",
    "What is the capital of Italy?",
    "Say hello",
    "Che ore sono?",
    "Explain why the sky is blue step by step.",
    "Give me a recipe for a chocolate cake.",
    "Write a brief email to my boss asking for vacation days.",
    "Solve this equation step by step and explain your reasoning: 3x + 5 = 20.",
    "Give me the JSON schema for a user profile with nested address fields.",
    "Draw a flowchart describing the checkout process of an e-commerce site.",
    "Summarize the trade-offs between REST and GraphQL.",
    "Outline the pros and cons of remote work.",
    "Hi, how are you?",
]


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS[model]
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def run_token_savings_test():
    print(f"Used tokenizer: {_TOKENIZER_INFO}")
    print("Initializing LLM pipeline...")
    handler = LLMHandler(temperature=0.0)
    print("-" * 90)

    rows = []
    route_logic_counts = defaultdict(int)
    model_counts = defaultdict(int)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\nTest {i}/{len(TEST_PROMPTS)}: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")

        raw_input_tokens = count_tokens(prompt)

        start = time.time()
        try:
            result = handler.invoke(prompt, enable_caveman=True)
        except Exception as e:
            print(f"EXECUTION ERROR: {e}")
            continue
        latency = time.time() - start

        routed_to = result.get("routed_to")
        route_logic = result.get("route_logic")
        optimized_text = result.get("optimized_prompt_text", "")
        response_text = result.get("response", "")

        actual_input_tokens = count_tokens(optimized_text)
        output_tokens = count_tokens(response_text)

        # --- Scenario reale: modello scelto dal routing, prompt compresso ---
        real_cost = cost_of(routed_to, actual_input_tokens, output_tokens)

        # --- Scenario baseline: sempre Pro, prompt NON compresso ---
        # Nota: per l'output usiamo lo stesso numero di token della risposta reale.
        # E' una semplificazione (un output "senza compressione" potrebbe differire
        # leggermente), ma isola bene l'effetto di routing + compressione sull'input.
        baseline_cost = cost_of(PRO_MODEL, raw_input_tokens, output_tokens)

        rows.append({
            "prompt": prompt,
            "routed_to": routed_to,
            "route_logic": route_logic,
            "raw_input_tokens": raw_input_tokens,
            "actual_input_tokens": actual_input_tokens,
            "output_tokens": output_tokens,
            "real_cost": real_cost,
            "baseline_cost": baseline_cost,
            "latency": latency,
        })

        route_logic_counts[route_logic] += 1
        model_counts[routed_to] += 1

        compression_pct = (
            100 * (1 - actual_input_tokens / raw_input_tokens) if raw_input_tokens else 0
        )
        print(f"   Modello: {routed_to} | Logica: {route_logic}")
        print(f"   Token input: {raw_input_tokens} -> {actual_input_tokens} "
              f"(compressione: {compression_pct:+.1f}%)")
        print(f"   Token output: {output_tokens} | Tempo: {latency:.2f}s")
        print(f"   Costo reale: ${real_cost:.6f} | Costo baseline (sempre Pro, no compressione): ${baseline_cost:.6f}")

    if not rows:
        print("\nNessun test completato con successo.")
        return

    total_raw_input = sum(r["raw_input_tokens"] for r in rows)
    total_actual_input = sum(r["actual_input_tokens"] for r in rows)
    total_output = sum(r["output_tokens"] for r in rows)
    total_real_cost = sum(r["real_cost"] for r in rows)
    total_baseline_cost = sum(r["baseline_cost"] for r in rows)

    tokens_saved_by_compression = total_raw_input - total_actual_input
    cost_saved = total_baseline_cost - total_real_cost
    cost_saved_pct = (cost_saved / total_baseline_cost * 100) if total_baseline_cost else 0

    print("\n" + "=" * 90)
    print("📊 FINAL REPORT - TOKEN SAVINGS AND COST")
    print("=" * 90)
    print(f"Prompts tested: {len(rows)}")
    print(f"Routing -> Flash: {model_counts.get(FLASH_MODEL, 0)} | Pro: {model_counts.get(PRO_MODEL, 0)}")
    print(f"Logic   -> Heuristic: {route_logic_counts.get('heuristic', 0)} | "
          f"Semantic: {route_logic_counts.get('semantic', 0)}")
    print("-" * 90)
    print(f"Raw input tokens (original prompts):          {total_raw_input}")
    print(f"Actual input tokens (after compression):      {total_actual_input}")
    print(f"Tokens saved by compression:                  {tokens_saved_by_compression} "
          f"({tokens_saved_by_compression / total_raw_input * 100:.1f}% of input)")
    print(f"Total output tokens:                          {total_output}")
    print("-" * 90)
    print(f"REAL scenario cost (hybrid routing + compression):  ${total_real_cost:.6f}")
    print(f"BASELINE scenario cost (always Pro, no compression): ${total_baseline_cost:.6f}")
    print(f"Estimated total savings:                             ${cost_saved:.6f} "
          f"({cost_saved_pct:.1f}%)")
    print("=" * 90)
    print("\nNote: The prices used in COST_PER_1K_TOKENS are indicative placeholders.")
    print("The absolute dollar amount is not reliable until you update it with the")
    print("actual rates of your Gemini plan. The savings percentage and the amount of")
    print("tokens saved, however, are already informative as they are.")


if __name__ == "__main__":
    run_token_savings_test()