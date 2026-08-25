import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# PATH CONFIGURATION (Dynamic resolution across sibling folders)
# -------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Add 'src' and 'prompts' directories to Python path
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "prompts"))

load_dotenv(dotenv_path=project_root / ".env")

from llm_client import LLMHandler, PRO_MODEL, FLASH_MODEL
from metrics_tracker import MetricsTracker

# Dynamic import of dataset from the sibling 'prompts' folder
try:
    from prompts import PROMPTS as TEST_CASES
except ImportError:
    try:
        from test_prompts import PROMPTS as TEST_CASES
    except ImportError:
        print("❌ ERROR: Could not find 'prompts.py' or 'test_prompts.py' in the 'prompts' folder.")
        print(f"Ensure the file exists at: {project_root / 'prompts' / 'prompts.py'}")
        sys.exit(1)

# -------------------------------------------------------------------------
# TOKENIZER AND PRICING SETUP
# -------------------------------------------------------------------------
try:
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        model = genai.GenerativeModel(FLASH_MODEL)
        return model.count_tokens(text).total_tokens

    _TOKENIZER_INFO = f"google.generativeai ({FLASH_MODEL} Native Tokenizer)"
except ImportError:
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    _TOKENIZER_INFO = "Fallback estimation (~1 token per 4 chars)"

# Rates per 1M tokens in USD
COST_PER_1M_TOKENS = {
    FLASH_MODEL: {"input": 0.075, "output": 0.30},
    PRO_MODEL:   {"input": 3.50, "output": 10.50},
}

def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates total cost in USD based on token counts."""
    rates = COST_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

# -------------------------------------------------------------------------
# BENCHMARK ENGINE
# -------------------------------------------------------------------------
def run_master_benchmark():
    print(f"[{_TOKENIZER_INFO}]")
    print(f"Loaded {len(TEST_CASES)} test prompts from 'prompts/' directory.")
    print("Initializing LLM Pipeline and MetricsTracker...")
    
    handler = LLMHandler(temperature=0.0)
    tracker = MetricsTracker(run_name="master_benchmark")

    results = []
    logic_counts = defaultdict(int)
    model_counts = defaultdict(int)

    for i, (category, prompt) in enumerate(TEST_CASES, 1):
        print("\n" + "=" * 90)
        print(f"TEST {i}/{len(TEST_CASES)} | CATEGORY: {category.upper()}")
        print(f"Prompt: '{prompt[:70].strip()}{'...' if len(prompt) > 70 else ''}'")

        # --- A. BASELINE (Direct Pro model call) ---
        start_base = time.time()
        try:
            ai_msg = handler.pro_model.invoke(prompt)
            base_resp = ai_msg.content
            usage = getattr(ai_msg, "usage_metadata", None)
            base_in = usage.get("input_tokens") if usage else count_tokens(prompt)
            base_out = usage.get("output_tokens") if usage else count_tokens(base_resp)
        except Exception as e:
            base_resp, base_in, base_out = f"ERR: {e}", count_tokens(prompt), 0
        
        base_lat = time.time() - start_base
        base_cost = cost_of(PRO_MODEL, base_in, base_out)

        tracker.log_call(PRO_MODEL, f"BASELINE | {category}", prompt, base_resp, base_lat, base_in, base_out)

        # --- B. PIPELINE (Dynamic Routing + Caveman Compression) ---
        start_pipe = time.time()
        try:
            res = handler.invoke(prompt, enable_caveman=True)
            pipe_resp = res.get("response", "")
            routed_to = res.get("routed_to", FLASH_MODEL)
            logic = res.get("route_logic", "error")
            pipe_in = res.get("api_in_tokens") or count_tokens(res.get("optimized_prompt_text", prompt))
            pipe_out = res.get("api_out_tokens") or count_tokens(pipe_resp)
        except Exception as e:
            pipe_resp, routed_to, logic, pipe_in, pipe_out = f"ERR: {e}", "ERROR", "ERROR", base_in, 0
            
        pipe_lat = time.time() - start_pipe
        pipe_cost = cost_of(routed_to, pipe_in, pipe_out)

        tracker.log_call(
            routed_to, 
            f"PIPELINE | {category} | {logic}", 
            res.get("optimized_prompt_text", prompt), 
            pipe_resp, 
            pipe_lat, 
            pipe_in, 
            pipe_out
        )

        # --- METRICS TRACKING AND LOGGING ---
        logic_counts[logic] += 1
        model_counts[routed_to] += 1
        comp_pct = ((base_in - pipe_in) / base_in * 100) if base_in else 0

        print(f"   -> Model Chosen: {routed_to} | Routing Logic: {logic.upper()}")
        print(f"   -> Latency:      {base_lat:.2f}s (Baseline) vs {pipe_lat:.2f}s (Pipeline)")
        print(f"   -> Input Tokens: {base_in} -> {pipe_in} (Compression: {comp_pct:+.1f}%)")
        print(f"   -> Cost (USD):   ${base_cost:.6f} (Baseline) vs ${pipe_cost:.6f} (Pipeline)")

        results.append({
            "b_in": base_in, 
            "p_in": pipe_in, 
            "b_cost": base_cost, 
            "p_cost": pipe_cost
        })

    tracker.save_to_csv()

    # --- FINAL COMPREHENSIVE REPORT ---
    t_b_in = sum(r["b_in"] for r in results)
    t_p_in = sum(r["p_in"] for r in results)
    t_b_cost = sum(r["b_cost"] for r in results)
    t_p_cost = sum(r["p_cost"] for r in results)
    
    saved_tokens = t_b_in - t_p_in
    saved_tokens_pct = (saved_tokens / t_b_in * 100) if t_b_in else 0
    saved_cost = t_b_cost - t_p_cost
    saved_cost_pct = (saved_cost / t_b_cost * 100) if t_b_cost else 0

    print("\n" + "█" * 90)
    print("🏆 MASTER BENCHMARK REPORT")
    print("█" * 90)
    print("Dataset Source        -> prompts/prompts.py")
    print(f"Model Distribution    -> Flash: {model_counts.get(FLASH_MODEL, 0)} | Pro: {model_counts.get(PRO_MODEL, 0)}")
    print(f"Routing Distribution  -> Heuristic: {logic_counts.get('heuristic', 0)} | Semantic: {logic_counts.get('semantic', 0)}")
    print("-" * 90)
    print(f"Input Tokens Saved:      {saved_tokens} ({saved_tokens_pct:.1f}%)")
    print(f"Total BASELINE Cost:     ${t_b_cost:.6f}")
    print(f"Total PIPELINE Cost:     ${t_p_cost:.6f}")
    print(f"Net Financial Savings:   ${saved_cost:.6f} ({saved_cost_pct:.1f}%)")
    print("█" * 90)

if __name__ == "__main__":
    run_master_benchmark()