import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from llm_client import LLMHandler, PRO_MODEL
from metrics_tracker import MetricsTracker 

# Prompts for testing, insert one or more
TEST_PROMPTS = [
    "Dear AI Assistant, I would like you to help me with a very simple problem: If Rose has 3 apples and gives 1 to Lily, how many apples does she have left? Please provide a clear and concise answer."
]

def run_benchmark():
    print("Initializing LLMHandler and MetricsTracker...")
    handler = LLMHandler(temperature=0.0)

    tracker = MetricsTracker(run_name="AB_test")

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Test: '{prompt[:50]}...'")

        #   ---   Baseline  ---   #
        print("   -> Executing Baseline...")
        start_base = time.time()
        try:
            # Direct call using LangChain
            base_response = handler.pro_model.invoke(prompt).content
            base_latency = time.time() - start_base
        except Exception as e:
            base_response = f"ERROR: {str(e)}"
            base_latency = 0.0

        # Logging baseline metrics
        tracker.log_call(
            model_name=PRO_MODEL,
            complexity="BASELINE (No routing/compression)", 
            prompt=prompt,
            response=base_response,
            latency=base_latency
        )

        #  ---   Pipeline  ---   #
        print("   -> Executing Pipeline ...")
        start_pipe = time.time()
        try:
            result = handler.invoke(prompt)
            pipe_response = result["response"]
            pipe_model = result["routed_to"]
            pipe_logic = result["route_logic"] # 'heuristic' o 'semantic'
            pipe_opt_prompt = result["optimized_prompt_text"]
            pipe_latency = time.time() - start_pipe
        except Exception as e:
            pipe_response = f"ERROR: {str(e)}"
            pipe_model = "ERROR"
            pipe_logic = "ERROR"
            pipe_opt_prompt = prompt
            pipe_latency = 0.0

        # Logging pipeline metrics
        tracker.log_call(
            model_name=pipe_model,
            complexity=f"PIPELINE ({pipe_logic})", 
            prompt=pipe_opt_prompt, 
            response=pipe_response,
            latency=pipe_latency
        )
        
        print(f"   ✓ Done with model: {pipe_model}")

    # ---   Saving Metrics  ---   #
    print("\n" + "="*50)
    tracker.save_to_csv()
    print("="*50)

if __name__ == "__main__":
    run_benchmark()