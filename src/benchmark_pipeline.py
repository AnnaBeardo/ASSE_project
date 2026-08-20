
import time
import sys
import importlib
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "prompts"))

from llm_client import LLMHandler, PRO_MODEL
from metrics_tracker import MetricsTracker 

PROMPTS_FILE = "caveman_prompts.py" # File containing prompts for benchmarking

def load_prompts_from_file(filename: str) -> list:
    """Dinamically load prompts from a Python file"""
    module_name = filename.replace(".py", "")
    
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "PROMPTS", [])
    except ModuleNotFoundError:
        print(f"Error: Module '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error while loading '{filename}': {e}")
        return []


def run_benchmark():
    print("Initializing LLMHandler and MetricsTracker...")
    handler = LLMHandler(temperature=0.0)
    tracker = MetricsTracker(run_name="benchmark")

    test_prompts = load_prompts_from_file(PROMPTS_FILE)

    if not test_prompts:
        print(f"No prompts found. Please ensure the '{PROMPTS_FILE}' file exists in the 'prompts' directory.")
        return

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Test: '{prompt[:50]}...'")

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
            complexity="BASELINE", 
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
            pipe_logic = result["route_logic"] 
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
            complexity=f"PIPELINE, routing logic: ({pipe_logic})", 
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