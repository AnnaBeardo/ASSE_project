import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from llm_client import LLMHandler, PRO_MODEL
from metrics_tracker import MetricsTracker 

# Prompts for testing, insert one or more
TEST_PROMPTS = [
    # =========================================================
    # 1. TASK STILE "CAVEMAN" (Software Engineering & Refactoring)
    # Atteso: Modello PRO. Ottimo per testare la % di token salvati.
    # =========================================================
    "My React component is re-rendering infinitely. I am passing an inline object as a prop to a child component, and using it in a useEffect dependency array. Explain exactly why this React re-render bug happens and how to fix it.",
    
    "My Express auth middleware is letting expired JWT tokens through. The expiry check uses Date.now() compared to the token's exp field. What's wrong and how do I fix this token expiry bug?",
    
    "Set up a robust PostgreSQL connection pool using the 'pg' library in Node.js. Include error handling for idle clients.",
    
    "Explain the architectural difference between git rebase and git merge. When should a team use rebase vs merge in a collaborative environment?",
    
    "```javascript\ngetData(function(a){\n  parseData(a, function(b){\n    saveData(b, function(c){\n      console.log('done');\n    });\n  });\n});\n```\nRefactor this nested callback hell to use modern async/await syntax.",
    
    "Compare a Microservices architecture with a Monolith. Detail the pros and cons regarding deployment, scalability, and operational overhead.",
    
    "Provide a Dockerfile for a multi-stage build of a React application served by Nginx. I want the final image to be as small as possible.",

    # =========================================================
    # 2. RAGIONAMENTO E MATEMATICA (High Complexity)
    # Atteso: Modello PRO (Catturato dall'Euristica per keyword/math)
    # =========================================================
    "Solve this logic puzzle step by step: A farmer has to cross a river with a wolf, a goat, and a cabbage. The boat only holds the farmer and one item. If left alone, the wolf eats the goat, and the goat eats the cabbage. How does he get everything across safely?",
    
    "Calculate the derivative of f(x) = x^3 * ln(x). Show all the steps of the product rule.",

    # =========================================================
    # 3. FATTUALI E CHIT-CHAT (Low Complexity)
    # Atteso: Modello FLASH (Catturato dall'Euristica o Semantico)
    # =========================================================
    "What is the capital of Australia?",
    
    "Translate 'Good morning, how are you today?' into Japanese.",
    
    "Hi AI, how is your day going? Just saying hello!",

    # =========================================================
    # 4. TASKS INTERMEDI / SPIEGAZIONI (Medium-Low Complexity)
    # Atteso: Modello FLASH (Gestito dal Semantic Router)
    # =========================================================
    "Write a polite, 3-sentence email to my boss asking for next Friday off because I have a family event.",
    
    "Give me a quick and easy recipe for a vegetarian pasta dish that takes under 20 minutes to make."
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