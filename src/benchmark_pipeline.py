import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from llm_client import LLMHandler, PRO_MODEL
from metrics_tracker import MetricsTracker 

# Prompts for testing, insert one or more
TEST_PROMPTS = [
    "Why is my React component re-rendering on every state update even though the props haven't changed? I'm passing an object as a prop.",
    "My Express auth middleware is letting expired JWT tokens through. The expiry check uses Date.now() compared to the token's exp field. What's wrong and how do I fix it?",
     "How do I set up a PostgreSQL connection pool in Node.js with proper timeout and error handling configuration?",
     "Explain the difference between git rebase and git merge. When should I use each one and what are the tradeoffs?",
     "Refactor this callback-based Node.js function to use async/await:\n\nfunction getUser(id, callback) {\n  db.query('SELECT * FROM users WHERE id = ?', [id], function(err, rows) {\n    if (err) return callback(err);\n    if (!rows.length) return callback(new Error('Not found'));\n    callback(null, rows[0]);\n  });\n}",
     "We have a monolithic Django app that's getting slow. The team is debating microservices. What are the key factors to consider before splitting up the monolith?",
     "Review this Express route handler for security issues:\n\napp.get('/api/users/:id', (req, res) => {\n  const query = `SELECT * FROM users WHERE id = ${req.params.id}`;\n  db.query(query).then(user => res.json(user));\n});",
      "Write a multi-stage Dockerfile for a Node.js TypeScript application that minimizes the final image size. The app uses npm and needs to compile TypeScript before running.",
      "My Node.js API endpoint that increments a counter in PostgreSQL sometimes returns the same value for concurrent requests. How do I fix this race condition?",
      "Implement a React error boundary component that catches render errors, shows a fallback UI with a retry button, and logs the error details." 
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