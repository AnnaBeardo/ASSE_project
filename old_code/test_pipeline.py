import time
from llm_client import LLMHandler
from metrics_tracker import MetricsTracker

def run_tests():
    llm = LLMHandler()
    tracker = MetricsTracker()

    # Test dataset
    test_prompts = [
        # {
        #     "id": 1,
        #     "complexity": "low",
        #     "text": "Hello, good morning! Could you please tell me what the capital of France is? Thank you very much!"
        # },
        # {
        #     "id": 2,
        #     "complexity": "low", # Usiamo flash per il codice semplice
        #     "text": "Hi there, I was wondering if you could help me. I need to know how to write a Python function to reverse a string. Kindly provide the code. Thanks!"
        # },
        # {
        #     "id": 3,
        #     "complexity": "high", # Usiamo pro per logica complessa
        #     "text": "Good evening. I would like you to explain the differences between classical inheritance and prototypal inheritance, and please provide a brief JSON summarizing the pros and cons."
        # },
        {
            "id": 1,
            "category": "Factual Q&A",
            "complexity": "low",
            "text": "Hello, good morning! Could you please tell me what the capital of France is? Thank you very much!"
        },
        {
            "id": 2,
            "category": "Basic Code",
            "complexity": "low",
            "text": "Hi there, I was wondering if you could help me. I need to know how to write a Python function to reverse a string. Kindly provide the code. Thanks!"
        },
        {
            "id": 3,
            "category": "Simple SQL",
            "complexity": "low",
            "text": "Hey! Could you please write a simple SQL query to select all users older than 18 from the 'customers' table? Thank you!"
        },
        
        # --- HIGH COMPLEXITY (Destinati a Pro) ---
        {
            "id": 4,
            "category": "Code Refactoring & Explanation",
            "complexity": "high",
            "text": "Good evening. I would like you to explain the time complexity of the bubble sort algorithm. Also, kindly provide a Python implementation and tell me why it is generally considered inefficient for large datasets."
        },
        {
            "id": 5,
            "category": "Logical Reasoning",
            "complexity": "high",
            "text": "Good morning. I need you to solve this logic puzzle: If all bloops are razzies and some razzies are lazzies, are all bloops definitely lazzies? Explain your reasoning step by step, please."
        },
        {
            "id": 6,
            "category": "System Architecture & Formatting",
            "complexity": "high",
            "text": "Hello! I am building a web app and I need to know how to structure my database. Please provide a JSON schema for a 'User' profile, including fields for name, email, and a list of roles. I would like you to ensure it strictly follows the OpenAPI 3.0 specification."
        }
    ]

    for test in test_prompts:
        print(f"\n--- ESECUZIONE TEST {test['id']} ({test['complexity'].upper()}) ---")
        
        start_time = time.time()
        
        # Invoking LLMHandler with prompt optimization enabled
        result = llm.invoke(test["text"], complexity=test["complexity"], enable_caveman=True)
        
        latency = time.time() - start_time
        
        # Recording metrics
        tracker.log_call(
            model_name="gemma-4-31b-it",
            complexity=test["complexity"],
            prompt=result["optimized_prompt_text"],
            response=result["response"],
            latency=latency
        )

        print(f"Originale: {test['text']}")
        print(f"Compresso: {result['optimized_prompt_text']}")
        print(f"Token Risparmiati: {result['compression_stats']['saved_tokens']}")
        print(f"Latenza: {latency:.2f}s")

    tracker.save_to_csv()

if __name__ == "__main__":
    run_tests()