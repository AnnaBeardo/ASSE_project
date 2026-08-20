import time
from anyio import Path
import tiktoken
import pandas as pd
import os
from datetime import datetime

def generate_metrics_filename(base_dir="data", prefix="metrics", run_name=""):
    """Generate a unique filename based on the current timestamp."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    base_dir = os.path.join(project_root, "data")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    suffix = f"_{run_name}" if run_name else ""
    filename = f"{prefix}{suffix}_{timestamp}.csv"
    
    return os.path.join(base_dir, filename)


class MetricsTracker:
    def __init__(self, run_name="", log_file=None): 
        self.log_file = log_file or generate_metrics_filename(run_name=run_name)
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.records = []

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        print(f"File metriche inizializzato: {self.log_file}")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken's tokenizer."""
        if not text or not isinstance(text, str):
            text = str(text) if text is not None else ""
        return len(self.tokenizer.encode(text))

    def calculate_cost(self, model_name: str, in_tokens: int, out_tokens: int) -> float:
        """Compute the cost based on the model and token counts."""
        model_lower = model_name.lower()

        #TODO: Lavorare ancora su questa parte, capire un modo definitivo per affrontare la questione dei costi

        if "flash" in model_lower:
            # Cost estimation for Gemini Flash-Lite
            rate_in, rate_out = 0.075, 0.30  
        elif "gemma" in model_lower:
            # Cost estimation for an open-source model with a ~30B parameter count
            # If running locally for free, change these values to 0.0, 0.0
            rate_in, rate_out = 0.80, 0.80  
        else:
            # Fallback for commercial PRO models (e.g., Gemini Pro, GPT-4)
            rate_in, rate_out = 3.50, 10.50  

        return (in_tokens * rate_in / 1_000_000) + (out_tokens * rate_out / 1_000_000)

    def log_call(self, model_name: str, complexity: str, prompt: str, response: str, latency: float):
        """Record metrics for a single API call."""
        if not prompt: prompt = ""
        if not isinstance(prompt, str): prompt = str(prompt)

        if not response: response = ""
        if not isinstance(response, str): response = str(response)

        in_tokens = self.count_tokens(prompt)
        out_tokens = self.count_tokens(response)
        cost = self.calculate_cost(model_name, in_tokens, out_tokens)

        record = {
            "model": model_name,
            "complexity": complexity,
            "latency_sec": round(latency, 4),
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "cost_USD": f"{cost:.6f}",
            "prompt": prompt,
            "response": response,
        }
        self.records.append(record)
        return record

    def save_to_csv(self):
        """Save all recorded metrics to a CSV file."""
        df = pd.DataFrame(self.records)
        if os.path.exists(self.log_file):
            df.to_csv(self.log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.log_file, index=False)
        print(f"Metriche salvate in {self.log_file}")