import time
import tiktoken
import pandas as pd
import os
from datetime import datetime

def generate_metrics_filename(base_dir="data", prefix="metrics", run_name=""):
    """Generate a unique filename based on the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_suffix = f"_{run_name}" if run_name else ""
    filename = f"{prefix}_{timestamp}{name_suffix}.csv"
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
        return len(self.tokenizer.encode(text))

    def calculate_cost(self, model_name: str, in_tokens: int, out_tokens: int) -> float:
        """Compute the cost based on the model and token counts."""

        #TODO: Cambiare calcoli in base ai modelli che scelgiamo di usare
        if "flash" in model_name.lower():
            rate_in, rate_out = 0.075, 0.30  # Costi Gemini 1.5 Flash
        else:
            rate_in, rate_out = 3.50, 10.50  # Costi Gemini 1.5 Pro

        return (in_tokens * rate_in / 1_000_000) + (out_tokens * rate_out / 1_000_000)

    def log_call(self, model_name: str, complexity: str, prompt: str, response: str, latency: float):
        """Record metrics for a single API call."""
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
            "cost_USD": round(cost, 6)
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

# Independent test for the MetricsTracker class
if __name__ == "__main__":
    tracker = MetricsTracker()
    
    start_time = time.time()
    time.sleep(0.5)  # Simulate a delay for testing latency
    end_time = time.time()
    
    tracker.log_call(
        model_name="gemini-flash-latest",
        complexity="low",
        prompt="Qual è la capitale dell'Italia?",
        response="Roma.",
        latency=end_time - start_time
    )
    
    tracker.save_to_csv()