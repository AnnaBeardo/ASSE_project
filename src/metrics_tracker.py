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
    COST_REGISTRY = {
        "gemini-3.1-flash-lite": {"input": 0.075, "output": 0.30},
        "gemma-4-31b-it":        {"input": 0.20,  "output": 0.20},
        "gemini-1.5-pro":        {"input": 3.50,  "output": 10.50},
        "default":               {"input": 0.0,   "output": 0.0}
    }

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

    def normalize_response(self, response) -> str:
        #Extract only the user-visible textual response. LangChain models may return structured content containing thinking/reasoning blocks and text blocks. Metrics should compare
        #the actual answer text consistently between baseline and pipeline.
        
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, list):
            text_parts = []

            for block in response:
                if isinstance(block, str):
                    text_parts.append(block)

                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text", "")

                        if text:
                            text_parts.append(str(text))

            if text_parts:
                return "\n".join(text_parts)

        try:
            text = response.text

            if text:
                return str(text)

        except (AttributeError, TypeError):
            pass

        return str(response)

    """ def calculate_cost(self, model_name: str, in_tokens: int, out_tokens: int) -> float:
        model_lower = model_name.lower()
        if model_lower == "error":
            return 0.0

        #TODO: Lavorare ancora su questa parte, capire un modo definitivo per affrontare la questione dei costi

        if "flash" in model_lower:
            model_lower = "gemini-3.1-flash-lite"  
        elif "gemma" in model_lower:
            model_lower = "gemma-4-31b-it"   

        try:
            # Calcola esattamente sulla base del listino ufficiale
            total_cost = calculate_cost_by_tokens(
                model_name=model_lower, 
                prompt_tokens=in_tokens, 
                completion_tokens=out_tokens
            )
            return float(total_cost)
        except Exception as e:
            print(f"Attenzione: Impossibile calcolare i costi per {model_name} in tokencost. Ritorno 0.0")
            return 0.0"""
    
    def calculate_cost(self, model_name: str, in_tokens: int, out_tokens: int) -> float:
        """Compute the cost based on the model and token counts using internal registry."""
        model_lower = model_name.lower()
        if model_lower == "error":
            return 0.0

        # Manteniamo la tua logica di normalizzazione dei nomi
        if "flash" in model_lower:
            model_lower = "gemini-3.1-flash-lite"  
        elif "gemma" in model_lower:
            model_lower = "gemma-4-31b-it"   

        # Recupera le tariffe dal nostro registro (usa default se non trova il modello)
        rates = self.COST_REGISTRY.get(model_lower, self.COST_REGISTRY["default"])

        if rates == self.COST_REGISTRY["default"]:
            print(f"Attenzione: Modello {model_lower} non trovato nel COST_REGISTRY. Costo 0.0")

        # Calcolo esatto per milione di token
        input_cost = (in_tokens / 1_000_000) * rates["input"]
        output_cost = (out_tokens / 1_000_000) * rates["output"]
        
        return input_cost + output_cost
    
    def log_call(self, model_name: str, complexity: str, prompt: str, response: str, latency: float, in_tokens: int = None, out_tokens: int = None):
        """Record metrics for a single API call."""
        if not prompt: prompt = ""
        if not isinstance(prompt, str): prompt = str(prompt)

        # response = self.normalize_response(response)
        if not response: response = ""
        if not isinstance(response, str): response = str(response)

        actual_in_tokens = in_tokens if in_tokens is not None else self.count_tokens(prompt)
        actual_out_tokens = out_tokens if out_tokens is not None else self.count_tokens(response)
        
        cost = self.calculate_cost(model_name, actual_in_tokens, actual_out_tokens)

        record = {
            "model": model_name,
            "complexity": complexity,
            "latency_sec": round(latency, 4),
            "in_tokens": actual_in_tokens,
            "out_tokens": actual_out_tokens,
            "total_tokens": actual_in_tokens + actual_out_tokens,
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