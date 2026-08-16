import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class SemanticRouter:

    def __init__(self, embedder, margin_threshold: float = 0.05, json_path: str = None):
        if json_path is None:
            json_path = os.path.join(_THIS_DIR, "routing_prompts.json")

        self.embedder = embedder
        self.margin_threshold = margin_threshold
        
        # Loading prompts from JSON file or fallback if not found
        self.flash_examples, self.pro_examples = self._load_utterances(json_path)
        
        # Precompute embeddings for the loaded prompts if an embedder is provided
        if self.embedder:
            self.flash_embeddings = self.embedder.encode(self.flash_examples)
            self.pro_embeddings = self.embedder.encode(self.pro_examples)
        else:
            self.flash_embeddings = None
            self.pro_embeddings = None

    def _load_utterances(self, json_path: str) -> tuple[list[str], list[str]]:
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    flash_examples = data.get("flash_examples", [])
                    pro_examples = data.get("pro_examples", [])
                    print(f"[SemanticRouter] Utterances from '{json_path}': "
                          f"{len(flash_examples)} Flash, {len(pro_examples)} Pro.")
                    return flash_examples, pro_examples
            except Exception as e:
                print(f"[SemanticRouter] Could not read {json_path}: {e}")

        #if the file is not found or an error occurs, return empty lists     
        raise FileNotFoundError(f"File not found: {json_path}")

    def route(self, prompt: str) -> str:
        if not self.embedder or not prompt:
            return "low"
        prompt_embedding = self.embedder.encode([prompt])
        flash_sim = np.max(cosine_similarity(prompt_embedding, self.flash_embeddings))
        pro_sim = np.max(cosine_similarity(prompt_embedding, self.pro_embeddings))
        return "high" if (pro_sim - flash_sim) >= self.margin_threshold else "low"