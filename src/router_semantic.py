import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:

    def __init__(self, embedder, threshold: float = 0.45):
        # Recieves an embedder and a threshold for routing decisions
        self.embedder = embedder
        self.threshold = threshold      # it's for the cosine similarity
        
        # Utterances (examples)
        self.flash_examples = [
            "What is the capital of France?",
            "Write a Python function to reverse a string.",
            "Write a simple SQL query to select all users.",
            "How do I say hello in Spanish?",
            "Give me a quick recipe for pancakes."
        ]
        
        self.pro_examples = [
            "Explain the time complexity of the bubble sort algorithm and why it is inefficient.",
            "Solve this logic puzzle step by step.",
            "Provide a JSON schema strictly following the OpenAPI 3.0 specification.",
            "Compare classical inheritance and prototypal inheritance in depth.",
            "Analyze this system architecture and propose a microservices refactoring."
        ]
        
        # Pre compute embeddings for the examples to speed up routing decisions
        if self.embedder:
            self.flash_embeddings = self.embedder.encode(self.flash_examples)
            self.pro_embeddings = self.embedder.encode(self.pro_examples)
        else:
            self.flash_embeddings = None
            self.pro_embeddings = None

    def route(self, prompt: str) -> str:
        """
        Calculates the semantic distance and routes to 'high' (Pro) or 'low' (Flash).
        """
        # Fallback of security: if the embedder is not loaded or the prompt is empty, use Flash
        if not self.embedder or not prompt:
            return "low"

        # Transform the input prompt into a vector
        prompt_embedding = self.embedder.encode([prompt])
        
        # Calculate the Cosine Similarity between the prompt and the known routes
        flash_sim = np.max(cosine_similarity(prompt_embedding, self.flash_embeddings))
        pro_sim = np.max(cosine_similarity(prompt_embedding, self.pro_embeddings))
        
        # If the prompt is more similar to the Pro examples and exceeds the threshold, activate Pro
        if pro_sim > flash_sim and pro_sim >= self.threshold:
            return "high"
            
        # Default route
        return "low"