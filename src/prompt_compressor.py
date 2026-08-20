import re
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Lazy import of sentence-transformers to avoid heavy dependencies if not needed
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class PromptCompressor:
    # This class implements a two-layer prompt compression system:
    # Layer 1: Rule-based cleaning (removes politeness, greetings, and unnecessary phrases)
    # Layer 2: Semantic compression (uses embeddings to retain only semantically relevant sentences)
    def __init__(self, encoding_name: str = "cl100k_base", semantic_threshold: float = 0.45):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.semantic_threshold = semantic_threshold
        
        # Initializing lightweight embedded model
        if HAS_ST:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.model = self.embedder  # Expose the embedder for external use 
        else:
            self.embedder = None
            print("Warning: sentence-transformers not installed. Semantic compression disabled.")

        self.courtesies_patterns = [
            r"(?i)\b(hello|hi|hey|good morning|good evening|good afternoon)\b",
            r"(?i)\b(thank you very much|thanks a lot|thank you|thanks|very much)\b",
            r"(?i)\b(could you|can you|would you|will you|please|kindly)\b",
            r"(?i)\b(i would like you to|i need you to|i was wondering if|i would like to know|i need to know)\b",
            r"(?i)\b(tell me|explain to me|help me with|provide me with|do you know)\b",
            r"(?i)\b(what is|what are|how do i|how to)\b"
        ]

        self.stopwords = {
            "the", "a", "an", "is", "are", "am", "was", "were", "be", "been",
            "i", "me", "my", "you", "your", "he", "him", "she", "her", "it",
            "what", "who", "where", "when", "why", "how", "do", "does", "did",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "there", "could", "would", "should", "can", "help", "know", "need"
        }

        self.compressor_system_prompt = (
            "You are an ultra-concise assistant. Answer the request directly and essential. "
            "ELIMINATE pleasantries, preambles, obvious explanations, and courtesy phrases. "
            "If code or JSON is requested, provide ONLY the code/JSON block without intro or outro."
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _rule_based_clean(self, text: str) -> str:
        """Syntatic cleaning"""
        cleaned = text

        for pattern in self.courtesies_patterns:
            cleaned = re.sub(pattern, "", cleaned)
            
        cleaned = re.sub(r'[,.!?]', ' ', cleaned)   # Removing punctuation
        
        # Removing stopwords
        words = cleaned.split()
        caveman_words = [
            word for word in words 
            if word.lower() not in self.stopwords
        ]
        
        return " ".join(caveman_words).strip()

    def _semantic_compress(self, text: str) -> str:
        """
        Layer 2: Semantic compression based on global intent.
        Evaluate how much each sentence contributes to the overall meaning of the prompt.
        """
        if not self.embedder or not text:
            return text

        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 3]
        
        # Avoid compressing if the prompt is already very short
        if len(sentences) <= 2:
            return text 

        
        global_embedding = self.embedder.encode([text]) # Computing global embedding for the entire prompt
        sentence_embeddings = self.embedder.encode(sentences) # Computing embeddings for each individual sentence

        similarities = cosine_similarity(sentence_embeddings, global_embedding).flatten()
        kept_sentences = []
        dynamic_threshold = min(self.semantic_threshold, 0.30)

        for idx, score in enumerate(similarities):
            is_boundary = (idx == 0 or idx == len(sentences) - 1)
            
            if is_boundary or score >= dynamic_threshold:
                kept_sentences.append(sentences[idx])

        return " ".join(kept_sentences)

    def optimize_prompt(self, original_prompt: str, enable_compression: bool = True) -> dict:
        if not enable_compression:
            return {
                "system_prompt": "",
                "user_prompt": original_prompt,
                "stats": {
                    "original_tokens": self.count_tokens(original_prompt),
                    "compressed_tokens": self.count_tokens(original_prompt),
                    "saved_tokens": 0
                }
            }

        semantically_compressed = self._semantic_compress(original_prompt)
        cleaned_text = self._rule_based_clean(semantically_compressed)

        orig_tokens = self.count_tokens(original_prompt)
        comp_tokens = self.count_tokens(cleaned_text)

        return {
            "system_prompt": self.compressor_system_prompt,
            "user_prompt": cleaned_text,
            "stats": {
                "original_tokens": orig_tokens,
                "compressed_tokens": comp_tokens,
                "saved_tokens": orig_tokens - comp_tokens
            }
        }