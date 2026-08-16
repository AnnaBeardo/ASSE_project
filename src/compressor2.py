import re
import tiktoken
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class PromptCompressor:
    """
    Two-layer Prompt Compression:
    - Layer 1: Semantic Context Filtering via local lightweight Embeddings (Local MiniLM)
    - Layer 2: Rule-based Syntactic & Courtesy Pruning
    - Output Steering: Caveman-style system prompt
    """
    def __init__(self, encoding_name: str = "cl100k_base", semantic_threshold: float = 0.35):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.semantic_threshold = semantic_threshold
        
        # Local Embedding Model (runs locally on CPU/GPU, zero API cost)
        if HAS_ST:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedder = None
            print("Warning: sentence-transformers not installed. Semantic compression disabled.")

        self.courtesies_patterns = [
            r"(?i)\b(hello|hi|hey|good morning|good evening|good afternoon)\b[,.]?",
            r"(?i)\b(thank you very much|thanks a lot|thank you|thanks|very much)\b[,.]?",
            r"(?i)\b(could you|can you|would you|will you|please|kindly)\b",
            r"(?i)\b(i would like you to|i need you to|i was wondering if|i would like to know|i need to know)\b",
            r"(?i)\b(tell me|explain to me|help me with|provide me with|do you know)\b"
        ]

        # Stopwords non distruttive per task di programmazione
        self.caveman_stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "you", "your", "he", "him", "she", "her", "it",
            "there", "could", "would", "should", "shall"
        }

        self.compressor_system_prompt = (
            "You are an ultra-concise assistant. Answer the request directly and essential. "
            "ELIMINATE pleasantries, preambles, obvious explanations, and courtesy phrases. "
            "If code or JSON is requested, provide ONLY the code/JSON block without intro or outro."
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _rule_based_clean(self, text: str) -> str:
        """Layer 2: Pulizia deterministica (Regex + Stopwords sicure)"""
        cleaned = text
        
        # 1. Rimozione convenevoli
        for pattern in self.courtesies_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        
        # 2. Normalizzazione spazi multipli
        cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()

        # 3. Filtraggio stopwords preservando blocchi di codice (backticks)
        if "```" not in cleaned:
            words = cleaned.split()
            cleaned_words = [w for w in words if w.lower() not in self.caveman_stopwords]
            cleaned = " ".join(cleaned_words)

        return cleaned.strip()

    def _semantic_compress(self, text: str) -> str:
        """
        Layer 1: Filtraggio semantico del contesto (Budget allocation).
        Preserva intatta l'istruzione primaria (ultima frase) e filtra le frasi di supporto.
        """
        if not self.embedder or not text or "```" in text:
            return text

        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 3]
        
        # Se il prompt è già conciso, non filtrarlo
        if len(sentences) <= 2:
            return text

        # L'ultima frase rappresenta comunemente la query/task esplicito
        target_query = sentences[-1]
        
        query_embedding = self.embedder.encode([target_query])
        sentence_embeddings = self.embedder.encode(sentences)

        similarities = cosine_similarity(sentence_embeddings, query_embedding).flatten()
        
        kept_sentences = []
        for idx, score in enumerate(similarities):
            # Budget Rule: La prima e l'ultima frase (query) hanno priorità
            is_critical = (idx == 0 or idx == len(sentences) - 1)
            
            if is_critical or score >= self.semantic_threshold:
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

        # Step 1: Filtro Semantico sul Contesto
        semantically_compressed = self._semantic_compress(original_prompt)

        # Step 2: Pulizia Sintattica e Stopwords
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