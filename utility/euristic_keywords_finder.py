import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


# Since it is an utility script, it is not part of the main project codebase. It is used to generate a comprehensive set of keywords for the heuristic router.

def extract_keywords(texts: list, top_n: int = 50) -> list:
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 2), 
        max_df=0.85, 
        min_df=5
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    sorted_indices = avg_weights.argsort()[::-1]
    
    # List of keywords/phrases
    return [str(feature_names[i]) for i in sorted_indices[:top_n]]

def build_comprehensive_keywords():
    print("Loading dataset...")

    # 1. REASONING (Dolly 15k)
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    complex_categories = {"brainstorming", "summarization", "extract_information", "classification", "creative_writing"}
    reasoning_prompts = [row["instruction"] for row in dolly if row["category"] in complex_categories]

    # 2. CODING (CodeAlpaca 20k)
    code_alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    coding_prompts = [row["instruction"] for row in code_alpaca]

    # 3. MATH (GSM8K)
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    math_prompts = [row["question"] for row in gsm8k]

    print("Calculating TF-IDF weights (this may take a minute)...")
    
    reasoning_kws = extract_keywords(reasoning_prompts, top_n=60)
    coding_kws = extract_keywords(coding_prompts, top_n=60)
    math_kws = extract_keywords(math_prompts, top_n=40)

    # create the src directory if it doesn't exist and prepare the file
    output_file = Path("src/router_keywords.py")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# File generato automaticamente con TF-IDF sui dataset HF\n\n")
        f.write(f"REASONING_KEYWORDS = set({reasoning_kws})\n\n")
        f.write(f"CODING_KEYWORDS = set({coding_kws})\n\n")
        f.write(f"MATH_KEYWORDS = set({math_kws})\n")

    print(f"\nSaving completed! File created in: {output_file.resolve()}")

if __name__ == "__main__":
    build_comprehensive_keywords()