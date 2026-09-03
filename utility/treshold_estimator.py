"""
Calibrazione del margin_threshold per SemanticRouter.

Idea: invece di scegliere a caso una soglia assoluta o un margine,
usiamo gli esempi già presenti in routing_prompts.json come "ground truth"
e testiamo, in leave-one-out, quale soglia sul margine (pro_sim - flash_sim)
separa meglio flash_examples da pro_examples.

Leave-one-out: per ogni esempio, lo togliamo temporaneamente dal proprio pool
prima di calcolare le similarity, cosi' non "bariamo" confrontando una frase
con se stessa (che darebbe sempre similarity 1.0 e falserebbe la calibrazione).

Output:
- statistiche (mean/std/min/max) del margine per classe (flash vs pro)
- sweep di soglie con accuracy, precision, recall, F1 per ciascuna
- soglia consigliata (miglior F1) e soglia alternativa (miglior accuracy)

Lanciare da qualsiasi cartella con:
    python calibrate_semantic_threshold.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# adds the src folder to the path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from prompt_compressor import PromptCompressor


def load_examples(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    flash_examples = data.get("flash_examples", [])
    pro_examples = data.get("pro_examples", [])
    return flash_examples, pro_examples


def leave_one_out_margins(embedder, flash_examples, pro_examples):
    """
    For each example, calculate the margin (pro_sim - flash_sim) in a leave-one-out fashion.

    Returns two lists of margins: those calculated on flash examples
    (which should be negative, i.e., more similar to flash) and those
    calculated on pro examples (which should be positive).
    """
    print("Calculating embeddings...")
    flash_embeddings = embedder.encode(flash_examples)
    pro_embeddings = embedder.encode(pro_examples)

    flash_margins = []
    for i in range(len(flash_examples)):
        query = flash_embeddings[i:i + 1]
        # Leave-one-out: exclude the current example from the pool
        pool_flash = np.delete(flash_embeddings, i, axis=0)

        flash_sim = np.max(cosine_similarity(query, pool_flash))
        pro_sim = np.max(cosine_similarity(query, pro_embeddings))
        flash_margins.append(pro_sim - flash_sim)

    pro_margins = []
    for i in range(len(pro_examples)):
        query = pro_embeddings[i:i + 1]
        pool_pro = np.delete(pro_embeddings, i, axis=0)

        flash_sim = np.max(cosine_similarity(query, flash_embeddings))
        pro_sim = np.max(cosine_similarity(query, pool_pro))
        pro_margins.append(pro_sim - flash_sim)

    return np.array(flash_margins), np.array(pro_margins)


def print_stats(name: str, margins: np.ndarray):
    print(f"{name}: n={len(margins)} | mean={margins.mean():.4f} | std={margins.std():.4f} "
          f"| min={margins.min():.4f} | max={margins.max():.4f} "
          f"| p25={np.percentile(margins, 25):.4f} | p75={np.percentile(margins, 75):.4f}")


def sweep_thresholds(flash_margins: np.ndarray, pro_margins: np.ndarray, lo=-0.3, hi=0.3, step=0.01):
    """
    For each candidate threshold, the classification rule is:
        margin >= threshold  ->  "high" (pro)
        margin <  threshold  ->  "low"  (flash)

    Calculates accuracy, precision, recall, F1 (positive class = "pro"/high).
    """
    thresholds = np.arange(lo, hi + step, step)
    results = []

    n_pro = len(pro_margins)
    n_flash = len(flash_margins)

    for t in thresholds:
        tp = int(np.sum(pro_margins >= t))          # pro examples correctly classified as high
        fn = n_pro - tp                               # pro examples incorrectly classified as low
        fp = int(np.sum(flash_margins >= t))          # flash examples incorrectly classified as high
        tn = n_flash - fp                             # flash examples correctly classified as low

        accuracy = (tp + tn) / (n_pro + n_flash)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": t, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    return results


def run_calibration():
    json_path = src_path / "routing_prompts.json"
    flash_examples, pro_examples = load_examples(json_path)
    print(f"Loaded {len(flash_examples)} Flash examples and {len(pro_examples)} Pro examples from '{json_path}'.")

    print("Loading embedder (same model used by SemanticRouter)...")
    compressor = PromptCompressor()
    embedder = compressor.model

    flash_margins, pro_margins = leave_one_out_margins(embedder, flash_examples, pro_examples)

    print("\n" + "=" * 70)
    print("📊 DISTRIBUTION OF MARGINS (pro_sim - flash_sim), leave-one-out")
    print("=" * 70)
    print_stats("Flash examples (expected: low/negative margin)", flash_margins)
    print_stats("Pro examples (expected: high/positive margin)", pro_margins)

    overlap = np.sum(flash_margins >= pro_margins.min()) if len(pro_margins) else 0
    print(f"\nFlash examples with margin >= the minimum of the Pro margins: {overlap} "
          f"(indicates how much the two distributions overlap)")

    print("\n" + "=" * 70)
    print("📈 SWEEP OF THRESHOLDS (margin_threshold)")
    print("=" * 70)
    print(f"{'thresh':>8} | {'acc':>6} | {'prec':>6} | {'rec':>6} | {'f1':>6} | tp/fp/tn/fn")
    print("-" * 70)

    results = sweep_thresholds(flash_margins, pro_margins)
    for r in results:
        # print only thresholds that are multiples of 0.02 (to avoid too many lines)
        if round(r["threshold"] * 100) % 2 == 0:
            print(f"{r['threshold']:8.3f} | {r['accuracy']:6.3f} | {r['precision']:6.3f} | "
                  f"{r['recall']:6.3f} | {r['f1']:6.3f} | "
                  f"{r['tp']:>2}/{r['fp']:>2}/{r['tn']:>2}/{r['fn']:>2}")

    best_f1 = max(results, key=lambda r: r["f1"])
    best_acc = max(results, key=lambda r: r["accuracy"])

    print("\n" + "=" * 70)
    print("✅ RECOMMENDATIONS")
    print("=" * 70)
    print(f"Threshold with best F1:       margin_threshold = {best_f1['threshold']:.3f} "
          f"(F1={best_f1['f1']:.3f}, accuracy={best_f1['accuracy']:.3f}, "
          f"precision={best_f1['precision']:.3f}, recall={best_f1['recall']:.3f})")
    print(f"Threshold with best accuracy: margin_threshold = {best_acc['threshold']:.3f} "
          f"(accuracy={best_acc['accuracy']:.3f}, F1={best_acc['f1']:.3f})")

    midpoint = (flash_margins.mean() + pro_margins.mean()) / 2
    print(f"Midpoint between the two means (simple heuristic): margin_threshold = {midpoint:.3f}")

    print("\nNote: if the two distributions overlap significantly (low max F1, e.g., < 0.7),")
    print("the issue is not just the threshold: this indicates that the embedder and/or examples in")
    print("routing_prompts.json do not adequately distinguish the two concepts 'simple' vs 'complex'.")
    print("In such cases, it is advisable to enrich/clean the JSON with more representative examples")
    print("before blindly trusting the threshold found here.")
    print("=" * 70)


if __name__ == "__main__":
    run_calibration()