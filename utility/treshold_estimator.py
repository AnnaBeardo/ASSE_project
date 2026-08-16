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

# Aggiunge la cartella 'src' al path di Python
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
    Calcola, per ogni esempio, il margine (pro_sim - flash_sim) usando
    leave-one-out: l'esempio stesso viene escluso dal proprio pool prima
    del calcolo della similarity, per non confrontarlo con se stesso.

    Ritorna due liste di margini: quelli calcolati sugli esempi flash
    (dovrebbero essere negativi, cioe' piu' simili a flash) e quelli
    calcolati sugli esempi pro (dovrebbero essere positivi).
    """
    print("Calcolo embedding per tutti gli esempi (puo' richiedere qualche secondo)...")
    flash_embeddings = embedder.encode(flash_examples)
    pro_embeddings = embedder.encode(pro_examples)

    flash_margins = []
    for i in range(len(flash_examples)):
        query = flash_embeddings[i:i + 1]
        # Leave-one-out: escludi l'esempio i-esimo dal pool flash
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
    Per ogni soglia candidata, la regola di classificazione e':
        margin >= threshold  ->  "high" (pro)
        margin <  threshold  ->  "low"  (flash)

    Calcola accuracy, precision, recall, F1 (classe positiva = "pro"/high).
    """
    thresholds = np.arange(lo, hi + step, step)
    results = []

    n_pro = len(pro_margins)
    n_flash = len(flash_margins)

    for t in thresholds:
        tp = int(np.sum(pro_margins >= t))          # pro classificati correttamente come high
        fn = n_pro - tp                               # pro classificati erroneamente come low
        fp = int(np.sum(flash_margins >= t))          # flash classificati erroneamente come high
        tn = n_flash - fp                             # flash classificati correttamente come low

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
    print(f"Caricati {len(flash_examples)} esempi Flash e {len(pro_examples)} esempi Pro da '{json_path}'.")

    print("Caricamento embedder (stesso modello usato da SemanticRouter)...")
    compressor = PromptCompressor()
    embedder = compressor.model

    flash_margins, pro_margins = leave_one_out_margins(embedder, flash_examples, pro_examples)

    print("\n" + "=" * 70)
    print("📊 DISTRIBUZIONE DEI MARGINI (pro_sim - flash_sim), leave-one-out")
    print("=" * 70)
    print_stats("Esempi FLASH (atteso: margine basso/negativo)", flash_margins)
    print_stats("Esempi PRO   (atteso: margine alto/positivo)", pro_margins)

    overlap = np.sum(flash_margins >= pro_margins.min()) if len(pro_margins) else 0
    print(f"\nEsempi flash con margine >= al minimo dei margini pro: {overlap} "
          f"(indica quanto le due distribuzioni si sovrappongono)")

    print("\n" + "=" * 70)
    print("📈 SWEEP DELLE SOGLIE (margin_threshold)")
    print("=" * 70)
    print(f"{'thresh':>8} | {'acc':>6} | {'prec':>6} | {'rec':>6} | {'f1':>6} | tp/fp/tn/fn")
    print("-" * 70)

    results = sweep_thresholds(flash_margins, pro_margins)
    for r in results:
        # stampiamo solo ogni 2 righe per non intasare l'output
        if round(r["threshold"] * 100) % 2 == 0:
            print(f"{r['threshold']:8.3f} | {r['accuracy']:6.3f} | {r['precision']:6.3f} | "
                  f"{r['recall']:6.3f} | {r['f1']:6.3f} | "
                  f"{r['tp']:>2}/{r['fp']:>2}/{r['tn']:>2}/{r['fn']:>2}")

    best_f1 = max(results, key=lambda r: r["f1"])
    best_acc = max(results, key=lambda r: r["accuracy"])

    print("\n" + "=" * 70)
    print("✅ RACCOMANDAZIONI")
    print("=" * 70)
    print(f"Soglia con miglior F1:       margin_threshold = {best_f1['threshold']:.3f} "
          f"(F1={best_f1['f1']:.3f}, accuracy={best_f1['accuracy']:.3f}, "
          f"precision={best_f1['precision']:.3f}, recall={best_f1['recall']:.3f})")
    print(f"Soglia con miglior accuracy: margin_threshold = {best_acc['threshold']:.3f} "
          f"(accuracy={best_acc['accuracy']:.3f}, F1={best_acc['f1']:.3f})")

    midpoint = (flash_margins.mean() + pro_margins.mean()) / 2
    print(f"Punto medio tra le due medie (euristica semplice): margin_threshold = {midpoint:.3f}")

    print("\nNota: se le due distribuzioni si sovrappongono molto (F1 max basso, es. < 0.7),")
    print("il problema non e' solo la soglia: significa che l'embedder e/o gli esempi in")
    print("routing_prompts.json non separano bene i due concetti 'semplice' vs 'complesso'.")
    print("In quel caso conviene arricchire/ripulire il JSON con esempi piu' rappresentativi")
    print("prima di fidarsi ciecamente della soglia trovata qui.")
    print("=" * 70)


if __name__ == "__main__":
    run_calibration()