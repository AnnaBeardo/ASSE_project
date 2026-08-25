"""
Test esteso per la pipeline di routing (heuristic + semantic).

Rispetto a test_routing.py, questo file:
- si concentra sui casi "ambigui" pensati per stressare i bug discussi:
  * fast-track "low" troppo aggressivo su word_count corto senza keyword
  * assenza di format_keywords (schema, architecture, step by step, ecc.)
  * corretta propagazione di route_logic quando interviene il semantic router
- stampa, per ogni prompt, la decisione dell'euristica PRIMA dell'eventuale
  fallback semantico, e le similarity score (flash_sim / pro_sim) quando
  il semantic router viene interpellato, per capire se il problema è
  nell'euristica o nella soglia/embedding del semantic router.

Lanciare da qualsiasi cartella con:
    python test_routing_extended.py
"""

import time
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Aggiunge la cartella 'src' al path di Python
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from llm_client import LLMHandler, PRO_MODEL, FLASH_MODEL


# Ogni caso: (prompt, expected_model, expected_heuristic_decision, note)
# expected_heuristic_decision è cosa DOVREBBE dire heuristic_route() da solo,
# prima di un eventuale passaggio al semantic router ("high" / "low" / "uncertain").
TEST_DATA = [
    # --- Format keywords: dovrebbero alzare strong_signals anche senza reasoning/math ---
    ("Design a microservices architecture and output the OpenAPI 3.0 schema.",
     PRO_MODEL, "high", "architecture+schema devono contare come segnali forti"),

    ("Give me the JSON schema for a user profile with nested address fields.",
     PRO_MODEL, "high", "schema+json, output strutturato complesso"),

    ("Draw a flowchart describing the checkout process of an e-commerce site.",
     PRO_MODEL, "high", "flowchart = format keyword"),

    ("Explain why the sky is blue step by step.",
     PRO_MODEL, "uncertain", "step by step da solo potrebbe non bastare per 2 segnali -> semantic decide"),

    # --- Fast-track "low" troppo aggressivo: prompt corti ma NON banali ---
    ("Summarize the trade-offs between REST and GraphQL.",
     PRO_MODEL, "uncertain", "corto, nessuna keyword esplicita, ma concettualmente denso"),

    ("Outline the pros and cons of remote work.",
     FLASH_MODEL, "uncertain", "corto, ambiguo, lasciare decidere al semantic router"),

    # --- Casi davvero banali: devono restare "low" anche con soglia più stretta ---
    ("Hi, how are you?", FLASH_MODEL, "low", "saluto puro"),
    ("Say hello", FLASH_MODEL, "low", "banale"),
    ("What is the capital of Italy?", FLASH_MODEL, "low", "domanda fattuale diretta"),
    ("Che ore sono?", FLASH_MODEL, "low", "domanda banale in italiano"),

    # --- Casi PRO evidenti (early exit su codice/query lunghe) ---
    ("```python\ndef calculate_fibonacci(n):\n    pass\n```",
     PRO_MODEL, "high", "code block esplicito"),
    ("Write a SQL query: SELECT * FROM users WHERE age > 30",
     PRO_MODEL, "high", "pattern SQL esplicito"),

    # --- Casi FLASH evidenti ma leggermente più lunghi (non devono diventare high per errore) ---
    ("Give me a recipe for a chocolate cake.",
     FLASH_MODEL, "low", "richiesta semplice, nessun reasoning/math/format"),
    ("Write a brief email to my boss asking for vacation days.",
     FLASH_MODEL, "low", "task comune, non richiede modello potente"),

    # --- Math/reasoning combinati: devono dare strong_signals >= 2 ---
    ("Solve this equation step by step and explain your reasoning: 3x + 5 = 20.",
     PRO_MODEL, "high", "math + reasoning + step by step -> 2+ segnali"),
]


def debug_semantic_scores(handler: LLMHandler, prompt: str):
    """Calcola manualmente flash_sim/pro_sim per diagnosticare la soglia del semantic router."""
    router = handler.semantic_router
    if not router.embedder:
        return None, None
    prompt_embedding = router.embedder.encode([prompt])
    flash_sim = float(np.max(cosine_similarity(prompt_embedding, router.flash_embeddings)))
    pro_sim = float(np.max(cosine_similarity(prompt_embedding, router.pro_embeddings)))
    return flash_sim, pro_sim


def run_extended_tests():
    print("Inizializzazione della Pipeline LLM (Caricamento Embedder e Modelli)...")
    start_init = time.time()

    handler = LLMHandler(temperature=0.0)

    print(f"Inizializzazione completata in {time.time() - start_init:.2f} secondi.")
    print("-" * 70)

    correct_final = 0
    correct_heuristic_stage = 0
    routing_methods = []

    for i, (prompt, expected_model, expected_heuristic, note) in enumerate(TEST_DATA, 1):
        print(f"\nTest {i}/{len(TEST_DATA)}")
        print(f"Prompt: '{prompt}'")
        print(f"Nota: {note}")

        # --- Stage 1: decisione euristica pura, PRIMA del fallback semantico ---
        heuristic_decision = handler.heuristic_router.heuristic_route(prompt)
        heuristic_ok = heuristic_decision == expected_heuristic
        correct_heuristic_stage += int(heuristic_ok)

        stage_icon = "✅" if heuristic_ok else "⚠️"
        print(f"{stage_icon} Euristica pura: '{heuristic_decision}' (atteso: '{expected_heuristic}')")

        # Se il caso atteso è "uncertain", mostriamo anche le similarity score
        if heuristic_decision == "uncertain" or expected_heuristic == "uncertain":
            flash_sim, pro_sim = debug_semantic_scores(handler, prompt)
            if flash_sim is not None:
                margin = pro_sim - flash_sim
                print(f"   ↳ Semantic scores: flash_sim={flash_sim:.3f} | pro_sim={pro_sim:.3f} "
                    f"| margin={margin:.3f} | margin_threshold={handler.semantic_router.margin_threshold}")

        # --- Stage 2: risultato finale della pipeline completa ---
        start_time = time.time()
        try:
            result = handler.invoke(prompt, enable_caveman=False)
            actual_model = result.get("routed_to")
            route_logic = result.get("route_logic")
            latency = time.time() - start_time

            final_ok = actual_model == expected_model
            correct_final += int(final_ok)
            final_icon = "✅" if final_ok else "❌"

            print(f"{final_icon} Risultato finale: {actual_model} | Logica: {route_logic} "
                  f"| Atteso: {expected_model} | Tempo: {latency:.2f}s")

            routing_methods.append(route_logic)

        except Exception as e:
            print(f"❌ ERRORE DI ESECUZIONE: {str(e)}")

    # --- Report finale ---
    n = len(TEST_DATA)
    final_accuracy = (correct_final / n) * 100
    heuristic_accuracy = (correct_heuristic_stage / n) * 100
    logic_counts = Counter(routing_methods)

    print("\n" + "=" * 70)
    print("📊 REPORT FINALE - TEST ESTESO")
    print("=" * 70)
    print(f"Accuratezza finale (modello scelto):      {final_accuracy:.1f}% ({correct_final}/{n})")
    print(f"Accuratezza euristica pura (stage 1):      {heuristic_accuracy:.1f}% ({correct_heuristic_stage}/{n})")
    print(f"Gestiti dall'Euristica (heuristic):        {logic_counts.get('heuristic', 0)}")
    print(f"Delegati al Semantic Router (semantic):    {logic_counts.get('semantic', 0)}")
    print("=" * 70)
    print("\nSuggerimento: se 'Delegati al Semantic Router' è ancora 0 nonostante ci siano")
    print("casi 'uncertain' attesi, il bug di propagazione in llm_client.py non è stato")
    print("ancora corretto. Se invece il semantic router viene chiamato ma sbaglia,")
    print("guarda le similarity score stampate sopra per tarare 'threshold' o arricchire")
    print("routing_prompts.json.")


if __name__ == "__main__":
    run_extended_tests()