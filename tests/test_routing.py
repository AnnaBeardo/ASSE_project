import time
import sys
from collections import Counter
from pathlib import Path

# Aggiunge la cartella 'src' al path di Python
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from llm_client import LLMHandler, PRO_MODEL, FLASH_MODEL

# Dataset di test: Prompt -> Modello atteso
TEST_DATA = [
    # --- CASI PRO EVIDENTI (Attivano l'Early Exit dell'Euristica) ---
    ("```python\ndef calculate_fibonacci(n):\n    pass\n```", PRO_MODEL),
    ("Write a SQL query: SELECT * FROM users WHERE age > 30", PRO_MODEL),
    ("Design a microservices architecture and output the OpenAPI 3.0 schema.", PRO_MODEL),
    
    # --- CASI FLASH EVIDENTI (Attivano il Fast Track Low dell'Euristica) ---
    ("What is the capital of Italy?", FLASH_MODEL),
    ("Say hello", FLASH_MODEL),
    ("Che ore sono?", FLASH_MODEL),
    
    # --- CASI AMBIGUI / INTERMEDI (Passano a 'uncertain' e attivano il Semantic Router) ---
    ("Explain why the sky is blue step by step.", PRO_MODEL),
    ("Give me a recipe for a chocolate cake.", FLASH_MODEL),
    ("Write a brief email to my boss asking for vacation days.", FLASH_MODEL),
    ("Solve this logic puzzle: If a rooster lays an egg on a pitched roof...", PRO_MODEL)
]

def run_routing_tests():
    print("Inizializzazione della Pipeline LLM (Caricamento Embedder e Modelli)...")
    start_init = time.time()
    
    # Inizializza l'handler
    handler = LLMHandler(temperature=0.0)
    
    print(f"Inizializzazione completata in {time.time() - start_init:.2f} secondi.")
    print("-" * 60)
    
    correct_predictions = 0
    routing_methods = []
    
    for i, (prompt, expected_route) in enumerate(TEST_DATA, 1):
        print(f"\nTest {i}/{len(TEST_DATA)}")
        print(f"Prompt: '{prompt}'")
        
        start_time = time.time()
        
        try:
            # Esegue l'invocazione (disattivando caveman opzionalmente se vuoi velocizzare, qui usa i default)
            result = handler.invoke(prompt, enable_caveman=False)
            actual_route = result.get("routed_to")
            route_logic = result.get("route_logic") # 'heuristic' o 'semantic'
            
            latency = time.time() - start_time
            
            # Valutazione
            if actual_route == expected_route:
                print(f"✅ PASSED | Rotta: {actual_route} | Logica: {route_logic} | Tempo: {latency:.2f}s")
                correct_predictions += 1
            else:
                print(f"❌ FAILED | Atteso: {expected_route}, Ottenuto: {actual_route} | Logica: {route_logic}")
                
            routing_methods.append(route_logic)
                
        except Exception as e:
            print(f"❌ ERRORE DI ESECUZIONE: {str(e)}")

    # Report Finale
    accuracy = (correct_predictions / len(TEST_DATA)) * 100
    logic_counts = Counter(routing_methods)
    
    print("\n" + "="*60)
    print("📊 REPORT FINALE PIPELINE DI ROUTING")
    print("="*60)
    print(f"Accuratezza Totale: {accuracy:.1f}% ({correct_predictions}/{len(TEST_DATA)})")
    print(f"Gestiti dall'Euristica (Fast Track / Early Exit): {logic_counts.get('heuristic', 0)}")
    print(f"Delegati al Semantic Router (Uncertain): {logic_counts.get('semantic', 0)}")
    print("="*60)

if __name__ == "__main__":
    run_routing_tests()