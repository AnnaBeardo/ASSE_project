from datasets import load_dataset
import json

def create_routing_dataset(output_file="routing_prompts.json", samples_per_class=100):
    print("Scaricando prompt reali, in inglese e non a risposta multipla...\n")
    flash_prompts = []
    pro_prompts = []

    # 1. LOW COMPLEXITY: Databricks Dolly 15k (Factual & Open QA)
    print("Recupero prompt LOW da 'databricks/databricks-dolly-15k'...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for row in dolly:
            # Filtriamo solo le domande di cultura generale / fattuali
            if row.get("category") == "open_qa":
                prompt = row.get("instruction", "").strip()
                # Prendiamo prompt puliti, lunghi il giusto e senza ritorni a capo strani
                if 20 < len(prompt) < 150 and "\n" not in prompt:
                    flash_prompts.append(prompt)
                
                if len(flash_prompts) >= samples_per_class:
                    break
    except Exception as e:
        print(f"Errore nel download di Dolly: {e}")

    # 2. HIGH COMPLEXITY: Code Alpaca (Logic, Algorithms, Refactoring)
    print("Recupero prompt HIGH da 'iamtarun/python_code_instructions_18k_alpaca'...")
    try:
        code_alpaca = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for row in code_alpaca:
            prompt = row.get("instruction", "").strip()
            # Prendiamo prompt che richiedono generazione di codice o logica strutturata
            if len(prompt) > 40:
                pro_prompts.append(prompt)
                
            if len(pro_prompts) >= samples_per_class:
                break
    except Exception as e:
        print(f"Errore nel download di Code Alpaca: {e}")

    # Salvataggio del file
    data = {
        "flash_examples": flash_prompts,
        "pro_examples": pro_prompts
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ File '{output_file}' generato con successo!")
    print(f" - Prompt LOW (Flash): {len(flash_prompts)}")
    if flash_prompts:
        print(f"   Esempio: \"{flash_prompts[0]}\"")
        
    print(f" - Prompt HIGH (Pro): {len(pro_prompts)}")
    if pro_prompts:
        print(f"   Esempio: \"{pro_prompts[0]}\"")

if __name__ == "__main__":
    create_routing_dataset()