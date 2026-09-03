from datasets import load_dataset
import json

# Since it is an utility script, it is not part of the main project codebase. It is used to generate a comprehensive set of keywords for the heuristic router.

def create_routing_dataset(output_file="routing_prompts.json", samples_per_class=100):
    print("Downloading real prompts, in English and not multiple choice...\n")
    flash_prompts = []
    pro_prompts = []

    # 1. LOW COMPLEXITY: Databricks Dolly 15k (Factual & Open QA)
    print("Recupero prompt LOW da 'databricks/databricks-dolly-15k'...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        for row in dolly:
            # Filtering only open QA prompts with reasonable length and no strange line breaks
            if row.get("category") == "open_qa":
                prompt = row.get("instruction", "").strip()
                if 20 < len(prompt) < 150 and "\n" not in prompt:
                    flash_prompts.append(prompt)
                
                if len(flash_prompts) >= samples_per_class:
                    break
    except Exception as e:
        print(f"Error in the download of Dolly: {e}")

    # 2. HIGH COMPLEXITY: Code Alpaca (Logic, Algorithms, Refactoring)
    print("Recupero prompt HIGH da 'iamtarun/python_code_instructions_18k_alpaca'...")
    try:
        code_alpaca = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for row in code_alpaca:
            prompt = row.get("instruction", "").strip()
            # Filtering for prompts that require code generation or structured logic
            if len(prompt) > 40:
                pro_prompts.append(prompt)
                
            if len(pro_prompts) >= samples_per_class:
                break
    except Exception as e:
        print(f"Error in the download of Code Alpaca: {e}")

    # Saving the file
    data = {
        "flash_examples": flash_prompts,
        "pro_examples": pro_prompts
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ File '{output_file}' generated with success!")
    print(f" - Prompt LOW (Flash): {len(flash_prompts)}")
    if flash_prompts:
        print(f"   Example: \"{flash_prompts[0]}\"")
        
    print(f" - Prompt HIGH (Pro): {len(pro_prompts)}")
    if pro_prompts:
        print(f"   Example: \"{pro_prompts[0]}\"")

if __name__ == "__main__":
    create_routing_dataset()