from pathlib import Path
import csv

from llm_client import LLMHandler, FLASH_MODEL, PRO_MODEL


BASE_DIR = Path(__file__).resolve().parent.parent

PROMPTS_FILE = BASE_DIR / "data" / "router_prompts.csv"
RESULTS_FILE = BASE_DIR / "data" / "router_results.csv"


def run_benchmark():
    if FLASH_MODEL == PRO_MODEL:
        raise ValueError(
            "FLASH_MODEL e PRO_MODEL sono uguali. "
            "Per il benchmark servono due modelli diversi."
        )

    llm = LLMHandler()

    with open(PROMPTS_FILE, "r", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        with open(
            RESULTS_FILE,
            "w",
            encoding="utf-8",
            newline=""
        ) as output_file:

            fieldnames = [
                "id",
                "prompt",
                "small_response",
                "large_response"
            ]

            writer = csv.DictWriter(
                output_file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for row in reader:
                prompt_id = row["id"]
                prompt = row["prompt"]

                print(f"Prompt {prompt_id}")

                small_response = llm.flash_model.invoke(prompt).content
                large_response = llm.pro_model.invoke(prompt).content

                writer.writerow({
                    "id": prompt_id,
                    "prompt": prompt,
                    "small_response": small_response,
                    "large_response": large_response
                })

    print(f"Risultati salvati in: {RESULTS_FILE}")


if __name__ == "__main__":
    run_benchmark()