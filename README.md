# LLM Optimization Pipeline: Dynamic Routing & Prompt Compression

This repository contains the source code and benchmarking suite for a Master's degree thesis project focused on optimizing Large Language Model (LLM) architectures. 

The project introduces an automated pipeline that drastically reduces token consumption and inference latency without sacrificing output quality. It achieves this by combining a "Caveman-style" prompt compression strategy with a two-layer dynamic router (heuristic and semantic), all orchestrated via **LangChain**.

## Key Features

* **LangChain Orchestration:** Utilizes LangChain Expression Language (LCEL) to standardize the workflow, decoupling the application logic from provider-specific APIs.
* **Dynamic Model Routing:** A two-layer router autonomously evaluates prompt complexity, delegating simple tasks to a low-latency model (`gemini-3.1-flash-lite`) and complex tasks to a flagship model (`gemma-4-31b-it`).
* **Output Compression ("Caveman" System Prompt):** Injects strict system directives to eliminate conversational fluff, slashing output tokens by an average of 78%.
* **Automated Benchmarking:** Includes a robust evaluation script to compare the Baseline (raw model) against the Pipeline, tracking latency, token usage, and costs.

---

## Getting Started

Follow these instructions to set up the project and run the benchmark on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system. You will also need an active API key for Google Gemini (and/or any other specific provider used in the project).

### 2. Installation
Clone the repository and navigate into the project directory:

```bash
git clone <YOUR_REPO_URL_HERE>
cd <YOUR_PROJECT_FOLDER>
```

Create and activate a virtual environment to isolate the project dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Configuration (Environment Variables)
For security reasons, API keys are never hardcoded in the source files. The project uses the `python-dotenv` library to load secrets securely.

1. Create a new file named `.env` in the root directory of the project.
2. Open the `.env` file and add your API keys:

```text
# .env file
GOOGLE_API_KEY=your_actual_api_key_here
```
*(Note: The `.env` file is already included in the `.gitignore` to prevent accidental uploads to GitHub).*

### 4. Usage

To run the full evaluation suite and generate the metrics CSV file, execute the benchmarking script:

```bash
python src/benchmark_pipeline.py
```

The script will iterate through the configured dataset (`prompts/extended_prompts.py`), running both the Baseline and the Pipeline for each task. The execution metrics will be automatically saved in the `data/` folder as a timestamped CSV file (e.g., `metrics_benchmark_YYYY-MM-DD.csv`).

---

## Project Structure

* `src/` - Contains the core application logic.
  * `llm_client.py` - LLM initialization and LangChain setup.
  * `euristic_router.py` - Heuristic routing logic.
  * `router_semantic.py` - Semantic routing logic.
  * `benchmark_pipeline.py` - The main executable script for A/B testing.
  * `prompt_compressor.py` - Implements input prompt compression.
  * `routing_prompts.json` - Contains the routing prompts used by the semantic routers
  * `metrics_tracker.py` - Tracks and logs latency, token usage, and costs.
* `prompts/` - Contains the testing datasets.
  * `diverse_prompts.py` - The 30-prompt evaluation suite.
  * `caveman_prompts.py` - The prompts used in the official Caveman project for the benchmark.
* `utility/` - Utility functions
    * `download_prompts.py` - Downloads the prompts to extract the keywords from.
    * `euristic_keyword_finder.py` - Extracts keywords from the downloaded prompts to be used in the heuristic router.
    * `treshold_estimator.py` - Estimates the threshold for the semantic router
* `data/` - Output directory for generated CSV metric reports.
* `requirements.txt` - Python dependencies.
* `.gitignore` - Excludes virtual environments, caches, and secret files (`.env`).

## Results Summary
Empirical validation demonstrates that the pipeline reduces inference latency by up to **15x-20x** on simple queries and achieves a consistent **~70-80% reduction** in token consumption across a diverse set of 30 factual, logical, and coding tasks.