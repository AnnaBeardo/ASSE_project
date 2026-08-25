import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Configurazione dei path del progetto
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

load_dotenv(dotenv_path=project_root / ".env")

from llm_client import LLMHandler, PRO_MODEL, FLASH_MODEL
from metrics_tracker import MetricsTracker

# -------------------------------------------------------------------------
# SETUP TOKENIZZATORE (Google Generative AI o Fallback)
# -------------------------------------------------------------------------
try:
    import google.generativeai as genai
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        model = genai.GenerativeModel(FLASH_MODEL)
        return model.count_tokens(text).total_tokens

    _TOKENIZER_INFO = f"google.generativeai ({FLASH_MODEL} Native Tokenizer)"
except ImportError:
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    _TOKENIZER_INFO = "Fallback (1 token ~ 4 caratteri)"


# Tabella prezzi (USD per milione di token)
COST_PER_1M_TOKENS = {
    FLASH_MODEL: {"input": 0.075, "output": 0.30},
    PRO_MODEL:   {"input": 3.50, "output": 10.50},
}

# -------------------------------------------------------------------------
# PROMPT DI TEST (1 per ciascuna delle 15 categorie di test_prompt_types)
# -------------------------------------------------------------------------
TEST_CASES_ONE_PER_CATEGORY = [
    (
        "simple_factual",
        "What is the capital of Portugal?"
    ),
    (
        "simple_conceptual",
        "What is photosynthesis and why is it important for plants?"
    ),
    (
        "short_logical_reasoning",
        "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain why."
    ),
    (
        "mathematical_reasoning",
        "A car travels 120 km at 60 km/h and then another 120 km at 40 km/h. What is its average speed for the entire journey? Show the reasoning."
    ),
    (
        "simple_coding",
        "Write a Python function that takes two integers and returns the larger one."
    ),
    (
        "complex_coding",
        """A Node.js service processes jobs from a queue using several asynchronous workers.
Sometimes two workers process the same job because they read its status before either one
updates it. Explain the race condition and redesign the processing logic so that each job
can be processed by only one worker, assuming PostgreSQL is used as the database."""
    ),
    (
        "security",
        """Review this Express endpoint for security vulnerabilities and explain how to fix them:

app.get('/users/:id', async (req, res) => {
    const query = `SELECT * FROM users WHERE id = ${req.params.id}`;
    const result = await db.query(query);
    res.json(result.rows);
});"""
    ),
    (
        "strict_format",
        """Return information about Italy using exactly this JSON structure:
{
  "country": string,
  "capital": string,
  "currency": string
}
Return only valid JSON and no additional text."""
    ),
    (
        "long_but_easy",
        """I am preparing a short general-knowledge quiz for a group of people who have very
different backgrounds. Some are university students, some work in offices, some are retired,
and a few are still in high school. The quiz will be printed on paper and used during an
informal evening event. It is not part of an exam, no specialist knowledge is expected,
and nobody will be allowed to use a phone or search engine. I have already written several
questions about geography, history, science, sports, music, literature, food, and cinema.
I am trying to keep the questions easy enough that most participants can answer at least
half of them. I also want each answer to be short, ideally one or two words, because the
person reading the answers aloud should be able to move quickly from one question to the
next. The geography section currently includes questions about rivers, mountains,
continents, flags, oceans, and European countries. There is no trick involved in the next
question, no hidden assumption, and no need to discuss politics, history, population,
language, culture, or the European Union. I do not need an explanation of how capitals are
selected, a list of Portuguese cities, travel advice, demographic information, or
alternative historical capitals. I only need the ordinary present-day answer that would
appear in a basic school atlas or a general-knowledge quiz. Please keep the final answer
as short as possible, because I am going to copy it directly into the answer sheet used
by the quiz host. What is the capital of Portugal?"""
    ),
    (
        "short_but_complex",
        "Prove that the square root of 2 is irrational."
    ),
    (
        "irrelevant_information",
        """Yesterday I was reorganizing my desk while listening to music. I found an old
notebook from university and started reading some unrelated notes about databases.
My laptop battery was almost empty, so I moved to another room and made some coffee.
None of this is relevant to the actual task. Explain the difference between a primary
key and a foreign key in a relational database."""
    ),
    (
        "information_dense",
        """Write a Python function named filter_users that accepts a list of dictionaries.
Keep only users whose age is at least 18 and whose active field is True.
Sort the resulting users by age in descending order.
If two users have the same age, sort them alphabetically by name.
Do not modify the original list.
Return an empty list if the input list is empty."""
    ),
    (
        "negation",
        """Given the list [5, 2, 5, 3, 2, 1], remove duplicate values but do not sort
the list. Preserve the order of the first occurrence of every value."""
    ),
    (
        "borderline",
        "Design a solution for managing user sessions in a web application."
    ),
    (
        "adversarial_routing",
        "What does the word architecture mean?"
    )
]


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calcola il costo totale in USD."""
    rates = COST_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


def run_benchmark_and_token_savings():
    print(f"Tokenizer backend: {_TOKENIZER_INFO}")
    print("Inizializzazione LLMHandler e MetricsTracker...")
    handler = LLMHandler(temperature=0.0)
    tracker = MetricsTracker(run_name="benchmark_token_savings")

    results_summary = []
    route_logic_counts = defaultdict(int)
    model_counts = defaultdict(int)

    for i, (category, prompt) in enumerate(TEST_CASES_ONE_PER_CATEGORY, 1):
        print("\n" + "=" * 90)
        print(f"TEST {i}/{len(TEST_CASES_ONE_PER_CATEGORY)} | CATEGORIA: {category}")
        print("=" * 90)
        print(f"Prompt: '{prompt[:70]}{'...' if len(prompt) > 70 else ''}'")

        # ---------------------------------------------------------------------
        # 1. BASELINE (Chiamata diretta al modello Pro, senza compressione)
        # ---------------------------------------------------------------------
        print("\n   -> Esecuzione BASELINE (Chiamata diretta PRO)...")
        start_base = time.time()
        try:
            ai_msg = handler.pro_model.invoke(prompt)
            base_response = ai_msg.content
            base_latency = time.time() - start_base
            
            usage = getattr(ai_msg, "usage_metadata", None)
            base_in = usage.get("input_tokens") if usage else count_tokens(prompt)
            base_out = usage.get("output_tokens") if usage else count_tokens(base_response)
        except Exception as e:
            print(f"   [Errore Baseline]: {e}")
            base_response = f"ERROR: {e}"
            base_latency = 0.0
            base_in = count_tokens(prompt)
            base_out = 0

        base_cost = cost_of(PRO_MODEL, base_in, base_out)

        tracker.log_call(
            model_name=PRO_MODEL,
            complexity=f"BASELINE | {category}",
            prompt=prompt,
            response=base_response,
            latency=base_latency,
            in_tokens=base_in,
            out_tokens=base_out
        )

        # ---------------------------------------------------------------------
        # 2. PIPELINE (Routing dinamico + Compressione Caveman)
        # ---------------------------------------------------------------------
        print("   -> Esecuzione PIPELINE (Routing + Caveman)...")
        start_pipe = time.time()
        try:
            result = handler.invoke(prompt, enable_caveman=True)
            pipe_response = result.get("response", "")
            routed_to = result.get("routed_to", FLASH_MODEL)
            route_logic = result.get("route_logic", "unknown")
            optimized_prompt = result.get("optimized_prompt_text", prompt)
            pipe_latency = time.time() - start_pipe

            pipe_in = result.get("api_in_tokens") or count_tokens(optimized_prompt)
            pipe_out = result.get("api_out_tokens") or count_tokens(pipe_response)
        except Exception as e:
            print(f"   [Errore Pipeline]: {e}")
            pipe_response = f"ERROR: {e}"
            routed_to = "ERROR"
            route_logic = "ERROR"
            optimized_prompt = prompt
            pipe_latency = 0.0
            pipe_in = base_in
            pipe_out = 0

        pipe_cost = cost_of(routed_to, pipe_in, pipe_out)

        tracker.log_call(
            model_name=routed_to,
            complexity=f"PIPELINE | {category} | {route_logic}",
            prompt=optimized_prompt,
            response=pipe_response,
            latency=pipe_latency,
            in_tokens=pipe_in,
            out_tokens=pipe_out
        )

        # Tracciamento statistiche
        route_logic_counts[route_logic] += 1
        model_counts[routed_to] += 1

        comp_saved = base_in - pipe_in
        comp_pct = (comp_saved / base_in * 100) if base_in > 0 else 0.0

        print(f"   ✓ Modello scelto: {routed_to} ({route_logic})")
        print(f"   ✓ Token Input: Baseline={base_in} vs Pipeline={pipe_in} (Compressione: {comp_pct:+.1f}%)")
        print(f"   ✓ Token Output: Baseline={base_out} vs Pipeline={pipe_out}")
        print(f"   ✓ Latenza: Baseline={base_latency:.2f}s vs Pipeline={pipe_latency:.2f}s")
        print(f"   ✓ Costo: Baseline=${base_cost:.6f} vs Pipeline=${pipe_cost:.6f}")

        results_summary.append({
            "category": category,
            "base_in": base_in,
            "pipe_in": pipe_in,
            "base_out": base_out,
            "pipe_out": pipe_out,
            "base_cost": base_cost,
            "pipe_cost": pipe_cost,
            "routed_to": routed_to,
            "route_logic": route_logic
        })

    # Salvataggio su file CSV
    tracker.save_to_csv()

    # ---------------------------------------------------------------------
    # REPORT FINALE INTEGRATO
    # ---------------------------------------------------------------------
    tot_base_in = sum(r["base_in"] for r in results_summary)
    tot_pipe_in = sum(r["pipe_in"] for r in results_summary)
    tot_base_out = sum(r["base_out"] for r in results_summary)
    tot_pipe_out = sum(r["pipe_out"] for r in results_summary)
    tot_base_cost = sum(r["base_cost"] for r in results_summary)
    tot_pipe_cost = sum(r["pipe_cost"] for r in results_summary)

    tokens_saved = tot_base_in - tot_pipe_in
    tokens_saved_pct = (tokens_saved / tot_base_in * 100) if tot_base_in else 0
    cost_saved = tot_base_cost - tot_pipe_cost
    cost_saved_pct = (cost_saved / tot_base_cost * 100) if tot_base_cost else 0

    print("\n" + "=" * 90)
    print("📊 REPORT FINALE BENCHMARK & TOKEN SAVINGS")
    print("=" * 90)
    print(f"Categorie testate: {len(results_summary)}")
    print(f"Distribuzione Modelli  -> Flash: {model_counts.get(FLASH_MODEL, 0)} | Pro: {model_counts.get(PRO_MODEL, 0)}")
    print(f"Distribuzione Routing  -> Heuristic: {route_logic_counts.get('heuristic', 0)} | Semantic: {route_logic_counts.get('semantic', 0)}")
    print("-" * 90)
    print(f"Token Input RAW (Baseline):           {tot_base_in}")
    print(f"Token Input Compressi (Pipeline):     {tot_pipe_in}")
    print(f"Token Input Risparmiati:              {tokens_saved} ({tokens_saved_pct:.1f}%)")
    print(f"Token Output Totali (Base / Pipe):   {tot_base_out} / {tot_pipe_out}")
    print("-" * 90)
    print(f"Costo Totale BASELINE (Sempre Pro):   ${tot_base_cost:.6f}")
    print(f"Costo Totale PIPELINE (Hybrid + Opt): ${tot_pipe_cost:.6f}")
    print(f"Risparmio Economico Totale:          ${cost_saved:.6f} ({cost_saved_pct:.1f}%)")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark_and_token_savings()