import time

from llm_client import LLMHandler, PRO_MODEL
from metrics_tracker import MetricsTracker


TEST_CASES = [

    # 1. SIMPLE FACTUAL
    (
        "simple_factual",
        "What is the capital of Portugal?"
    ),
    (
        "simple_factual",
        "What is the chemical symbol for gold?"
    ),

    # 2. SIMPLE CONCEPTUAL
    (
        "simple_conceptual",
        "What is photosynthesis and why is it important for plants?"
    ),
    (
        "simple_conceptual",
        "Explain in simple terms what inflation means in economics."
    ),

    # 3. SHORT LOGICAL REASONING
    (
        "short_logical_reasoning",
        "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain why."
    ),
    (
        "short_logical_reasoning",
        "No reptiles are mammals. All snakes are reptiles. Can any snake be a mammal? Explain the reasoning."
    ),

    # 4. MATHEMATICAL REASONING
    (
        "mathematical_reasoning",
        "A car travels 120 km at 60 km/h and then another 120 km at 40 km/h. What is its average speed for the entire journey? Show the reasoning."
    ),
    (
        "mathematical_reasoning",
        "A bag contains 5 red balls, 4 blue balls, and 3 green balls. Two balls are drawn without replacement. What is the probability that both balls are the same color? Show the calculation."
    ),

    # 5. SIMPLE CODING
    (
        "simple_coding",
        "Write a Python function that takes two integers and returns the larger one."
    ),
    (
        "simple_coding",
        "Write a JavaScript function that takes an array of numbers and returns their sum."
    ),

    # 6. COMPLEX CODING / DEBUGGING
    (
        "complex_coding",
        """A Node.js service processes jobs from a queue using several asynchronous workers.
Sometimes two workers process the same job because they read its status before either one
updates it. Explain the race condition and redesign the processing logic so that each job
can be processed by only one worker, assuming PostgreSQL is used as the database."""
    ),
    (
        "complex_coding",
        """The following Python function becomes extremely slow when processing several million
records. Identify the main performance problems and propose a more efficient implementation:

def find_duplicates(values):
    duplicates = []
    for i in range(len(values)):
        for j in range(len(values)):
            if i != j and values[i] == values[j] and values[i] not in duplicates:
                duplicates.append(values[i])
    return duplicates
"""
    ),

    # 7. CODE REVIEW / SECURITY
    (
        "security",
        """Review this Express endpoint for security vulnerabilities and explain how to fix them:

app.get('/users/:id', async (req, res) => {
    const query = `SELECT * FROM users WHERE id = ${req.params.id}`;
    const result = await db.query(query);
    res.json(result.rows);
});
"""
    ),
    (
        "security",
        """Review the following Python code for security problems:

@app.route('/download')
def download():
    filename = request.args.get('file')
    return send_file('/var/app/uploads/' + filename)

Identify possible attacks and propose a secure implementation.
"""
    ),

    # 8. STRICT OUTPUT FORMAT
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
        "strict_format",
        """List the numbers 1 through 5 and their squares as a CSV with exactly these columns:
number,square
Do not include Markdown, explanations, or any text outside the CSV."""
    ),

    # 9. LONG BUT EASY
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
        "long_but_easy",
        """Imagine that I am organizing a classroom activity for students who are learning
very basic chemistry vocabulary. Before the activity starts, I want to prepare a small
answer key so that I can check their responses quickly. The students have already studied
atoms, elements, the periodic table, symbols, metals, nonmetals, and a few common laboratory
materials. They do not need any discussion of atomic structure, isotopes, electron
configuration, oxidation states, mining, economics, jewelry, industrial applications, or
the historical origin of element names. They also do not need a description of the
periodic table or instructions for performing an experiment. The exercise is intentionally
simple and is meant only to test whether they recognize standard chemical symbols. Several
neighboring questions ask for the symbols of oxygen, hydrogen, iron, copper, silver, and
sodium. I want the answers to be very short so they fit in a narrow column on a printed
worksheet. There is no ambiguity about notation, no alternative convention to consider,
and no trick involving compounds or ions. I am not asking for the atomic number, atomic
mass, Latin name, electron count, or any other property. I am also not asking how the
symbol was chosen or why it differs from the English word. The response should contain
only the standard chemical symbol used internationally for the element in its neutral
elemental form. No sentence is necessary, and no additional context would be useful for
this exercise. The specific element for this question is gold. What is the chemical symbol
for gold?"""
    ),

    # 10. SHORT BUT COMPLEX
    (
        "short_but_complex",
        "Prove that the square root of 2 is irrational."
    ),
    (
        "short_but_complex",
        "Is every bounded sequence convergent? Justify rigorously."
    ),

    # 11. IRRELEVANT INFORMATION
    (
        "irrelevant_information",
        """Yesterday I was reorganizing my desk while listening to music. I found an old
notebook from university and started reading some unrelated notes about databases.
My laptop battery was almost empty, so I moved to another room and made some coffee.
None of this is relevant to the actual task. Explain the difference between a primary
key and a foreign key in a relational database."""
    ),
    (
        "irrelevant_information",
        """Our company started in 1998 and originally had an office next to a railway station.
The office moved twice, the company logo changed in 2010, and the cafeteria was renovated
last year. We currently have about 200 employees and use several unrelated internal tools.
For the actual question, I only need to know: what is the difference between HTTP and HTTPS?"""
    ),

    # 12. INFORMATION-DENSE
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
        "information_dense",
        """Design an API endpoint for creating a user. It must accept name, email, and age.
Name and email are required. Age must be an integer between 18 and 120.
Email addresses must be unique.
Return HTTP 201 when the user is created, HTTP 400 for invalid input,
and HTTP 409 when the email already exists.
The response must be JSON and must never expose the user's password."""
    ),

    # 13. NEGATIONS / EXCEPTIONS
    (
        "negation",
        """Given the list [5, 2, 5, 3, 2, 1], remove duplicate values but do not sort
the list. Preserve the order of the first occurrence of every value."""
    ),
    (
        "negation",
        """Rewrite the following sentence without changing its meaning.
Do not shorten it, do not make it more formal, and do not remove the word "not":
"The service should not restart when the configuration file is missing."
"""
    ),

    # 14. AMBIGUOUS / BORDERLINE
    (
        "borderline",
        "Design a solution for managing user sessions in a web application."
    ),
    (
        "borderline",
        "Help me improve the reliability of a data processing pipeline."
    ),

    # 15. ADVERSARIAL ROUTING
    (
        "adversarial_routing",
        "What does the word architecture mean?"
    ),
    (
        "adversarial_routing",
        "Answer step by step: what is 2 + 2?"
    ),
]


def main():
    handler = LLMHandler(temperature=0.0)
    tracker = MetricsTracker(run_name="prompt_types")

    for i, (category, prompt) in enumerate(TEST_CASES, start=1):
        print("\n" + "=" * 80)
        print(f"TEST {i}/{len(TEST_CASES)}")
        print(f"CATEGORY: {category}")
        print("=" * 80)
        print(f"Original prompt: {prompt}")

        # ---------------------------------------------------------------------
        # BASELINE
        # ---------------------------------------------------------------------
        print("\n-> BASELINE")

        start_time = time.time()

        try:
            # invoke() ora restituisce un AIMessage
            ai_msg = handler.pro_model.invoke(prompt) 
            baseline_response = ai_msg.content
            
            # Estrai token esatti
            usage = ai_msg.usage_metadata
            base_in = usage.get("input_tokens", 0) if usage else None
            base_out = usage.get("output_tokens", 0) if usage else None
            
            baseline_latency = time.time() - start_time
        except Exception as e:
            baseline_response = f"ERROR: {str(e)}"
            baseline_latency = 0.0
            base_in = base_out = 0

        tracker.log_call(
            model_name=PRO_MODEL,
            complexity=f"BASELINE | {category}",
            prompt=prompt,
            response=baseline_response,
            latency=baseline_latency,
            in_tokens=base_in,    
            out_tokens=base_out   
        )

        # ---------------------------------------------------------------------
        # PIPELINE
        # ---------------------------------------------------------------------
        print("-> PIPELINE")

        start_time = time.time()

        try:
            result = handler.invoke(prompt, enable_caveman=True)

            pipeline_response = result["response"]
            pipeline_model = result["routed_to"]
            route_logic = result["route_logic"]
            optimized_prompt = result["optimized_prompt_text"]
            compression_stats = result["compression_stats"]
            
            # Recupera i token esatti aggiunti in llm_client.py
            pipe_in = result.get("api_in_tokens")
            pipe_out = result.get("api_out_tokens")

            pipeline_latency = time.time() - start_time

        except Exception as e:
            # ... blocco except invariato, ma aggiungi token a 0 ...
            pipe_in = pipe_out = 0

        tracker.log_call(
            model_name=pipeline_model,
            complexity=f"PIPELINE | {category} | {route_logic}",
            prompt=optimized_prompt,
            response=pipeline_response,
            latency=pipeline_latency,
            in_tokens=pipe_in,     
            out_tokens=pipe_out    
        )

        print(f"Model: {pipeline_model}")
        print(f"Routing logic: {route_logic}")
        print(
            "Compression: "
            f"{compression_stats['original_tokens']} -> "
            f"{compression_stats['compressed_tokens']} tokens "
            f"(saved {compression_stats['saved_tokens']})"
        )

    tracker.save_to_csv()


if __name__ == "__main__":
    main()