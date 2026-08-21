# File manually cleaned from the list generated with TF-IDF on the HF dataset.
# Removed: stopwords, auxiliary/generic verbs, and hyper-specific nouns from the
# original dataset (animal species, cities, films...) that do not indicate complexity
# and caused false positives on trivial prompts.

REASONING_KEYWORDS = {
    "reasoning",
    "show reasoning",
    "show the reasoning",
    "explain the reasoning",
    "show calculation",
    "show the calculation",
    "justify",
    "analyze",
    "analyse",
    "compare",
    "trade-off",
    "trade-offs",
    "tradeoffs",
    "evaluate",
    "debug",
    "review",
    "redesign",
    "reliability",
}


CODING_KEYWORDS = {
    "code",
    "function",
    "python",
    "javascript",
    "node.js",
    "sql",
    "postgresql",
    "api",
    "endpoint",
    "algorithm",
    "database",
    "list",
    "array",
    "dictionary",
    "express",
    "flask",
    "class",
    "query",
    "script",
}


MATH_KEYWORDS = {
    "equation",
    "probability",
    "calculation",
    "derivative",
    "integral",
    "matrix",
    "sequence",
    "convergent",
    "convergence",
    "bounded",
    "irrational",
    "theorem",
    "induction",
    "regression",
    "optimization",
    "average speed",
    "square root",
}


FORMAT_KEYWORDS = {
    "json",
    "yaml",
    "xml",
    "schema",
    "openapi",
    "swagger",
    "csv",
    "flowchart",
    "uml",
    "erd",
    "microservices",
    "api spec",
    "system design",
    "structured format",
    "step by step",
    "outline",
    "bullet points",
    "numbered list",
}