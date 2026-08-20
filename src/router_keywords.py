# File manually cleaned from the list generated with TF-IDF on the HF dataset.
# Removed: stopwords, auxiliary/generic verbs, and hyper-specific nouns from the
# original dataset (animal species, cities, films...) that do not indicate complexity
# and caused false positives on trivial prompts.

REASONING_KEYWORDS = set([
    'identify', 'classify', 'summarize', 'summary', 'ideas', 'types',
    'bulleted list', 'classify following', 'identify instrument',
    'identify animal', 'given text',
])

CODING_KEYWORDS = set([
    'create', 'code', 'function', 'array', 'python', 'string', 'strings',
    'numbers', 'program', 'javascript', 'query', 'sql', 'generate', 'java',
    'print', 'prints', 'sql query', 'html', 'table', 'elements', 'element',
    'class', 'create function', 'write code', 'write function', 'write sql',
    'write python', 'algorithm', 'data', 'value', 'sum', 'given string',
    'given list', 'given array', 'loop', 'returns', 'integers', 'input',
    'output', 'construct', 'convert', 'calculate', 'following code',
    'sort', 'script', 'dictionary', 'snippet',
])

MATH_KEYWORDS = set([
    'total', 'cost', 'costs', 'money', 'hours', 'hour', 'minutes', 'week',
    'half', 'years', 'year', 'twice', 'bought', 'buy', 'miles', 'students',
    'days', 'pounds', 'month', 'pay', 'feet', 'books',
])

FORMAT_KEYWORDS = set([
    "json", "yaml", "xml", "schema", "openapi", "swagger",
    "csv", "markdown table", "table format",
    "architecture", "diagram", "flowchart", "uml", "erd",
    "microservices", "api spec", "system design", "template", "boilerplate", "structured format",
    "bullet points", "numbered list", "step by step",
    "config file", "dockerfile", "yaml file", "spec sheet",
])