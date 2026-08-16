import re
from router_keywords import REASONING_KEYWORDS, CODING_KEYWORDS, MATH_KEYWORDS

class ModelRouter:
    def __init__(self):
        self.reasoning_keywords = REASONING_KEYWORDS
        self.code_keywords = CODING_KEYWORDS
        self.math_keywords = MATH_KEYWORDS

        # Precompile regex patterns for code detection to improve performance
        self.code_patterns = re.compile(
            r"(?i)("
            r"```|"                           # Markdown code block
            r"\bdef\s+[a-z_]\w*\s*\(|"        # Python function
            r"\bclass\s+[a-z_]\w*|"           # OOP classes
            r"\bimport\s+[a-z_]\w*|"          # Python/JS/Java imports
            r"\bfrom\s+[a-z_]\w*\s+import|"   # Python specific imports
            r"\bselect\b.*\bfrom\b|"          # SQL queries
            r"\bfunction\s+[a-z_]\w*\s*\(|"   # JS/PHP functions
            r"=>|"                            # Arrow functions (JS/C#)
            r"public\s+static\s+void|"        # Java main
            r"<!doctype\s+html>|<html.*?>"    # HTML base tags
            r")"
        )

    def heuristic_route(self, prompt: str) -> str:
        """
        Evaluates the prompt and returns a routing decision: "high", "low", or "uncertain".
        """
        # --- Early exits --- #
        
        # Explicit code
        if self.code_patterns.search(prompt):
            return "high"

        word_count = len(prompt.split())

        # Long prompt
        if word_count > 150:
            return "high"

        # --- Computing strong signals --- #
        # If the prompt touches at least two categories, it's likely complex
        prompt_lower = prompt.lower()
        strong_signals = 0
        
        if any(kw in prompt_lower for kw in self.reasoning_keywords):
            strong_signals += 1
            
        if hasattr(self, 'format_keywords') and any(kw in prompt_lower for kw in self.format_keywords):
            strong_signals += 1

        if any(kw in prompt_lower for kw in self.math_keywords):
            strong_signals += 1

        if strong_signals >= 2:
            return "high"
        if word_count <= 30 and strong_signals == 0:
            return "low"

        return "uncertain"