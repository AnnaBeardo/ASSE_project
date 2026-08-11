import re 

class ModelRouter:

    def route(self, prompt: str) -> str:
        score = 0
        prompt_lower = prompt.lower()

        word_count = len(prompt.split())

        if word_count > 200:
            score += 2
        elif word_count > 80:
            score += 1

        # Search for reasoning-related keywords in the prompts
        reasoning_keywords = [
            "analyze",
            "analyse",
            "compare",
            "derive",
            "prove",
            "explain why",
            "step by step",
        ]

        if any(keyword in prompt_lower for keyword in reasoning_keywords):
            score += 1

        # Search for code patterns in the prompt
        code_patterns = [
            r"```",
            r"\bdef\s+\w+\(",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+"
        ]

        if any(re.search(pattern, prompt) for pattern in code_patterns):
            score += 1

        # Search for math-related keywords in the prompt
        math_keywords = [
            "equation",
            "matrix",
            "integral",
            "derivative",
        ]

        if any(keyword in prompt_lower for keyword in math_keywords):
            score += 1

        if score >= 2:
            return "high"

        return "low"