import re

from router_keywords import (
    REASONING_KEYWORDS,
    CODING_KEYWORDS,
    MATH_KEYWORDS,
    FORMAT_KEYWORDS,
)


class ModelRouter:
    def __init__(self):
        self.reasoning_keywords = REASONING_KEYWORDS
        self.code_keywords = CODING_KEYWORDS
        self.math_keywords = MATH_KEYWORDS
        self.format_keywords = FORMAT_KEYWORDS

        self.code_patterns = re.compile(
            r"("
            r"```|"
            r"\bdef\s+[a-z_]\w*\s*\(|"
            r"\bclass\s+[a-z_]\w*|"
            r"\bimport\s+[a-z_]\w*|"
            r"\bfrom\s+[a-z_]\w*\s+import|"
            r"\bselect\b.*\bfrom\b|"
            r"\bfunction\s+[a-z_]\w*\s*\(|"
            r"=>|"
            r"public\s+static\s+void|"
            r"<!doctype\s+html>|"
            r"<html.*?>|"
            r"app\.(?:get|post|put|delete|patch)\s*\("
            r")",
            re.IGNORECASE | re.DOTALL,
        )

        # Patterns that indicate likely complex reasoning or derivation.
        self.hard_complexity_patterns = re.compile(
            r"\b("
            r"prove|"
            r"proof|"
            r"rigorously|"
            r"derive|"
            r"derivation|"
            r"race condition|"
            r"deadlock|"
            r"vulnerability|"
            r"vulnerabilities|"
            r"security vulnerabilities|"
            r"performance problem|"
            r"performance problems|"
            r"time complexity|"
            r"space complexity|"
            r"redesign"
            r")\b",
            re.IGNORECASE,
        )

        # Formats that are complex enough to warrant PRO on their own.
        self.strong_format_patterns = re.compile(
            r"\b(flowchart|uml|erd)\b",
            re.IGNORECASE,
        )

        # Requests that require strict output formatting.
        self.strict_output_patterns = re.compile(
            r"("
            r"\breturn\s+only\b|"
            r"\bno\s+additional\s+text\b|"
            r"\bdo\s+not\s+include\s+markdown\b|"
            r"\bexactly\b.*\b(?:json|csv|yaml|xml)\b|"
            r"\b(?:json|csv|yaml|xml)\b.*\bexactly\b"
            r")",
            re.IGNORECASE | re.DOTALL,
        )

        # Cases where the prompt is a simple arithmetic expression, 
        # even with a request for step-by-step reasoning.
        self.simple_arithmetic_pattern = re.compile(
            r"^\s*"
            r"(?:answer\s+step\s+by\s+step:\s*)?"
            r"(?:what\s+is\s+)?"
            r"\d+\s*[\+\-\*/]\s*\d+"
            r"\s*\??\s*$",
            re.IGNORECASE,
        )

        # Factual questions that remain simple even if they are embedded
        # in a lot of irrelevant context.
        self.simple_factual_patterns = re.compile(
            r"("
            r"what\s+is\s+the\s+capital\s+of|"
            r"what\s+is\s+the\s+chemical\s+symbol\s+for|"
            r"what\s+does\s+the\s+word\b.+?\bmean"
            r")",
            re.IGNORECASE | re.DOTALL,
        )

        # General-purpose patterns for ambiguous requests.
        self.ambiguous_patterns = re.compile(
            r"\b("
            r"design a solution|"
            r"improve the reliability"
            r")\b",
            re.IGNORECASE,
        )

        self.constraint_patterns = re.compile(
            r"\b("
            r"must|"
            r"required|"
            r"do not|"
            r"don't|"
            r"never|"
            r"preserve|"
            r"at least|"
            r"at most|"
            r"exactly|"
            r"without|"
            r"if"
            r")\b",
            re.IGNORECASE,
        )

    def _keyword_matches(self, text: str, keywords: set) -> set:
        matches = set()

        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"

            if re.search(pattern, text, re.IGNORECASE):
                matches.add(keyword)

        return matches

    def heuristic_route(self, prompt: str) -> str:
        """
        Returns:
            "high"      -> use PRO
            "low"       -> use FLASH
            "uncertain" -> delegate to SemanticRouter
        """

        if not prompt:
            return "low"

        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        # ------------------------------------------------------------------
        # 1. Evident complexity patterns
        # ------------------------------------------------------------------

        if self.code_patterns.search(prompt):
            return "high"

        if self.hard_complexity_patterns.search(prompt):
            return "high"

        # ------------------------------------------------------------------
        # 2. Evidently simple patterns
        # ------------------------------------------------------------------

        if self.simple_arithmetic_pattern.match(prompt):
            return "low"

        if self.simple_factual_patterns.search(prompt):
            return "low"

        # ------------------------------------------------------------------
        # 3. Analysis of keywords and categories
        # ------------------------------------------------------------------

        reasoning_matches = self._keyword_matches(
            prompt_lower,
            self.reasoning_keywords,
        )

        coding_matches = self._keyword_matches(
            prompt_lower,
            self.code_keywords,
        )

        math_matches = self._keyword_matches(
            prompt_lower,
            self.math_keywords,
        )

        format_matches = self._keyword_matches(
            prompt_lower,
            self.format_keywords,
        )

        # Patterns that indicate strong format requirements.
        if self.strong_format_patterns.search(prompt):
            return "high"

        # More than one strong format requirement.
        if len(format_matches) >= 2:
            return "high"

        if self.strict_output_patterns.search(prompt):
            return "high"

        has_reasoning = bool(reasoning_matches)
        has_coding = bool(coding_matches)
        has_math = bool(math_matches)
        has_format = bool(format_matches)

        # ------------------------------------------------------------------
        # 4. Simple coding patterns
        # ------------------------------------------------------------------

        if (
            has_coding
            and not has_reasoning
            and not has_math
            and not has_format
            and word_count <= 25
        ):
            return "low"

        # ------------------------------------------------------------------
        # 5. Prompt information-dense / many constraints
        # ------------------------------------------------------------------

        constraint_count = len(
            self.constraint_patterns.findall(prompt_lower)
        )

        if constraint_count >= 3:
            return "high"

        # ------------------------------------------------------------------
        # 6. Combination of strong signals
        # ------------------------------------------------------------------

        strong_signals = sum(
            [
                has_reasoning,
                has_coding,
                has_math,
                has_format,
            ]
        )

        if strong_signals >= 2:
            return "high"

        # ------------------------------------------------------------------
        # 7. Intentional ambiguity
        # ------------------------------------------------------------------

        if self.ambiguous_patterns.search(prompt):
            return "uncertain"

        # ------------------------------------------------------------------
        # 8. Combination of weak signals and prompt length
        # ------------------------------------------------------------------

        if word_count <= 15 and strong_signals == 0:
            return "low"

        # Un prompt lungo NON è automaticamente complesso.
        if word_count > 250 and strong_signals == 0:
            return "uncertain"

        return "uncertain"