import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompt_compressor import PromptCompressor

from euristic_router import ModelRouter
from router_semantic import SemanticRouter 

load_dotenv()

FLASH_MODEL = "gemini-3.1-flash-lite"
PRO_MODEL = "gemma-4-31b-it"

class LLMHandler:
    def __init__(self, temperature: float = 0.0):
        self.flash_model = ChatGoogleGenerativeAI(
            model=FLASH_MODEL, 
            temperature=temperature
        )
        self.pro_model = ChatGoogleGenerativeAI(
            model=PRO_MODEL, 
            temperature=temperature
        )
        self.parser = StrOutputParser()
        self.compressor = PromptCompressor()
        
        self.heuristic_router = ModelRouter()

        embedder_instance = self.compressor.model 
        self.semantic_router = SemanticRouter(embedder=embedder_instance)

    def invoke(self, prompt: str, enable_caveman: bool = True) -> dict:
        """
        Optimize prompt, execute hybrid routing and return response with stats.
        """

        # --- Dynamic routing --- #
        # Heuristic first, then Semantic if uncertain, done on the original prompt before any compression
        route_decision = self.heuristic_router.heuristic_route(prompt)
        used_semantic = False

        if route_decision == "uncertain":
            route_decision = self.semantic_router.route(prompt)
            used_semantic = True

        model = self.pro_model if route_decision == "high" else self.flash_model

        # --- Compression --- #
        optimization_res = self.compressor.optimize_prompt(prompt, enable_compression=enable_caveman)
        
        system_text = optimization_res["system_prompt"]
        user_text = optimization_res["user_prompt"]

        # --- Building LangChain --- #
        messages = []
        if system_text:
            messages.append(("system", system_text))
        messages.append(("user", "{input}"))
        
        prompt_template = ChatPromptTemplate.from_messages(messages)
        chain = prompt_template | model | self.parser

        chain = chain.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True
        )
        
        # --- Invocation and response --- #
        response = chain.invoke({"input": user_text})

        return {
            "response": response,
            "routed_to": PRO_MODEL if route_decision == "high" else FLASH_MODEL,
            "route_logic": "semantic" if used_semantic else "heuristic",
            "optimized_prompt_text": f"[System: {system_text}] User: {user_text}",
            "compression_stats": optimization_res["stats"]
        }