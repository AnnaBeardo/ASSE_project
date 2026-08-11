import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from prompt_compressor import PromptCompressor

load_dotenv()

FLASH_MODEL = "gemma-4-31b-it"
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

    def invoke(self, prompt: str, complexity: str = "low", enable_caveman: bool = True) -> dict:
        """
        Optimize prompt, execute routing and return response with stats.
        """
        # 1. Optimisation
        optimization_res = self.compressor.optimize_prompt(prompt, enable_compression=enable_caveman)
        
        system_text = optimization_res["system_prompt"]
        user_text = optimization_res["user_prompt"]

        # 2. Dynamic Routing (PLACEHOLDER)
        model = self.pro_model if complexity == "high" else self.flash_model

        # 3. Constructing chain using LangChain
        if system_text:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_text),
                ("user", "{input}")
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("user", "{input}")
            ])

        chain = prompt_template | model | self.parser
        
        # 4. Invoke
        response = chain.invoke({"input": user_text})

        return {
            "response": response,
            "optimized_prompt_text": f"[System: {system_text}] User: {user_text}",
            "compression_stats": optimization_res["stats"]
        }

# Test di utilizzo
if __name__ == "__main__":
    llm = LLMHandler()
    
    sample_prompt = "Ciao buongiorno, potresti gentilmente scrivermi una funzione in Python per invertire una stringa? Grazie!"
    
    print("--- ESECUZIONE SENZA OTTIMIZZAZIONE ---")
    res_raw = llm.invoke(sample_prompt, complexity="low", enable_caveman=False)
    print(f"Risposta:\n{res_raw['response']}\n")

    print("--- ESECUZIONE CON OTTIMIZZAZIONE ---")
    res_opt = llm.invoke(sample_prompt, complexity="low", enable_caveman=True)
    print(f"Prompt inviato: {res_opt['optimized_prompt_text']}")
    print(f"Risposta:\n{res_opt['response']}")
    print(f"Statistiche Token Input: {res_opt['compression_stats']}")