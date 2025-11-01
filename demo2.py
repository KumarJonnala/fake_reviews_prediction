import requests

# 🔧 Local Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

# 🧩 Function to query Ollama
def query_llm(prompt: str, model: str = "qwen2:0.5b") -> str:
    """Send a prompt to a locally running Ollama model."""
    response = requests.post(
        OLLAMA_API,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]

prompt = """
Classify the sentiment (Positive, Negative, or Neutral) of the following text:
"I love using my new laptop, it's fast and lightweight but the battery could last longer."
Explain your reasoning.
"""
print(query_llm(prompt))



# llm_with_explanations = query_llm(explanation_prompt)
# print("\n=== LLM Evaluation of LIME ===\n")
# print(llm_with_explanations)
