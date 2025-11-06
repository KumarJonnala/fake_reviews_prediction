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

# 🧾 Categories (20 newsgroups example)
newsgroups_classes = {
    'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
    'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
    'talk.politics.misc', 'talk.religion.misc'
}

# 🧍 Example document
doc = """NASA's new Artemis mission aims to return humans to the Moon by 2026, 
focusing on sustainable lunar exploration and preparation for Mars missions."""

# 🧠 Step 1: Ask LLM to classify the document
base_prompt = f"""
Please classify the following document into one of these categories:
{newsgroups_classes}

Document:
\"\"\"{doc}\"\"\"

Return:
- The single most likely category
- A detailed explanation for your reasoning
"""

llm_explanation = query_llm(base_prompt)
print("\n=== LLM Classification ===\n")
print(llm_explanation)

# 🔍 Step 2: Suppose you already ran LIME or SHAP and got word importance scores
lime_exp = [("NASA", 0.3), ("lunar", 0.25), ("Mars", 0.2), ("exploration", 0.15), ("mission", 0.1)]
pred_label = "sci.space"  # Example predicted label
class_names = list(newsgroups_classes)

# Build a readable string of LIME explanations
combined_features = "\n".join([f"{word}: {score:.2f}" for word, score in lime_exp])

# 🧩 Step 3: Ask LLM to evaluate the LIME explanation
explanation_prompt = f"""
Given the document:
\"\"\"{doc}\"\"\"

Predicted category: '{pred_label}'

LIME explanation (feature importance):
{combined_features}

Question:
1. Do you think '{pred_label}' is the correct category based on this LIME explanation?
2. Explain why or why not.
"""

llm_with_explanations = query_llm(explanation_prompt)
print("\n=== LLM Evaluation of LIME ===\n")
print(llm_with_explanations)
