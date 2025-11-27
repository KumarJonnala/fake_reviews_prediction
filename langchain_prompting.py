import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain

# Load your CSV
df = pd.read_csv("merged_reviews.csv")   # matches uploaded file name
reviews = df["review"].tolist()

# Define zero-shot classification prompt
prompt_template = """
You are a strict classifier for product reviews.
Your task: classify a review as either REAL or FAKE.

Guidelines:
- REAL: personal experiences, detailed, honest sentiment, balanced.
- FAKE: overly generic, marketing-like, exaggerated, repeated phrases, unnatural enthusiasm.

Respond only with one word: REAL or FAKE.

Review: "{text}"
Answer:
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_template
)

# Load small model
llm = OllamaLLM(model="llama3.1")

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Classify each review
results = []
for r in reviews:
    result = chain.invoke({"text": r})
    label = result["text"].strip()
    results.append(label)

# Add results back to DataFrame
df["prediction"] = results

# Save outputs
df.to_csv("classified_reviews.csv", index=False)

print(df.head())