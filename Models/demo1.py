import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen2:0.5b", "prompt": "Hello, world!", "stream": False}
)
print(response.json()["response"])
