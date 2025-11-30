import requests

# Testaa suoraan HTTP:llä
print("Testataan suoraa HTTP-kutsua...")
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2:3b",
        "messages": [
            {"role": "user", "content": "Sano 'moi' suomeksi"}
        ],
        "stream": False
    },
    timeout=30
)

print(f"Status: {response.status_code}")
print(f"Vastaus: {response.json()}")

# Nyt testaa LangChainilla
print("\n---\nTestataan LangChainilla...")
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=2048,  # Pienennä konteksti-ikkunaa
)

messages = [HumanMessage(content="Sano 'moi' suomeksi.")]
response = llm.invoke(messages)
print(f"LangChain vastaus: {response.content}")

