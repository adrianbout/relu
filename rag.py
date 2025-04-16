import json
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# === Load local sentence-transformer model ===
embedder = SentenceTransformer('all_MiniLM_L6_v2', trust_remote_code=True)

# === Load your menu JSON ===
with open("menu.json", "r", encoding="utf-8") as f:
    menu_data = json.load(f)

# Format items for embedding
menu_lines = [
    f"{item['Item Name']} - {item['Price']}. {item['Description']}"
    for item in menu_data
]

# === Create ChromaDB collection ===
menu_embeddings = embedder.encode(menu_lines)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("menu")

for i, text in enumerate(menu_lines):
    collection.add(documents=[text], ids=[str(i)], embeddings=[menu_embeddings[i].tolist()])

# === Chat loop ===
while True:
    question = input("\nCustomer: ")
    if not question.strip():
        continue

    # Embed and query
    question_vector = embedder.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[question_vector], n_results=1)
    top_match = results['documents'][0][0]  # Only one item returned
    context = top_match

    # === Super short prompt ===
    prompt = f"""You are a restaurant assistant. Be brief and only answer the question directly using the menu below. Do not explain your thinking.

Menu:
{context}

Q: {question}
A:"""

    # === Call Ollama (DeepSeek) with streaming ===
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "options": {"num_predict": 100}
        },
        stream=True
    )

    # Stream response cleanly
    full_response = ""
    for line in res.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                full_response += chunk.get("response", "")
            except Exception as e:
                print("[Warning] Skipped a chunk due to error:", e)

    print("Waiter:", full_response.strip() or "[No response from model]")




