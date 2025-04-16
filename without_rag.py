import json
import requests

# === Load full menu from JSON ===
with open("menu.json", "r", encoding="utf-8") as f:
    menu_data = json.load(f)

# Format the menu as plain text for the prompt (with fallback fields)
menu_text = "\n".join([
    f"{item.get('Item Name', 'Unknown Item')} - {item.get('Price', 'N/A')}. {item.get('Description', 'No description')}"
    for item in menu_data
])


# === Ask questions directly to Ollama ===
while True:
    question = input("\nCustomer: ")
    if not question.strip():
        continue

    # Final prompt with few-shot examples
    prompt = f"""You are a helpful and concise restaurant assistant. Answer the customer's question based only on the menu below.

                Prices are listed in L.L, which stands for Lebanese Pounds.

                Be brief and answer directly using the item name and price. Do not explain your reasoning.

Menu:
{menu_text}

Examples:
Customer: What is the price of Iced Latte?
Answer: Iced Latte is L.L 300,000.

Customer: How much is the Cinnamon Rolls?
Answer: Cinnamon Rolls is L.L 500,000.

Customer: {question}
Answer:"""

    # Send to Ollama (DeepSeek)
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "options": {
                "num_predict": 100
            }
        },
        stream=True
    )

    # Stream response live
    full_response = ""
    for line in res.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode("utf-8"))
                full_response += chunk.get("response", "")
            except Exception as e:
                print("[Warning] Skipped chunk due to:", e)

    if not full_response.strip():
        print("Waiter: Sorry, I couldn't find that item in the menu.")
    else:
        print("Waiter:", full_response.strip())

