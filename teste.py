import requests

url = "http://127.0.0.1:8000/chat"

payload = {
    "pergunta": "O valor da multa foi mantido ou reduzido?",
    "top_k": 2
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("\nResposta:")
print(response.json())