import requests
import json

# Test the chat API endpoint
url = "http://localhost:8000/chat"
headers = {
    "Content-Type": "application/json"
}

# Test payload
payload = {
    "query": "What is Physical AI?",
    "top_k": 5
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")