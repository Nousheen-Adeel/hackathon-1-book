import requests
import json

# Test the chat API endpoint
url = "http://localhost:8000/chat"
headers = {
    "Content-Type": "application/json"
}

# Test payload - using the correct field name according to QuestionRequest model
payload = {
    "question": "What is Physical AI?"
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Also test the /ask endpoint
url = "http://localhost:8000/ask"
try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(f"Status Code (/ask): {response.status_code}")
    print(f"Response (/ask): {response.text}")
except Exception as e:
    print(f"Error (/ask): {e}")