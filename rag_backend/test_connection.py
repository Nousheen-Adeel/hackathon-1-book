import requests
import json

# Test the chat API endpoint with timeout
url = 'http://localhost:8000/chat'
headers = {'Content-Type': 'application/json'}
payload = {'query': 'Hello', 'top_k': 5}

try:
    print("Sending request to:", url)
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.text}')
except requests.exceptions.Timeout:
    print('Request timed out - server might not be responding')
except requests.exceptions.ConnectionError as e:
    print(f'Connection error - server might not be running: {e}')
except Exception as e:
    print(f'Error: {e}')