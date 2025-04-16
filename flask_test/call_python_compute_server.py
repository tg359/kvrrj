import requests

# where to post the data
url = "http://127.0.0.1:5000/run_code"

data = {"num1": 3, "num2": "hi"}  # Replace with your actual data

response = requests.post(url, json=data)

# Print the raw response content
print("Response Content:", response.text)

# Attempt to decode the JSON response
try:
    json_response = response.json()
    print("JSON Response:", json_response)
except requests.exceptions.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
