import requests

# Define the URL
url = "https://social-mistakenly-python.ngrok-free.app/"

# Define the JSON data
data = {
   "gender": "neutral",
   "beta_values": "0.0 0.0 -5.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
}

# Set the headers
headers = {
    'Content-Type': 'application/json'
}

# Send the POST request
response = requests.post(url, json=data, headers=headers)

# Check the response
if response.status_code == 200:
    print("Data sent successfully!")
    print("Response:", response.json())
else:
    print("Failed to send data.")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
