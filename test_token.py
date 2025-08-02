import requests

api_token = "hf_1234abcd5678efgh9012ijkl3456mnop"  # ton vrai token ici

headers = {"Authorization": f"Bearer {api_token}"}
url = "https://api-inference.huggingface.co/models/sayeed99/segformer_b3_clothes"

response = requests.get(url, headers=headers)

print("Code HTTP :", response.status_code)
print("Réponse :", response.text)
