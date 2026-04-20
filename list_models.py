import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
client = genai.Client()

print("Available models:")
for model in client.models.list():
    if 'embedContent' in model.supported_methods:
        print(f"- {model.name}")
