import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No API key found in .env")
    exit(1)

print(f"Loaded API Key: {api_key[:5]}...{api_key[-5:]}")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

try:
    print("Testing connection to Google Gemini API...")
    response = model.generate_content("Say 'Hello, Gemini works!'")
    print("\nSUCCESS!")
    print(response.text)
except Exception as e:
    print(f"\nFAILED:")
    print(e)
