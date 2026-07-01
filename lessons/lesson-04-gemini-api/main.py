from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

#Read the API key
api_key = os.getenv("GEMINI_API_KEY")

# Check whether the API key exists
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# Create a Gemini client
client = genai.Client(api_key=api_key)

# Ask Gemini a question
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain Retrieval-Augmented Generation(RAG) in two simple sentences."

)

# Print the answer
print(response.text)
