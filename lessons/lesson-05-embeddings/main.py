from google import genai
from dotenv import load_dotenv
import os 

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=api_key)

response=client.models.embed_content(
    model = "gemini-embedding-001",
    contents="I love programming."

)

embedding = response.embeddings[0].values

print(type(embedding))
print(len(embedding))
print(embedding[:10])