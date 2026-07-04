from pathlib import Path
from dotenv import load_dotenv
import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Read API key
api_key = os.getenv("GEMINI_API_KEY")

#Create Gemini client
client = genai.Client(api_key=api_key)

# Read file
file_path = Path("sample.txt")



with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chunk_size=2
lines=text.split("\n")

chunks=[]

for i in range(0,len(lines),chunk_size):
    chunk = "\n".join(lines[i:i+chunk_size])
    chunks.append(chunk)

#Generate embeddings
embeddings=[]

documents=[]


for chunk in chunks:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    vector = response.embeddings[0].values
    
    documents.append({
        "text":chunk,
        "embedding":vector
    })

question = "What are embeddings?"

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding=response.embeddings[0].values

scores=[]

for document in documents:
    score=cosine_similarity(
        [question_embedding],
        [document["embedding"]]
    )[0][0]

    scores.append(score)

best_index=scores.index(max(scores))

best_document=documents[best_index]

print(best_document["text"])
