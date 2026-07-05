from pathlib import Path
from dotenv import load_dotenv
import os
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")

#gemini client
client = genai.Client(api_key=api_key)

file_path=Path("sample.txt")


with open(file_path,"r",encoding="utf-8") as file:
    text = file.read()

chunk_size=2
lines=text.split("\n")

chunks=[]

for i in range(0,len(lines),chunk_size):
    chunk="\n".join(lines[i:i+chunk_size])
    chunks.append(chunk)

#Generate embedding
embeddings=[]

documents=[]

for chunk in chunks:
    response=client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    vector = response.embeddings[0].values

    documents.append({
        "text":chunk,
        "embedding":vector
    })

question="what are embeddings?"

response=client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding=response.embeddings[0].values


results=[]

for document in documents:
    score = cosine_similarity(
        [question_embedding],
        [document["embedding"]]
    )[0][0]

    results.append({
        "text":document["text"],
        "score":score
    })

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

top_k = 2
top_chunks=results[:top_k]

print("\nTop Retrieved Chunks\n")

for chunk in top_chunks:
    print(f"Score : {chunk["score"]:.4f}")
    print(chunk["text"])
    print("_"*40)


context =""

for chunk in top_chunks:
    context+=chunk["text"]+"\n\n"

print(context)


prompt=f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.
If the answer is not in the context, say:
"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

print(prompt)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)
