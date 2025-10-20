
import os
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import requests

from db import SimpleVectorDB  

# --------------------------
# Config
# --------------------------
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "TaylorAI/bge-micro-v2")
BLOGS_JSON_PATH = os.environ.get("BLOGS_JSON_PATH", "blog.json")
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_API_KEY = "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"
VECTORS_PATH = "vectors.npy"        # File to store vectors
METADATA_PATH = "metadata.json" 

# --------------------------
# Initialize app
# --------------------------
app = FastAPI(title="DIY Vector Search System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB = SimpleVectorDB()
EMBEDDER = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    metric: str = "cosine"

class ChatRequest(BaseModel):
    query: str
    k: int = 5

@app.on_event("startup")
def load_data():
    global EMBEDDER, DB
    EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    
    if os.path.exists(VECTORS_PATH) and os.path.exists(METADATA_PATH):
        # ---- Load precomputed vectors ----
        print("[Startup] Loading precomputed vectors from disk...")
        vectors = np.load(VECTORS_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for i, vec in enumerate(vectors):
            DB.insert(metadata[i]["id"], vec, metadata[i]["text"])
        print(f"[Startup] Loaded {len(DB)} entries (dim={DB.dim})")

    else:
        # ---- Compute embeddings from JSON ----
        if not os.path.exists(BLOGS_JSON_PATH):
            raise FileNotFoundError(f"{BLOGS_JSON_PATH} not found!")

        with open(BLOGS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts, ids = [], []
        for item in data:
            ids.append(item["id"])
            texts.append(item["metadata"]["text"])

        print(f"[Startup] Embedding {len(texts)} documents...")
        vectors = EMBEDDER.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Save vectors and metadata for future use
        np.save(VECTORS_PATH, vectors)
        metadata_to_save = [{"id": ids[i], "text": texts[i]} for i in range(len(ids))]
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

        # Insert into DB
        for i, vec in enumerate(vectors):
            DB.insert(ids[i], vec, texts[i])
        print(f"[Startup] Indexed {len(DB)} entries (dim={DB.dim})")


@app.get("/health")
def health():
    return {"status": "ok", "count": len(DB), "dim": DB.dim}


@app.post("/search")
def search(req: SearchRequest):
    if not DB.entries:
        raise HTTPException(status_code=500, detail="Database is empty.")
    q_vec = EMBEDDER.encode([req.query])[0]
    results = DB.search(q_vec, k=req.k, metric=req.metric)
    return results


@app.post("/chat")
def chat(req: ChatRequest):
    """RAG endpoint: retrieve context + call an LLM."""
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="Missing LLM_API_KEY environment variable")

    # Step 1: Retrieve relevant documents
    q_vec = EMBEDDER.encode([req.query])[0]
    retrieved = DB.search(q_vec, k=req.k, metric="cosine")

    context = "\n\n".join([r["text"] for r in retrieved])

    # Step 2: Call LLM API (Groq)

    payload = {
        "model": "llama-3.3-70b-versatile",  # ✅ valid Groq model
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use only the provided context to answer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}\nAnswer clearly and concisely. If the context does not contain the answer, respond with 'No information on that'."}
        ],
        "temperature": 0.4,
    }

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    response = requests.post(LLM_API_URL, headers=headers, json=payload)

    print("[LLM DEBUG]", response.status_code, response.text)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=response.text)
    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return {"answer": answer, "context_docs": retrieved}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
