from fastapi import FastAPI, HTTPException
from pathlib import Path
from embeddings.base import EmbeddingEngine
from storage.vector_store import SimpleVectorStore


app = FastAPI()

embedding_engine = EmbeddingEngine()
vdb = SimpleVectorStore()


@app.get("/search")
def search(query: str):
    """This endpoint runs instantly because the model is already in RAM"""
    query_vector = embedding_engine.embed_query(query)
    results = vdb.query(query_vector, top_k=2)
    
    # Format results to return as standard JSON
    return [
        {"score": score, "text": doc.text, "metadata": doc.metadata} 
        for doc, score in results
    ]