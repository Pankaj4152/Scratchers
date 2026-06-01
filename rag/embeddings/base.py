import torch
from typing import List
from sentence_transformers import SentenceTransformer
from ingestion.document import Document

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model '{model_name}' into memory...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_chunks(self, chunks: List[Document]) -> List[List[float]]:
        """
        Takes a list of our custom Document chunks, extracts the text, 
        and converts them into a list of vector embeddings (floating-point numbers).
        """
        texts_to_embed = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts_to_embed, 
            batch_size=32,
            convert_to_numpy=False, 
            show_progress_bar=False
        )

        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Converts a user query string into a vector embedding.
        """
        return self.model.encode(query, convert_to_numpy=False)