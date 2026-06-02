import numpy as np
from typing import List, Tuple, Dict, Any
import os
import json

from ingestion.document import Document
from config import STORAGE_FOLDER, SIMILARITY_THRESHOLD

class SimpleVectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """Pairs each text chunk with its corresponding vector embedding and saves them."""
        for doc, emb in zip(documents, embeddings):
            self.documents.append(doc)
            # Convert the list of floats to a NumPy array for fast math operations
            self.vectors.append(np.array(emb))

    def get_ingested_files(self) -> List[str]:
        """
        Returns a list of unique filenames that have been ingested into the vector store.
        """
        return {doc.metadata.get("filename") for doc in self.documents if doc.metadata}
    
    def check_file_status(self, filename: str, current_hash: str) -> str:
        """
        Checks the state of a file against the database.
        Returns: 'NEW', 'UNCHANGED', or 'MODIFIED'
        """
        file_chunks = [doc for doc in self.documents if doc.metadata.get("filename") == filename]
        stored_hash = file_chunks[0].metadata.get("filehash") if file_chunks else None
        if not file_chunks:
            return "NEW"
        
        if stored_hash == current_hash:
            return "UNCHANGED"
        return "MODIFIED"

    def remove_document_by_filename(self, filename: str):
        """
        Deletes all text chunks and vector embeddings associated with a specific filename.
        This cleans out old data completely before an overwrite.
        """
        new_docs = []
        new_vecs = []

        for doc, vec in zip(self.documents, self.vectors):
            if doc.metadata.get("filename") != filename:
                new_docs.append(doc)
                new_vecs.append(vec)

        deleted_count = len(self.documents) - len(new_docs)
        self.documents = new_docs
        self.vectors = new_vecs
        if deleted_count > 0:
            print(f"Purged {deleted_count} stale chunks of '{filename}' from memory database.")

            
    def _cosine_similarity(self, v1 :np.ndarray, v2: np.ndarray) -> float:
        """
        Calculates the cosine similarity score between two vectors.
        Formula: (A · B) / (||A|| * ||B||)
        Returns a value between -1 and 1, where 1 means identical meaning.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))
    
    def query(self, query_vector: List[float], top_k: int = 5, threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[Document, float]]:
        """
        Compares the query vector against all stored vectors,
        ranks them by similarity, and returns the top_k matching documents.
        """
        query_vec = np.array(query_vector)
        scores = []

        for idx, doc_vec in enumerate(self.vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            # if score >= threshold:
            scores.append((self.documents[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
    
    def persist(self, directory: str = STORAGE_FOLDER):
        """
        Saves the memory state (vectors and documents) to physical files on the disk.
        Vectors are stored as high-performance NumPy binaries; text chunks are stored as JSON."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 1. Save Vectors as a stacked binary matrix
        if self.vectors:
            vector_matrix = np.vstack(self.vectors)
            np.save(os.path.join(directory, "vectors.npy"), vector_matrix)
        else:
            print("Warning: No vectors to save.")

        # 2. Save Documents using the Dataclass serialization bridge
        doc_data = [doc.to_dict() for doc in self.documents]
        with open(os.path.join(directory, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=4)

        print(f"Successfully persisted {len(self.documents)} items to disk folder: '{directory}/'")


    def load(self, directory: str = STORAGE_FOLDER) -> bool:
        """
        Loads the vectors and document textures back into your active RAM from disk storage.
        
        :return: True if the database loaded successfully, False if no prior storage was found.
        """

        vector_path = os.path.join(directory, "vectors.npy")
        doc_path = os.path.join(directory, "documents.json")

        if not os.path.exists(vector_path) or not os.path.exists(doc_path):
            print(f"No existing vector store found in '{directory}/'. Starting with an empty store.")
            return False
        
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
            self.documents = [Document.from_dict(item) for item in doc_data]

        vector_matrix = np.load(vector_path)
        self.vectors = [row for row in vector_matrix]

        print(f"Successfully loaded {len(self.documents)} items from disk folder: '{directory}/'")
        return True