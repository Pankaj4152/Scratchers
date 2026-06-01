from typing import List
from config import *
from ingestion.document import Document


class RecursiveCharacterChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_document(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks based on the specified chunk size and overlap.
        """
        chunked_docs = []
        for doc in documents:
            chunks = self._split_text(doc.text, self.separators)

            for i, chunk in enumerate(chunks):
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunked_docs.append(Document(text=chunk, metadata=chunk_metadata))
        return chunked_docs
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits the input text using the provided separators until the chunks are within the specified chunk size."""
        if len(text) <= self.chunk_size:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:] if len(separators) > 1 else separators

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            join_str = separator if current_chunk else ""
            potential_chunk = current_chunk + join_str + split

            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk

            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + join_str + split
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size and remaining_separators != [""]:
                final_chunks.extend(self._split_text(chunk, remaining_separators))
            else:
                final_chunks.append(chunk)
        return final_chunks