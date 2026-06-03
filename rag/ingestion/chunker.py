from typing import List
from config import *
from ingestion.document import Document
import re
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """
    Base class for all chunking strategies.

    Responsibilities:
        - Split documents into smaller chunks
        - Preserve metadata
        - Add chunk metadata for traceability
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split raw text into chunks.

        Args:
            text: Input text.

        Returns:
            List of text chunks.
        """
        pass

    def split_documents(self, documents: List[Document]) ->List[Document]:
        """
        Split a list of documents into chunked documents.
        """
        chunked_docs = []
        for document in documents:
            chunks = self.split_text(document.text)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "chunk_type": self.__class__.__name__
                })
                chunked_docs.append(Document(text=chunk, metadata=chunk_metadata))
        return chunked_docs
    


class ParagraphChunker(BaseChunker):
    """
    Split text into paragraphs.

    A paragraph is defined as text separated
    by one or more blank lines.
    """
    
    def split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        
        return [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n", text)
            if chunk.strip()
        ]





class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = ["\n\n", "\n", ". ","? ","! ", " ", ""]

    
    
    def split_text(self, text: str, separators: List[str]) -> List[str]:
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