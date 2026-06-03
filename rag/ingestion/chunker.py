from typing import List
from config import *
from ingestion.document import Document
import re
from abc import ABC, abstractmethod
from enum import Enum
import tiktoken


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


class SentenceTokenizerType(Enum):
    REGEX = "regex"
    NLTK = "nltk"
    SPACY = "spacy"

class SentenceChunker(BaseChunker):

    def __init__(self, tokenizer_type: SentenceTokenizerType = SentenceTokenizerType.REGEX):

        self.tokenizer_type = tokenizer_type

        if tokenizer_type == SentenceTokenizerType.SPACY:
            import spacy
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
            # self.nlp = spacy.load(
            #     SPACY_MODEL,
            #     disable=[
            #         "tagger",
            #         "ner",
            #         "lemmatizer",
            #         "attribute_ruler"
            #     ]
            # )
        elif tokenizer_type == SentenceTokenizerType.NLTK:
            try:
                from nltk.tokenize import PunktSentenceTokenizer
                self.tokenizer = PunktSentenceTokenizer()
            except Exception as e:
                raise RuntimeError(
                    "NLTK Punkt tokenizer is not available."
                ) from e
            
    def _regex_split(
        self,
        text: str
    ):
        return [
            s.strip()
            for s in re.split(
                r'(?<=[.!?])\s+',
                text
            )
            if s.strip()
        ]
    
    def _nltk_split(self, text: str):
        return self.tokenizer.tokenize(text)

    def _spacy_split(self, text: str):
        doc = self.nlp(text)
        return [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip()
        ]                

    def split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        # if self.tokenizer_type == SentenceTokenizerType.REGEX:
        #     return self._regex_split(text)

        # elif self.tokenizer_type == SentenceTokenizerType.NLTK:
        #     return self._nltk_split(text)

        # elif self.tokenizer_type == SentenceTokenizerType.SPACY:
        #     return self._spacy_split(text)
        methods = {
            SentenceTokenizerType.REGEX: self._regex_split,
            SentenceTokenizerType.NLTK: self._nltk_split,
            SentenceTokenizerType.SPACY: self._spacy_split,
        }

        try:
            return methods[self.tokenizer_type](text)
        except KeyError:
            raise ValueError(
                f"Unsupported tokenizer: {self.tokenizer_type}"
            )                                       


class TokenChunker(BaseChunker):
    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=50,
        encoding_name="cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(
            encoding_name
        )

    def split_text(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            start += (self.chunk_size - self.chunk_overlap)

        return chunks
    


class RecursiveCharacterChunker(BaseChunker):
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = ["\n\n", "\n", ". ","? ","! ", " ", ""]

    
    def split_text(self, text: str) -> List[str]:
        """Recursively splits the input text using the provided separators."""
        return self._split_recursive(text, self.separators)
    

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Helper method for recursive splitting."""
        if not separators:
            return [text] if text.strip() else []
        
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
                final_chunks.extend(self._split_recursive(chunk, remaining_separators))
            else:
                final_chunks.append(chunk)
        return final_chunks