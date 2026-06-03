from ingestion.chunker import (
    ParagraphChunker,
    SentenceChunker,
    SentenceTokenizerType,
    TokenChunker,
    RecursiveCharacterChunker
)
from ingestion.document import Document


# Sample test document
sample_text = """Retrieval-Augmented Generation (RAG) is a powerful technique in modern NLP. 
It combines the strengths of retrieval and generation models.

The RAG architecture has three main components: a retriever, a reader, and a generator. 
The retriever finds relevant documents. The reader extracts relevant passages. The generator produces the final answer.

RAG systems work by first retrieving relevant context from a knowledge base. Then, this context is used to augment the input prompt. 
Finally, a language model generates the response based on the augmented prompt.

One key advantage of RAG is that it can leverage external knowledge without fine-tuning the language model. 
This makes RAG systems more flexible and updatable. Another advantage is that RAG can provide citations and improve interpretability."""

sample_doc = Document(
    text=sample_text,
    metadata={"source": "test.txt", "file_name": "sample.txt"}
)

# Test each chunker
chunkers = {
    "ParagraphChunker": ParagraphChunker(),
    "SentenceChunker (REGEX)": SentenceChunker(SentenceTokenizerType.REGEX),
    "SentenceChunker (NLTK)": SentenceChunker(SentenceTokenizerType.NLTK),
    "TokenChunker (500 tokens)": TokenChunker(chunk_size=500, chunk_overlap=50),
    "TokenChunker (100 tokens)": TokenChunker(chunk_size=100, chunk_overlap=10),
    "RecursiveCharacterChunker (300 chars)": RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50),
    "RecursiveCharacterChunker (700 chars)": RecursiveCharacterChunker(chunk_size=700, chunk_overlap=100),
}

print("=" * 80)
print("CHUNKER COMPARISON TEST")
print("=" * 80)

results = {}

for name, chunker in chunkers.items():
    chunks = chunker.split_documents([sample_doc])
    results[name] = chunks
    
    print(f"\n{name}:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
    print(f"  Min/Max: {min(len(c.text) for c in chunks)}/{max(len(c.text) for c in chunks)} chars")
    
    print(f"  Chunk preview:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.text[:60].replace('\n', ' ')
        print(f"    [{i}] {preview}...")
    
    if len(chunks) > 3:
        print(f"    ... and {len(chunks) - 3} more chunks")

print("\n" + "=" * 80)