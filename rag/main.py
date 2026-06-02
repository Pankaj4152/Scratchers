from pathlib import Path
import hashlib
import dotenv
from ingestion.loader import FileLoader
from ingestion.chunker import RecursiveCharacterChunker
from embeddings.base import EmbeddingEngine
from storage.vector_store import SimpleVectorStore
from config import STORAGE_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_THRESHOLD, LLM_MODEL, TOP_K

from generation.llm import LLMOrchestrator
from generation.prompt import PromptBuilder

import os
from dotenv import load_dotenv
load_dotenv()

chunker = RecursiveCharacterChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embedding_engine = EmbeddingEngine()
store = SimpleVectorStore()
store.load(directory=STORAGE_FOLDER)

# Initialize the LLM engine once on boot
llm_engine = LLMOrchestrator(model_name=LLM_MODEL)

def calculate_file_hash(path: Path) -> str:
    """Generates a SHA256 hash for the given file to detect changes."""
    hasher = hashlib.sha256()
    # Read as binary bytes to safely handle PDFs/Docx along with plain text
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def add_doc(file_path: str):
    path_obj = Path(file_path)
    if not path_obj.exists():
        print(f"File not found: {file_path}")
        return
    
    # 1. Calculate the current file's content fingerprint
    current_hash = calculate_file_hash(path_obj)
    filename = path_obj.name

    # 2. Check how this matches our existing vector store database
    status = store.check_file_status(filename, current_hash)

    if status == "UNCHANGED":
        print(f"\n[Skip]: '{filename}' hasn't changed. No updates needed.")
        return
    
    elif status == "MODIFIED":
        print(f"\n[Update]: Detected content modifications in '{filename}'. Overwriting old data...")
        # Wipe out old entries from RAM before computing new ones
        store.remove_document_by_filename(filename)
    
    else:
        print(f"\n[New]: Processing new document: {filename}...")
    
    
    # 3. Process the file data
    loader = FileLoader(path_obj)
    raw_docs = loader.load()
    
    # 4. Inject the file_hash into the raw document metadata before chunking
    # This ensures every downstream chunk inherits the file fingerprint!
    for doc in raw_docs:
        doc.metadata["file_hash"] = current_hash

    # 5. Run your standard chunking and vector processing pipeline
    chunks = chunker.split_document(raw_docs)
    chunk_vectors = embedding_engine.embed_chunks(chunks)

    # 6. Save to storage matrix and persist safely to disk files
    store.add_documents(chunks, chunk_vectors)
    store.persist(directory=STORAGE_FOLDER)
    print("Progress saved and synced to disk storage.")

def search(query: str):
    if not store.documents:
        print("The vector database is currently empty. Please add a document first!")
        return []
    
    query_vector = embedding_engine.embed_query(query)
    results = store.query(query_vector, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD)

    return results

def run_rag_pipeline(query: str):
    if not store.documents:
        print("The vector database is currently empty. Please add a document first!")
        return

    # 1. RETRIEVAL PHASE: Convert question to vector and find matched nodes
    query_vector = embedding_engine.embed_query(query)
    search_results = store.query(query_vector, top_k=3, threshold=SIMILARITY_THRESHOLD)

    if not search_results:
        print("\n[System]: No relevant document contexts passed the similarity threshold filter.")
        return

    # Extract just the Document entities from our (Document, Score) tuple results
    retrieved_docs = [doc for doc, score in search_results]

    # 2. PROMPT BUILDING PHASE: Wrap context and question together cleanly
    final_prompt = PromptBuilder.build_rag_prompt(query, retrieved_docs)

    # 3. GENERATION PHASE: Pass the structured payload to the target LLM
    print("\nThinking... (Querying LLM Engine)")
    llm_response = llm_engine.generate_response(final_prompt)

    print("\n" + "="*50)
    print(" ANSWER:")
    print("="*50)
    print(llm_response)
    print("="*50)
    
    # Print semantic citations for transparency
    print("\nSources Used:")
    for doc, score in search_results:
        print(f" - {doc.metadata.get('file_name')} (Similarity Score: {score:.4f}, Chunk: {doc.metadata.get('chunk_index')})")

def main():
    print("Welcome to the RAG Demo! You can add documents and ask questions about them.")

    while True:
        try:
            action = input("\nChoose an action: [1] Add Document, [2] Search, [3] Exit: ")
            
            if action == '1':
                file_path = input("Enter the path to your document (txt, md, pdf, docx): ")
                add_doc(file_path)
                print("Document added successfully!")
            
            elif action == '2':
                query = input("Enter your search query: ")
                run_rag_pipeline(query)
            
            elif action == '3':
                print("Goodbye!")
                break
            
            else:
                print("Invalid option. Please choose 1, 2, or 3.")

        except KeyboardInterrupt:
            # Clean exit handling if developer triggers a manual interrupt via terminal
            print("\n\nProcess interrupted by user. Exiting safely.")
            break

if __name__ == "__main__":
    main()