from pathlib import Path
from ingestion.loader import FileLoader
from ingestion.chunker import RecursiveCharacterChunker
from embeddings.base import EmbeddingEngine
from storage.vector_store import SimpleVectorStore
from config import STORAGE_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_THRESHOLD



chunker = RecursiveCharacterChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embedding_engine = EmbeddingEngine()
store = SimpleVectorStore()

store.load(directory=STORAGE_FOLDER)

def add_doc(file_path: str):
    path_obj = Path(file_path)
    if not path_obj.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"\nProcessing document: {path_obj.name}...")
    
    # 1. load document data
    loader = FileLoader(path_obj)
    raw_docs = loader.load()
    
    # split into chunks
    chunks = chunker.split_document(raw_docs)
    # embed chunks into vectors
    chunk_vectors = embedding_engine.embed_chunks(chunks)
    # add to vector store
    store.add_documents(chunks, chunk_vectors)
    store.persist(directory=STORAGE_FOLDER)
    print("Progress saved and synced to disk storage.")

def search(query: str):
    if not store.documents:
        print("The vector database is currently empty. Please add a document first!")
        return []
    
    query_vector = embedding_engine.embed_query(query)
    results = store.query(query_vector, top_k=2, threshold=SIMILARITY_THRESHOLD)

    return results

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
                results = search(query)
                if results:
                    for doc, score in results:
                        print(f"\n[Score: {score:.4f}] {doc.text} {doc.metadata['filename']} {doc.metadata['chunk_index']}")
                else:
                    print("No relevant documents found.")
            
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