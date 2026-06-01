# RAG — Retrieval-Augmented Generation (small demo)

This folder contains a minimal, from-scratch RAG (Retrieval-Augmented Generation) demo.

It demonstrates the core pipeline:
- ingest documents (plain text file),
- clean and chunk text,
- generate embeddings, and
- store vectors + metadata to an on-disk store (`.storage/`).

The code is intentionally simple and educational — it's a good starting point for experimenting with vector stores, chunking strategies, and embedding backends.

Contents (quick)
- `config.py` — configuration constants (chunk sizes, storage folder, thresholds).
- `requirements.txt` — Python dependencies used by the demo.
- `main.py` — small CLI to add documents and run simple queries.
- `server.py` — FastAPI server exposing a `/search` endpoint for queries.
- `embeddings/` — embedding engine(s) (currently a thin wrapper around a model).
- `ingestion/` — document loading, cleaning, chunking, and Document model.
- `storage/` — simple vector store implementation that persists vectors (`.npy`) and metadata (`.json`) under `.storage/`.
- `sample/` — example text documents you can ingest.
- `.storage/` — runtime persisted store (vectors + documents). This is created automatically when you add documents.

Quick start (Windows)

1. Create and activate a virtual environment (recommended):

```bat
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bat
pip install -r requirements.txt
```

3. Add the sample documents to the store (CLI):

```bat
python main.py
```



Files and responsibilities

- `config.py`
	- Holds constants like `CHUNK_SIZE`, `CHUNK_OVERLAP`, `STORAGE_FOLDER`, and similarity thresholds.

- `ingestion/loader.py`
	- Loads files from disk and returns raw text.

- `ingestion/cleaner.py`
	- Cleans text (normalization, whitespace, simple heuristics) before chunking.

- `ingestion/chunker.py`
	- Splits long text into overlapping chunks. Tune chunk size and overlap via `config.py`.

- `ingestion/document.py`
	- Data model for a document and its chunks (metadata + text + id).

- `embeddings/base.py`
	- Thin abstraction around an embedding model. Swap in any embedding backend here.

- `storage/vector_store.py`
	- Simple vector store. Persists metadata to `.storage/documents.json` and vectors to `.storage/vectors.npy`.
	- Provides nearest-neighbour search using cosine similarity.

- `main.py`
	- CLI entrypoint with simple add/search flows for quick experiments.



Troubleshooting

- No results returned
	- Check `.storage/documents.json` exists and contains entries. If empty, you haven't added documents.

- Embedding model errors
	- If `sentence-transformers` or `torch` aren't installed or compatible, embeddings will fail. Install the packages listed above and verify your Python version.

- Performance
	- This demo stores vectors in a NumPy array and performs a brute-force similarity search. For large datasets, use FAISS or another index.

Extending the demo (ideas)

- Add a production-grade vector index (FAISS, Annoy, hnswlib).
- Add batching when embedding many chunks.
- Store richer metadata and implement filtering/scoped search.
- Add tests for ingestion, chunking, and search.

License and notes

This code is an educational demo. Reuse and adapt freely.
