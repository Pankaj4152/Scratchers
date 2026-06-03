from ingestion.loader import FileLoader
from ingestion.chunker import RecursiveCharacterChunker, SentenceChunker, SentenceTokenizerType
from embeddings.base import EmbeddingEngine
from storage.vector_store import SimpleVectorStore
from ingestion.document import Document
from pathlib import Path
from ingestion.loader import FileLoader
# Ground truth dataset
test_cases = [
    {
        "query": "What is the stipend for Tier 3 employees?",
        "expected_chunk_contains": "$1,800 USD",
        "description": "Tier 3 workspace allowance"
    },
    {
        "query": "How long are Slack messages retained?",
        "expected_chunk_contains": "30-day automated deletion policy",
        "description": "Slack retention policy"
    },
    {
        "query": "When are Tier 1 employees required to be in the office?",
        "expected_chunk_contains": "every Tuesday and Thursday",
        "description": "Tier 1 hybrid schedule"
    },
    {
        "query": "What are the core working hours for Tier 1?",
        "expected_chunk_contains": "10:00 AM to 4:00 PM",
        "description": "Tier 1 core hours"
    },
    {
        "query": "How often must Tier 2 employees travel to an office?",
        "expected_chunk_contains": "once per fiscal quarter",
        "description": "Tier 2 travel requirement"
    },
    {
        "query": "What is the parental leave policy?",
        "expected_chunk_contains": "16 weeks of fully paid leave",
        "description": "Parental leave benefit"
    },
    {
        "query": "How many years are required to qualify for a sabbatical?",
        "expected_chunk_contains": "five years of continuous full-time service",
        "description": "Sabbatical eligibility"
    },
    {
        "query": "What security requirement applies to GitHub repositories?",
        "expected_chunk_contains": "hardware security keys",
        "description": "GitHub MFA policy"
    },
    {
        "query": "What happens if a business decision is made in Slack?",
        "expected_chunk_contains": "document it in Notion within 48 hours",
        "description": "Knowledge management policy"
    },
    {
        "query": "How many vacation days do Tier 3 employees receive?",
        "expected_chunk_contains": "28 statutory annual vacation days",
        "description": "Tier 3 vacation policy"
    }
]

# Create sample document
loader = FileLoader(Path("F:\\Projects\\Scratchers\\rag\\sample\\employee_handbook_2026.txt"))
sample_text = loader.load()[0].text

sample_doc = Document(
    text=sample_text,
    metadata={"source": "sample.txt", "file_name": "sample.txt"}
)

# Test different configurations
configurations = [
    {
        "name": "RecursiveChunker (300 chars)",
        "chunker": RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50)
    },
    {
        "name": "RecursiveChunker (700 chars)",
        "chunker": RecursiveCharacterChunker(chunk_size=700, chunk_overlap=100)
    },
    {
        "name": "SentenceChunker (Regex)",
        "chunker": SentenceChunker(SentenceTokenizerType.REGEX)
    }
]

embedding_engine = EmbeddingEngine()

print("=" * 80)
print("RETRIEVAL EVALUATION TEST")
print("=" * 80)

for config in configurations:
    print(f"\n\nTesting: {config['name']}")
    print("-" * 80)
    
    # Setup
    chunker = config['chunker']
    store = SimpleVectorStore()
    
    # Ingest
    chunks = chunker.split_documents([sample_doc])
    embeddings = embedding_engine.embed_chunks(chunks)
    store.add_documents(chunks, embeddings)
    
    print(f"Total chunks: {len(chunks)}")
    
    # Evaluate
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        query = test_case["query"]
        expected = test_case["expected_chunk_contains"].lower()
        
        # Retrieve
        query_embedding = embedding_engine.embed_query(query)
        results = store.query(query_embedding, top_k=1, threshold=0.0)
        
        if results:
            retrieved_text = results[0][0].text.lower()
            score = results[0][1]
            
            # Check if expected content is in retrieved chunk
            if expected in retrieved_text:
                print(f"  ✅ PASS: '{test_case['description']}'")
                print(f"     Score: {score:.4f}")
                passed += 1
            else:
                print(f"  ❌ FAIL: '{test_case['description']}'")
                print(f"     Score: {score:.4f}")
                print(f"     Retrieved: {retrieved_text[:80]}...")
                failed += 1
        else:
            print(f"  ❌ FAIL: '{test_case['description']}' - No results")
            failed += 1
    
    print(f"\nResults: {passed}/{len(test_cases)} passed ({100*passed//len(test_cases)}%)")

print("\n" + "=" * 80)