"""Test re-ranking feature"""
from rag_app import EndeeRAG, load_sample_documents

print("=" * 60)
print("Testing Re-Ranking Feature")
print("=" * 60)

# Initialize with re-ranking
print("\n1. Initializing RAG with re-ranking...")
rag = EndeeRAG(use_reranking=True)
print(f"✅ Re-ranking enabled: {rag.use_reranking}")
print(f"✅ Cross-encoder loaded: {rag.reranker is not None}")

# Load documents
print("\n2. Loading documents...")
docs = load_sample_documents()
success, msg = rag.insert_documents(docs)
print(f"✅ {msg}")

# Test query WITH re-ranking
print("\n3. Testing query WITH re-ranking...")
print("-" * 60)
query = "What is machine learning?"
result = rag.query(query, top_k=3, use_llm=False)

print(f"\n📝 Query: {query}")
print(f"💡 Answer: {result['answer'][:100]}...")
print(f"\n📚 Top 3 Sources (with re-ranking):")
for i, src in enumerate(result['sources'], 1):
    print(f"\n  Source {i}:")
    print(f"    Original score: {src.get('original_score', src['score']):.4f}")
    if 'rerank_score' in src:
        print(f"    Re-rank score:  {src['rerank_score']:.4f} 🔄")
    print(f"    Text: {src['text'][:80]}...")

print("\n" + "=" * 60)
print("✅ Re-ranking test complete!")
print("=" * 60)
