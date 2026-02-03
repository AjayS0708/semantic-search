"""RAG Application using Endee Vector Database with Hybrid Search"""

import os
import re
import requests
import msgpack
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class EndeeRAG:
    """RAG system using Endee vector database for retrieval"""
    
    def __init__(self, endee_url: str = "http://localhost:8080/api/v1", model_name: str = "all-MiniLM-L6-v2", use_reranking: bool = True, use_hybrid: bool = True):
        """Initialize RAG system with optional re-ranking and hybrid search"""
        self.endee_url = endee_url
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384
        self.doc_store = {}
        self.use_reranking = use_reranking
        self.use_hybrid = use_hybrid
        self.bm25 = None
        self.tokenized_corpus = []
        self.doc_ids = []
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') if use_reranking else None
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 search"""
        return re.findall(r'\w+', text.lower())
    
    def check_endee_connection(self) -> bool:
        """Check if Endee service is available"""
        try:
            response = requests.get(f"{self.endee_url}/health", timeout=2)
            return response.status_code == 200 and response.json().get('status') == 'ok'
        except Exception:
            return False
    
    def create_collection(self, collection_name: str = "documents") -> bool:
        """Create an index (collection) in Endee"""
        try:
            response = requests.post(
                f"{self.endee_url}/index/create",
                json={"index_name": collection_name, "dim": self.embedding_dim, "space_type": "cosine", "m": 16, "ef_con": 200},
                timeout=5
            )
            return response.status_code in [200, 201]
        except Exception:
            return False
    
    def insert_documents(self, documents: List[Dict[str, str]], collection: str = "documents") -> Tuple[bool, str]:
        """Insert documents into Endee vector database"""
        try:
            inserted_count = 0
            for doc in documents:
                text = doc.get('text', '')
                doc_id = doc.get('id', str(hash(text)))
                vector = self.model.encode([text])[0].tolist()
                self.doc_store[doc_id] = text
                
                if self.use_hybrid:
                    self.tokenized_corpus.append(self.tokenize(text))
                    self.doc_ids.append(doc_id)
                
                response = requests.post(
                    f"{self.endee_url}/index/{collection}/vector/insert",
                    json={"id": doc_id, "vector": vector, "text": text, **{k: v for k, v in doc.items() if k not in ['id', 'text', 'vector']}},
                    timeout=5
                )
                
                if response.status_code in [200, 201]:
                    inserted_count += 1
            
            if self.use_hybrid and self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            return True, f"Successfully inserted {inserted_count}/{len(documents)} documents"
        except Exception as e:
            return False, f"Error inserting documents: {str(e)}"
    
    def rerank_documents(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """Re-rank documents using cross-encoder for better accuracy"""
        if not self.use_reranking or not self.reranker or not documents:
            return documents[:top_k]
        
        try:
            pairs = [(query, doc['text']) for doc in documents]
            scores = self.reranker.predict(pairs)
            
            for i, doc in enumerate(documents):
                doc['rerank_score'] = float(scores[i])
                doc['original_score'] = doc['score']
            
            return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
        except Exception:
            return documents[:top_k]
    
    def bm25_search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """Perform BM25 keyword search"""
        if not self.bm25 or not self.doc_ids:
            return {}
        
        try:
            tokenized_query = self.tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            doc_scores = {doc_id: float(score) for doc_id, score in zip(self.doc_ids, scores)}
            return dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
        except Exception:
            return {}
    
    def hybrid_fusion(self, vector_results: List[Dict], bm25_scores: Dict[str, float], alpha: float = 0.7) -> List[Dict]:
        """Combine vector and BM25 scores with weighted fusion"""
        vec_scores = [doc.get('score', 0) for doc in vector_results] if vector_results else [0]
        vec_range = max(vec_scores) - min(vec_scores) or 1
        min_vec = min(vec_scores)
        
        bm25_vals = list(bm25_scores.values()) if bm25_scores else [0]
        bm25_range = max(bm25_vals) - min(bm25_vals) or 1
        min_bm25 = min(bm25_vals)
        
        hybrid_docs = {}
        
        for doc in vector_results:
            doc_id = doc['id']
            norm_vec = (doc.get('score', 0) - min_vec) / vec_range
            hybrid_docs[doc_id] = {
                'text': doc['text'], 'id': doc_id,
                'vector_score': doc.get('score', 0), 'bm25_score': 0,
                'hybrid_score': alpha * norm_vec
            }
        
        for doc_id, bm25_score in bm25_scores.items():
            norm_bm25 = (bm25_score - min_bm25) / bm25_range
            if doc_id in hybrid_docs:
                hybrid_docs[doc_id]['bm25_score'] = bm25_score
                hybrid_docs[doc_id]['hybrid_score'] += (1 - alpha) * norm_bm25
            else:
                hybrid_docs[doc_id] = {
                    'text': self.doc_store.get(doc_id, f"Document {doc_id}"),
                    'id': doc_id, 'vector_score': 0, 'bm25_score': bm25_score,
                    'hybrid_score': (1 - alpha) * norm_bm25
                }
        
        sorted_docs = sorted(hybrid_docs.values(), key=lambda x: x['hybrid_score'], reverse=True)
        for doc in sorted_docs:
            doc['score'] = doc['hybrid_score']
        
        return sorted_docs
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        try:
            query_vector = self.model.encode([query])[0].tolist()
            fetch_k = top_k * 3 if self.use_reranking else top_k
            
            response = requests.post(
                f"{self.endee_url}/index/documents/search",
                json={"vector": query_vector, "k": fetch_k},
                timeout=5
            )
            
            if response.status_code != 200:
                return []
            
            results = msgpack.unpackb(response.content, raw=False)
            context_docs = [
                {'text': self.doc_store.get(result[1], f"Document {result[1]}"), 'score': 1 - result[0], 'id': result[1]}
                for result in results if len(result) >= 2
            ]
            
            if self.use_hybrid and self.bm25:
                bm25_scores = self.bm25_search(query, top_k=fetch_k)
                context_docs = self.hybrid_fusion(context_docs, bm25_scores, alpha=0.7)
            
            return self.rerank_documents(query, context_docs, top_k) if self.use_reranking else context_docs[:top_k]
        except Exception:
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict], use_llm: bool = False) -> str:
        """Generate answer using retrieved context"""
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        relevant_docs = [doc for doc in context_docs if doc.get('score', 0) >= 0.25]
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question. Try rephrasing or asking about topics in the knowledge base."
        
        context_texts = [doc.get('text', '') or doc.get('metadata', {}).get('text', '') for doc in relevant_docs if doc.get('text', '') or doc.get('metadata', {}).get('text', '')]
        context = "\n\n".join(context_texts)
        
        if use_llm:
            if os.getenv("GEMINI_API_KEY") and GEMINI_AVAILABLE:
                return self._generate_with_gemini(query, context)
            elif os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
                return self._generate_with_openai(query, context)
        
        return self._generate_simple_answer(query, context_texts)
    
    def _generate_simple_answer(self, query: str, context_texts: List[str]) -> str:
        """Return raw context without LLM generation"""
        return "\n\n".join(text.strip() for text in context_texts[:3])
    
    def _generate_with_gemini(self, query: str, context: str) -> str:
        """Generate answer using Google Gemini"""
        try:
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Provide a clear, concise answer based on the context. If the context doesn't contain enough information to fully answer the question, provide what information is available and mention what's missing.

Answer:"""
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return response.text
        except Exception:
            return self._generate_simple_answer(query, context.split('\n'))
    
    def _generate_with_openai(self, query: str, context: str) -> str:
        """Generate answer using OpenAI"""
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            prompt = f"""Answer the question based on the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, max_tokens=300
            )
            return response.choices[0].message.content
        except Exception:
            return self._generate_simple_answer(query, context.split('\n'))
    
    def query(self, question: str, top_k: int = 3, use_llm: bool = False) -> Dict:
        """Complete RAG pipeline: retrieve and generate"""
        retrieved_docs = self.retrieve_context(question, top_k)
        answer = self.generate_answer(question, retrieved_docs, use_llm)
        return {"query": question, "answer": answer, "sources": retrieved_docs, "num_sources": len(retrieved_docs)}


def load_sample_documents(file_path: str = "data/documents.txt") -> List[Dict[str, str]]:
    """Load documents from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [{"id": f"doc_{idx}", "text": line.strip(), "source": file_path}
                    for idx, line in enumerate(f, 1) if line.strip()]
    except Exception:
        return []


def main():
    """Demo of RAG system"""
    print("=" * 60)
    print("RAG System with Endee Vector Database")
    print("=" * 60)
    
    rag = EndeeRAG()
    
    print("\n1. Checking Endee connection...")
    if not rag.check_endee_connection():
        print("⚠️  Endee service not running. Please start Endee and see README.md.")
        return
    print("✓ Connected")
    
    print("\n2. Loading documents...")
    documents = load_sample_documents()
    print(f"✓ Loaded {len(documents)} documents")
    
    print("\n3. Inserting documents...")
    success, message = rag.insert_documents(documents)
    print(f"{'✓' if success else '✗'} {message}")
    
    print("\n4. Running queries...")
    test_queries = ["What is Artificial Intelligence?", "Tell me about vector databases", "How does machine learning work?"]
    use_llm = bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    print(f"💡 {'Using LLM' if use_llm else 'Using simple extraction'}")
    
    for query in test_queries:
        print(f"\n📝 {query}")
        result = rag.query(query, top_k=2, use_llm=use_llm)
        print(f"💡 Answer: {result['answer'][:150]}...")
        print(f"📚 Sources: {result['num_sources']}")
    
    print("\n✅ Demo complete! Run 'streamlit run streamlit_app.py' for the web interface.")


if __name__ == "__main__":
    main()
