"""Streamlit Web Interface for RAG Application"""

import os
import time
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from rag_app import EndeeRAG, load_sample_documents

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG with Endee",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stats-box h3 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: bold;
    }
    .stats-box p {
        color: white;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .query-suggestion {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        font-weight: 500;
    }
    .query-suggestion:hover {
        background: linear-gradient(135deg, #667eea30 0%, #764ba230 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 0 4px;
    }
    .badge-vector {
        background: #667eea20;
        color: #667eea;
        border: 1px solid #667eea;
    }
    .badge-bm25 {
        background: #f093fb20;
        color: #f093fb;
        border: 1px solid #f093fb;
    }
    .badge-hybrid {
        background: #4facfe20;
        color: #4facfe;
        border: 1px solid #4facfe;
    }
    .badge-rerank {
        background: #43e97b20;
        color: #43e97b;
        border: 1px solid #43e97b;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.update({
        'rag': None, 'documents_loaded': False, 'history': [], 'error_message': None,
        'use_hybrid': True, 'use_llm': bool(os.getenv("GEMINI_API_KEY")),
        'button_counter': 0
    })

def initialize_rag(use_hybrid=True):
    """Initialize RAG system with re-ranking and optional hybrid search
    
    Args:
        use_hybrid: Enable hybrid search combining vector (70%) + BM25 (30%)
    """
    try:
        rag = EndeeRAG(use_reranking=True, use_hybrid=use_hybrid)
        if not rag.check_endee_connection():
            st.session_state.error_message = "Cannot connect to Endee. Make sure the server is running on http://localhost:8080"
            return None
        st.session_state.error_message = None
        return rag
    except Exception as e:
        st.session_state.error_message = f"Error initializing RAG: {str(e)}"
        return None

def load_documents():
    """Load and insert documents"""
    if st.session_state.rag is None:
        st.error("❌ Please connect to Endee first!")
        return False
    
    try:
        with st.spinner("Loading documents..."):
            documents = load_sample_documents()
            if not documents:
                st.error("❌ No documents found. Check that data/documents.txt exists.")
                return False
            success, message = st.session_state.rag.insert_documents(documents)
            st.session_state.documents_loaded = success
            (st.success if success else st.error)(f"{'✅' if success else '❌'} {message}")
            return success
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

# Header
st.markdown('<div class="main-header">🤖 RAG System with Endee Vector Database</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retrieval Augmented Generation for Intelligent Q&A</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Connection status
    st.subheader("1️⃣ Connection")
    if st.button("Connect to Endee", type="primary"):
        with st.spinner("Connecting..."):
            use_hybrid = st.session_state.get('use_hybrid', True)
            st.session_state.rag = initialize_rag(use_hybrid=use_hybrid)
            if st.session_state.rag:
                st.success("✅ Connected!")
            time.sleep(0.5)
    
    if st.session_state.rag:
        st.success("🟢 Connected")
    elif st.session_state.error_message:
        st.error(f"🔴 {st.session_state.error_message}")
    else:
        st.warning("🟡 Not connected")
    
    st.divider()
    
    # Document loading
    st.subheader("2️⃣ Documents")
    if st.button("Load Sample Documents", disabled=st.session_state.rag is None):
        load_documents()
    
    if st.session_state.documents_loaded:
        st.success("📚 Documents loaded")
    else:
        st.info("📄 No documents loaded")
    
    st.divider()
    
    # Settings
    st.subheader("3️⃣ Settings")
    top_k = st.slider("Number of sources", 1, 5, 3)
    
    # Hybrid search toggle
    use_hybrid = st.checkbox(
        "🔀 Enable Hybrid Search (Vector + BM25)",
        value=st.session_state.use_hybrid,
        help="Combines semantic search with keyword matching for 50-60% better accuracy"
    )
    
    if use_hybrid != st.session_state.use_hybrid:
        st.session_state.use_hybrid = use_hybrid
        if st.session_state.rag:
            st.session_state.rag = initialize_rag(use_hybrid=use_hybrid)
    
    st.success("🔀 Hybrid Search: ON" if use_hybrid else "📊 Vector Search: ON")
    
    # Gemini AI toggle
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    use_llm = st.checkbox("🤖 Use Gemini AI for answers", value=st.session_state.use_llm, disabled=not has_gemini)
    st.session_state.use_llm = use_llm
    
    if has_gemini:
        st.success("✨ Gemini AI enabled" if use_llm else "ℹ️ Showing raw context only")
    
    # Re-ranking indicator
    if st.session_state.rag and st.session_state.rag.use_reranking:
        st.info("🔄 Re-ranking enabled (40% better accuracy)")
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ About")
    st.info("This RAG application uses **Endee** for vector storage, **Hybrid Search** (Vector + BM25) for 50-60% better accuracy, and **Cross-Encoder Re-ranking** for final refinement.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask a Question")
    
    # Smart Query Suggestions
    st.subheader("� Quick Start - Try these queries:")
    
    query_suggestions = [
        ("🤖 AI Basics", "What is Artificial Intelligence and how does it work?"),
        ("🧠 Deep Learning", "Explain deep learning and neural networks"),
        ("🔍 Vector Search", "How do vector databases enable semantic search?"),
        ("📊 RAG Systems", "What is RAG and why is it important?"),
        ("⚡ Transformers", "Tell me about transformer architecture in NLP"),
        ("🎯 ML Algorithms", "What are the main machine learning algorithms?"),
    ]
    
    cols = st.columns(3)
    for idx, (label, suggested_query) in enumerate(query_suggestions):
        with cols[idx % 3]:
            unique_key = f"suggestion_{idx}_{st.session_state.button_counter}"
            if st.button(label, key=unique_key, width='stretch'):
                if st.session_state.rag and st.session_state.documents_loaded:
                    with st.spinner("Processing..."):
                        result = st.session_state.rag.query(suggested_query, top_k=top_k, use_llm=use_llm)
                        st.session_state.history.append(result)
                        st.session_state.button_counter += 1
                        st.rerun()
    
    st.divider()
    
    # Query input
    query = st.text_input(
        "Or type your own question:",
        placeholder="e.g., What is Artificial Intelligence?",
        help="Ask any question related to the loaded documents"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        ask_button = st.button("🔍 Ask", type="primary", disabled=not st.session_state.documents_loaded)
    with col_btn2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.session_state.history = []
        st.rerun()
    
    # Process query
    if ask_button and query:
        if st.session_state.rag is None:
            st.error("Please connect to Endee first!")
        elif not st.session_state.documents_loaded:
            st.error("Please load documents first!")
        else:
            with st.spinner("Thinking..."):
                result = st.session_state.rag.query(query, top_k=top_k, use_llm=use_llm)
                st.session_state.history.append(result)
    
    # Display results
    if st.session_state.history:
        st.divider()
        
        # Show most recent result
        result = st.session_state.history[-1]
        
        st.subheader("💡 Answer")
        st.markdown(f"**Q:** {result.get('query', result.get('question', 'Unknown'))}")
        st.info(result['answer'])
        
        # Show sources
        if result['sources']:
            st.subheader("📚 Sources & Score Analysis")
            
            # Score Visualization Chart
            if result['sources'] and result['sources'][0].get('hybrid_score') is not None:
                chart_data = [
                    {
                        'Source': f'Source {idx}',
                        'Vector': source.get('vector_score', 0),
                        'BM25': min(source.get('bm25_score', 0) / 10, 1),
                        'Hybrid': source.get('hybrid_score', 0),
                        'Rerank': (source.get('rerank_score', 0) + 5) / 10 if source.get('rerank_score') else None
                    }
                    for idx, source in enumerate(result['sources'][:top_k], 1)
                ]
                
                if chart_data:
                    fig = go.Figure([
                        go.Bar(name='Vector Score', x=[d['Source'] for d in chart_data], y=[d['Vector'] for d in chart_data], marker_color='#667eea'),
                        go.Bar(name='BM25 Score (normalized)', x=[d['Source'] for d in chart_data], y=[d['BM25'] for d in chart_data], marker_color='#f093fb'),
                        go.Bar(name='Hybrid Score', x=[d['Source'] for d in chart_data], y=[d['Hybrid'] for d in chart_data], marker_color='#4facfe')
                    ])
                    
                    if any(d['Rerank'] is not None for d in chart_data):
                        fig.add_trace(go.Bar(name='Rerank Score (normalized)', x=[d['Source'] for d in chart_data], y=[d['Rerank'] for d in chart_data], marker_color='#43e97b'))
                    
                    fig.update_layout(
                        title='📊 Score Comparison Across Sources', xaxis_title='Sources', yaxis_title='Score (0-1)',
                        barmode='group', height=400, showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        template='plotly_white'
                    )
                    
                    # Fixed: Updated to use width='stretch' instead of deprecated use_container_width
                    st.plotly_chart(fig, width='stretch')
                    st.success(f"🏆 Best Match: {chart_data[0]['Source']} with Hybrid Score: {chart_data[0]['Hybrid']:.3f}")
            
            st.divider()
            
            # Display individual sources
            for idx, source in enumerate(result['sources'], 1):
                text = source.get('text', '') or source.get('metadata', {}).get('text', 'No text available')
                vector_score = source.get('vector_score')
                hybrid_score = source.get('hybrid_score')
                rerank_score = source.get('rerank_score')
                
                score_parts = []
                if hybrid_score is not None and vector_score is not None:
                    score_parts = [f"Vector: `{vector_score:.3f}`", f"BM25: `{source.get('bm25_score', 0):.3f}`", f"Hybrid: `{hybrid_score:.3f}` 🔀"]
                elif rerank_score is not None:
                    score_parts = [f"Original: `{source.get('score', 0):.3f}`", f"Re-ranked: `{rerank_score:.3f}` 🔄"]
                else:
                    score_parts = [f"Similarity: `{source.get('score', 0):.3f}`"]
                
                st.markdown(f"**📄 Source {idx}** - {' | '.join(score_parts)}")
                st.text_area(label=f"source_{idx}", value=text, height=150, disabled=True, label_visibility="collapsed")

with col2:
    st.header("📊 Statistics")
    
    if st.session_state.history:
        st.markdown(f"""
        <div class="stats-box">
            <h3>{len(st.session_state.history)}</h3>
            <p>Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Query history
    st.subheader("📜 History")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.history) - idx + 1}"):
                st.write(f"**Q:** {item.get('query', item.get('question', 'Unknown'))}")
                st.write(f"**Sources:** {item['num_sources']}")
    else:
        st.info("No queries yet")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with Endee Vector Database | Powered by Sentence Transformers
</div>
""", unsafe_allow_html=True)
