import streamlit as st
import json
import os

# OVERRIDE SQLITE3 FOR CHROMADB IN STREAMLIT CLOUD
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme and CSS
st.markdown("""
<style>
    .main-header { font-family: 'Georgia', serif; color: #1a1a2e; }
    .citation-box { background-color: #f0f0f5; border-left: 4px solid #4a4a8a; padding: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_catalog():
    catalog_path = os.path.join(os.path.dirname(__file__), "papers", "paper_catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)["papers"]
    return []

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner="Initializing Database for the first time. This may take a minute...")
def get_rag_pipeline():
    try:
        from src.rag_pipeline import RAGPipeline
        import os
        
        # Determine if we need to build the database
        db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
        if not os.path.exists(db_path) or not os.listdir(db_path):
            st.info("Vector database not found on this instance. Re-building index from catalog...")
            from src.build_index import main as build_index_main
            build_index_main()
            
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG Pipeline: {e}")
        return None

if "rag" not in st.session_state:
    st.session_state.rag = get_rag_pipeline()

papers = load_catalog()
all_titles = [p["title"] for p in papers]

# --- SIDEBAR ---
with st.sidebar:
    st.title("📚 Research Copilot")
    st.markdown("Your AI research assistant")
    
    selected_papers = st.multiselect(
        "Filter by papers:",
        options=all_titles
    )
    
    strategy = st.selectbox(
        "Prompt Strategy:",
        ["Standard", "JSON Output", "Few-Shot", "Chain-of-Thought"]
    )
    
    if st.button("Clear Chat Iteractions"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Catalog Stats")
    st.markdown(f"- **Papers Indexed:** {len(papers)}")

# --- MAIN CHAT AREA ---
st.header("💬 Chat with your papers")

def display_citations(citations):
    with st.expander("Sources & Citations"):
        for c in citations:
            st.markdown(f"""
            <div class="citation-box">
                <strong>Paper:</strong> {c.get('paper')} ({c.get('year')})<br>
                <strong>Authors:</strong> {c.get('authors')}<br>
                <em>"{c.get('quote')}"</em>
            </div>
            """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            display_citations(message["citations"])

if prompt := st.chat_input("Ask a question about the papers in the catalog..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag:
        with st.chat_message("assistant"):
            with st.spinner("Searching papers & generating answer..."):
                response, citations = st.session_state.rag.query(
                    prompt, 
                    strategy=strategy, 
                    filter_papers=selected_papers
                )
                
                st.markdown(response)
                if citations:
                    display_citations(citations)
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "citations": citations
            })
    else:
        st.error("RAG pipeline is not available. Have you run `python src/build_index.py`?")
