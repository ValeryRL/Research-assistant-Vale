import os
import google.generativeai as genai
from .retrieval import ChromaVectorStore, EmbeddingModel
from .prompts import PROMPTS
from dotenv import load_dotenv

load_dotenv()

# Configure GenAI if key is present
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class RAGPipeline:
    def __init__(self, db_path="./chroma_db", model_type="google"):
        self.vector_store = ChromaVectorStore(persist_directory=db_path)
        self.vector_store.create_collection("papers_small_chunks")
        self.model_type = model_type
        self.embedding_model = EmbeddingModel("all-MiniLM-L6-v2")

    def query(self, user_question: str, strategy: str = "Standard", filter_papers: list = None):
        # 1. Embed Query
        query_embedding = self.embedding_model.embed_query(user_question)
        
        # 2. Retrieve contexts
        where_clause = None
        if filter_papers:
            # If a simple filter is possible
            pass
            
        results = self.vector_store.query(query_embedding, n_results=3, where=where_clause)
        
        # 3. Format context
        contexts = []
        citations = []
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                contexts.append(doc_text)
                
                # Format citation for the frontend
                year = meta.get('year', 'n.d.')
                authors = meta.get('authors', 'Unknown Authors')
                title = meta.get('title', 'Unknown Title')
                citation = {
                    "paper": title,
                    "authors": authors,
                    "year": year,
                    "quote": doc_text[:150] + "..." # Just an excerpt
                }
                citations.append(citation)
                
        context_str = "\n\n---\n\n".join(contexts)
        
        # 4. Prompt Engineering
        prompt_template = PROMPTS.get(strategy, PROMPTS["Standard"])
        final_prompt = prompt_template.format(context=context_str, question=user_question)
        
        # 5. Call LLM
        response = self._call_llm(final_prompt)
        
        return response, citations
        
    def _call_llm(self, prompt: str, retries: int = 2) -> str:
        if self.model_type == "google" and os.getenv("GOOGLE_API_KEY"):
            model = genai.GenerativeModel('gemini-1.5-flash')
            last_error = ""
            for attempt in range(retries):
                try:
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    last_error = str(e)
                    if "504" not in str(e) and "Deadline Exceeded" not in str(e):
                        break # Only retry on timeouts
            return f"⚠️ Error calling Google API after {retries} intentos: {last_error}. Por favor, vuelve a intentar tu pregunta."
        else:
            return "⚠️ LLM is not configured. Please add `GOOGLE_API_KEY` to an `.env` file in the root directory.\n\n**Here is the generated prompt that would have been sent:**\n\n```\n" + prompt + "\n```"
