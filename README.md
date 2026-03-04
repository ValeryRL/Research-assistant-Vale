# Research Copilot: Political Change & Geopolitics in the Middle East

A conversational AI assistant allowing researchers and students to interact with an academic literature catalog covering contemporary political change and geopolitics in the Middle East. It uses local or API-based embeddings and Google's Gemini LLM to answer questions over a curated set of 20 academic papers contextually via RAG.

## Features
- **Chat Interface**: Interact with the assistant in a conversational interface built on Streamlit.
- **Paper Browser**: Easily filter literature over an indexed catalog of 20 detailed academic papers with full metadata.
- **Citation Display**: Every answer relies exclusively on retrieved chunks and explicitly provides citations with authors, publication year, and related excerpts.
- **Prompt Engineering Configuration**: Choose dynamically between Standard, JSON Output, Few-Shot, and Chain-of-Thought prompting strategies.
- **Search Filters**: Restrict the context to only the papers relevant to your specific research queries.

## Architecture
1. **Document Ingestion (`src/ingest.py`)**: PyMuPDF processes the 20 PDFs to extract text, followed by metadata aggregation and token-aware segment chunking (testing both 256 and 1024 token configurations using `tiktoken`).
2. **Vector Store (`src/build_index.py`, `src/retrieval.py`)**: `sentence-transformers` creates dense embeddings which are stored securely along with complex metadata arrays in a persistent `ChromaDB` collection.
3. **Retrieval-Augmented Generation (`src/rag_pipeline.py`)**: Matches user queries via cosine-similarity in the ChromaDB. The results are injected as context into dynamically loaded prompt templates, feeding the Generative AI Model.
4. **App UI (`app.py`)**: A Streamlit interface combining interactive components mapped heavily to the RAG pipeline backend.

## Installation
Ensure Python 3.9+ is installed. Run the following command from the repository root:
```bash
pip install -r requirements.txt
```

Set up your API Key for Gemini via the `.env` file (see the provided `.env.example`).

## Usage
1. Provide the source `papers/` with the original PDFs and the `paper_catalog.json`.
2. Generate the chunks: `python src/ingest.py`
3. Index the chunks to the Vector Database: `python src/build_index.py`
4. Run the UI: `python -m streamlit run app.py`

## Technical Details
- **Chunking Configurations**: Currently evaluating small (256 tokens / 25 overlap) for high precision retrieval and large (1024 tokens / 100 overlap) for comprehensive contextual reasoning.
- **Prompt Strategies**: Included are zero-shot strict compliance, strict JSON formatting, multi-turn equivalent Few-Shot matching, and explicit Chain-of-Thought reasoning.
- **Embeddings**: Employs `sentence-transformers/all-MiniLM-L6-v2` locally for zero-cost and quick iteration embedding generation.
- **Token Usage Estimates**: Small chunk size queries average 1.5k tokens, large chunk queries demand up to 6k tokens of prompt window payload.

## Evaluation Results
(Pending formal query evaluations against Truth metrics using the designated evaluation script). Initial feedback proves Few-Shot limits hallucination well, while Chain-Of-Thought significantly enhances analytical responses.

## Limitations
- **PDF Extraction Artifacts**: Certain PDFs feature two-column layouts or nested graphics wherein `fitz` fails to maintain strictly correct spatial ordering.
- **RAG Myopia**: Highly synthetic queries spanning ideas implicitly written across 5+ papers might lose contextual density if retrieval cuts-off at `k=5`.
- **LLM Context Sizes**: Currently restricted to reasonable chunks to avoid exceeding context window limits of cheaper LLMs.

**Future Improvement**: Implement parent-child retrieval strategies or semantic reranking components like Cohere Rerank to dramatically limit context injection noise, alongside multi-modal PDF OCR logic.

## Author Information
- **Name**: Alexander Quispe (Student / Contributor)
- **Course**: Prompt Engineering
- **Date**: March 2026
