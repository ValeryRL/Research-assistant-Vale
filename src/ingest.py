import os
import re
import fitz  # PyMuPDF
import tiktoken
import json

def extract_text_from_pdf(pdf_path: str) -> dict:
    """ Extract text and metadata from a PDF file. """
    doc = fitz.open(pdf_path)
    full_text = ""
    pages = []
    warnings = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text)
        })
        full_text += f"\n[PAGE {page_num + 1}]\n{text}"
    
    metadata = doc.metadata
    doc.close()
    return {
        "text": full_text,
        "metadata": metadata,
        "pages": pages,
        "total_pages": len(pages),
        "extraction_warnings": warnings
    }

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Remove page numbers and headers (customize per document)
    text = re.sub(r'\n\d+\n', '\n', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    return text.strip()

class TokenChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, model: str = "gpt-4"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.encoding_for_model(model)

    def chunk_text(self, text: str, metadata: dict = None) -> list[dict]:
        """Split text into overlapping chunks."""
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
                "metadata": metadata or {}
            })
            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1
            
        return chunks

def process_document(pdf_path: str, paper_metadata: dict, chunker_small: TokenChunker, chunker_large: TokenChunker):
    """Extract, clean, and chunk a document into small and large configurations."""
    extraction_result = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_extracted_text(extraction_result["text"])
    
    # Add extraction stats to metadata
    doc_metadata = {
        **paper_metadata,
        "total_pages": extraction_result["total_pages"],
        "source": os.path.basename(pdf_path)
    }
    
    small_chunks = chunker_small.chunk_text(cleaned_text, doc_metadata)
    large_chunks = chunker_large.chunk_text(cleaned_text, doc_metadata)
    
    return small_chunks, large_chunks

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    papers_dir = os.path.join(project_root, "papers")
    catalog_path = os.path.join(papers_dir, "paper_catalog.json")
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)["papers"]
        
    chunker_small = TokenChunker(chunk_size=256, chunk_overlap=25)
    chunker_large = TokenChunker(chunk_size=1024, chunk_overlap=100)
    
    all_small_chunks = []
    all_large_chunks = []
    
    for paper in catalog:
        pdf_path = os.path.join(papers_dir, paper["filename"])
        print(f"Processing {paper['filename']}...")
        if os.path.exists(pdf_path):
            small, large = process_document(pdf_path, paper, chunker_small, chunker_large)
            all_small_chunks.extend(small)
            all_large_chunks.extend(large)
        else:
            print(f"File not found: {pdf_path}")
            
    print(f"Total small chunks generated: {len(all_small_chunks)}")
    print(f"Total large chunks generated: {len(all_large_chunks)}")
    
    cache_dir = os.path.join(project_root, "data")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save the chunks to disk for later retrieval to avoid re-processing PDFs
    with open(os.path.join(cache_dir, "small_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_small_chunks, f)
    with open(os.path.join(cache_dir, "large_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_large_chunks, f)
        
    print("Chunks cached in data/ directory.")
