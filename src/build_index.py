import json
import os
try:
    from .retrieval import ChromaVectorStore, EmbeddingModel
except ImportError:
    from retrieval import ChromaVectorStore, EmbeddingModel

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    chunks_file = os.path.join(project_root, "data", "small_chunks.json")
    db_path = os.path.join(project_root, "chroma_db")
    
    if not os.path.exists(chunks_file):
        print(f"Chunks file not found at {chunks_file}. Generating them now...")
        try:
            from .ingest import main as ingest_main
            ingest_main()
        except ImportError:
            try:
                from ingest import main as ingest_main
                ingest_main()
            except ImportError:
                from src.ingest import main as ingest_main
                ingest_main()
            
    if not os.path.exists(chunks_file):
        print("Fatal error: Chunks still not generated.")
        return
        
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} text chunks.")
    
    model = EmbeddingModel("all-MiniLM-L6-v2")
    store = ChromaVectorStore(persist_directory=db_path)
    store.create_collection("papers_small_chunks")
    
    # Process in batches to avoid memory issues and print progress
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]
        ids = [f"{chunk['metadata']['id']}_chunk_{chunk['chunk_id']}" for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        
        print(f"Processing batch {i//batch_size + 1}/{total_batches}...")
        embeddings = model.embed_documents(texts)
        
        store.add_documents(ids, texts, embeddings, metadatas)
        
    print("Indexing complete. The vector database is ready.")

if __name__ == "__main__":
    main()
