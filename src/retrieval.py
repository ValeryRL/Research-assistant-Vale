import os
import chromadb
from chromadb.config import Settings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("Please install sentence-transformers to use local embeddings: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        # Generate embeddings in batches if necessary, but sentence-transformers handles lists natively
        embeddings = self.model.encode(documents, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None

    def create_collection(self, name: str):
        self.collection = self.client.get_or_create_collection(
            name=name, 
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]
    ):
        # Flatten metadata to make it compatible with ChromaDB
        flattened_metadatas = []
        for meta in metadatas:
            flat = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    flat[k] = v
                elif isinstance(v, list):
                    flat[k] = ", ".join(map(str, v))
                else:
                    flat[k] = str(v)
            flattened_metadatas.append(flat)
            
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=flattened_metadatas
        )

    def query(
        self, query_embedding: list[float], n_results: int = 5, where: dict = None
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
