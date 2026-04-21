import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import os
import pickle
from src.services.embedding_service import EmbeddingService

class VectorStore:
    def __init__(self, index_name="log_index", config_path="config.yaml"):
        self.embedding_service = EmbeddingService(config_path)
        self.index_name = index_name
        self.index = None
        self.metadata = []
        self.load_index()
    
    def load_index(self):
        """Load existing FAISS index or create new one"""
        index_path = os.path.join(self.embedding_service.vector_store_path, f"{self.index_name}.faiss")
        meta_path = os.path.join(self.embedding_service.vector_store_path, f"{self.index_name}_meta.pkl")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Create new empty index
            self.index = None
            self.metadata = []
    
    def add_documents(self, texts: List[str], metadatas: List[Dict]):
        """Add documents to vector store"""
        if not texts:
            return
        
        embeddings = self.embedding_service.encode(texts)
        
        if self.index is None:
            # Create new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadatas)
        
        self.save_index()
    
    def save_index(self):
        """Save index and metadata"""
        if self.index:
            index_path = os.path.join(self.embedding_service.vector_store_path, f"{self.index_name}.faiss")
            meta_path = os.path.join(self.embedding_service.vector_store_path, f"{self.index_name}_meta.pkl")
            
            faiss.write_index(self.index, index_path)
            with open(meta_path, 'wb') as f:
                pickle.dump(self.metadata, f)
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Dict = None) -> List[Tuple[float, Dict]]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_service.encode_single(query).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                # Apply metadata filtering if provided
                if filter_metadata:
                    match = all(
                        self.metadata[idx].get(key) == value 
                        for key, value in filter_metadata.items()
                    )
                    if not match:
                        continue
                
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + dist)
                results.append((similarity, self.metadata[idx]))
        
        return sorted(results, key=lambda x: x[0], reverse=True)
    
    def size(self):
        """Get number of documents in index"""
        return self.index.ntotal if self.index else 0