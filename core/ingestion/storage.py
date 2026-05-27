"""
core/ingestion/storage.py

Handles storing chunks in Qdrant vector database.
Each chunk becomes a Qdrant point with:
- id: chunk_id
- vector: embedding of embedding_text
- payload: all chunk fields as JSON
"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from core.schemas.models import Chunk


class QdrantStorage:
    """
    Manages storage and retrieval of chunks in Qdrant.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "pdf_chunks"
    ):
        """
        Initialize Qdrant storage connection.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
    
    def create_collection(self, vector_size: int = 768) -> bool:
        """
        Create a new collection if it doesn't exist.
        
        Args:
            vector_size: Size of embedding vectors (default 768 for many models)
        
        Returns:
            True if created, False if already exists
        """
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
        return False
    
    def store_chunks(
        self,
        chunks: List[Chunk],
        vectors: Optional[List[List[float]]] = None
    ) -> int:
        """
        Store chunks in Qdrant.
        
        Args:
            chunks: List of Chunk objects to store
            vectors: Optional pre-computed embedding vectors.
                    If None, stores with placeholder vectors.
        
        Returns:
            Number of points stored
        """
        points = []
        
        for i, chunk in enumerate(chunks):
            # Use provided vector or placeholder
            vector = vectors[i] if vectors else [0.0] * 768
            
            point = PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.to_dict()
            )
            points.append(point)
        
        # Upsert all points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return len(points)
    
    def get_chunks_by_document(self, document_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            document_name: Name of the document to retrieve
        
        Returns:
            List of chunk payloads
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name)
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        return [point.payload for point in results[0]]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
        
        Returns:
            Chunk payload or None if not found
        """
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[chunk_id],
            with_payload=True
        )
        
        if results:
            return results[0].payload
        return None
    
    def delete_document(self, document_name: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Args:
            document_name: Name of the document to delete
        
        Returns:
            Number of points deleted
        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name)
                    )
                ]
            )
        )
        return result.status.completed_count
    
    def count_chunks(self) -> int:
        """
        Get total number of chunks in the collection.
        """
        result = self.client.count(
            collection_name=self.collection_name
        )
        return result.count
    
    def list_documents(self) -> List[str]:
        """
        List all unique document names in the collection.
        """
        # Scroll through all points and collect unique document names
        documents = set()
        offset = None
        
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["document_name"],
                with_vectors=False
            )
            
            for point in results:
                if point.payload and "document_name" in point.payload:
                    documents.add(point.payload["document_name"])
            
            if next_offset is None:
                break
            offset = next_offset
        
        return list(documents)