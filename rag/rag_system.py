import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from models.config import rag_config
import os

class RAGSystem:
    """RAG system for IoT sensor data querying"""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.initialized = False
        
    def initialize(self):
        """Initialize embedding model and ChromaDB"""
        print(f"Initializing RAG system...")
        
        # Load embedding model
        print(f"Loading embeddings: {rag_config.embedding_model}")
        self.embedding_model = SentenceTransformer(rag_config.embedding_model)
        
        # Initialize ChromaDB
        os.makedirs(rag_config.chroma_persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=rag_config.chroma_persist_dir
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=rag_config.collection_name,
            metadata={"description": "IoT sensor data embeddings"}
        )
        
        self.initialized = True
        print(f"✓ RAG system initialized")
        print(f"✓ Collection: {rag_config.collection_name}")
        print(f"✓ Documents in collection: {self.collection.count()}")
    
    def _format_sensor_document(self, record: Dict) -> str:
        """Convert sensor record to text document"""
        
        doc = f"Timestamp: {record['timestamp']}\n"
        doc += f"Category: {record['category']}\n"
        doc += f"Sensor Type: {record['sensor_type']}\n"
        doc += f"Device ID: {record['device_id']}\n"
        doc += "Data:\n"
        
        for key, value in record['data'].items():
            doc += f"  - {key}: {value}\n"
        
        return doc
    
    def ingest_data(self, json_file: str = "data/sensor_data.json"):
        """Ingest sensor data into ChromaDB"""
        
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        print(f"\nIngesting data from {json_file}...")
        
        # Load data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Prepare documents
        documents = []
        metadatas = []
        ids = []
        
        for idx, record in enumerate(data):
            # Create text document
            doc_text = self._format_sensor_document(record)
            documents.append(doc_text)
            
            # Metadata for filtering
            metadatas.append({
                "category": record['category'],
                "sensor_type": record['sensor_type'],
                "device_id": record['device_id'],
                "timestamp": record['timestamp']
            })
            
            ids.append(f"sensor_{idx}")
        
        # Batch insert
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            print(f"  Processed {min(i+batch_size, len(documents))}/{len(documents)} records")
        
        print(f"✓ Ingested {len(documents)} documents")
        print(f"✓ Total documents in collection: {self.collection.count()}")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """Query the RAG system"""
        
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        # Build filter
        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        
        if not self.initialized:
            return {"error": "Not initialized"}
        
        total = self.collection.count()
        
        # Count by category (simple approach)
        all_data = self.collection.get()
        
        surveillance_count = sum(1 for m in all_data['metadatas'] if m['category'] == 'surveillance')
        agriculture_count = sum(1 for m in all_data['metadatas'] if m['category'] == 'agriculture')
        
        return {
            "total_documents": total,
            "surveillance": surveillance_count,
            "agriculture": agriculture_count
        }


# Test the RAG system
if __name__ == "__main__":
    rag = RAGSystem()
    rag.initialize()
    
    # Ingest data
    rag.ingest_data()
    
    # Get stats
    stats = rag.get_stats()
    print(f"\n{'='*50}")
    print("RAG System Stats:")
    print(f"{'='*50}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Surveillance: {stats['surveillance']}")
    print(f"Agriculture: {stats['agriculture']}")
    
    # Test queries
    print(f"\n{'='*50}")
    print("Test Query: Motion detection events")
    print(f"{'='*50}")
    
    results = rag.query("motion detection in parking area", n_results=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Category: {result['metadata']['category']}")
        print(f"Sensor: {result['metadata']['sensor_type']}")
        print(f"Device: {result['metadata']['device_id']}")
        print(f"Relevance Score: {1 - result['distance']:.2f}")