from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for LLM model"""
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    device: str = os.getenv("DEVICE", "mps")
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "iot_sensors")
    chunk_size: int = 200
    chunk_overlap: int = 50

@dataclass
class APIConfig:
    """Configuration for API"""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))

# Global config instances
model_config = ModelConfig()
rag_config = RAGConfig()
api_config = APIConfig()