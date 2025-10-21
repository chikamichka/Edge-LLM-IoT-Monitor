from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag.rag_llm_pipeline import RAGLLMPipeline
from contextlib import asynccontextmanager
import uvicorn
import os
import json
import asyncio

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global pipeline
    
    # Startup
    print("\n🚀 Starting Edge LLM IoT Monitor API...")
    pipeline = RAGLLMPipeline()
    pipeline.initialize(load_llm=True)
    print("✓ API Ready with Streaming Support!\n")
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down...")
    if pipeline and pipeline.llm_handler.model_loaded:
        pipeline.llm_handler.unload_model()
    print("✓ Cleanup complete")

# Create FastAPI app
app = FastAPI(
    title="Edge LLM IoT Monitor",
    description="AI-powered IoT monitoring with RAG, LLM, and Real-time Streaming",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
if os.path.exists("api/static"):
    app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Request models
class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None
    mode: str = "analysis"
    n_results: int = 5
    max_tokens: int = 200

class QueryResponse(BaseModel):
    query: str
    mode: str
    response: str
    retrieved_documents: int
    metadata: dict

class StatsResponse(BaseModel):
    total_documents: int
    surveillance: int
    agriculture: int

# Dashboard
@app.get("/")
async def root():
    """Serve dashboard"""
    if os.path.exists("api/static/index.html"):
        return FileResponse("api/static/index.html")
    return {
        "service": "Edge LLM IoT Monitor",
        "status": "running",
        "version": "2.0.0",
        "features": ["RAG", "LLM", "Streaming", "LoRA Fine-tuned"],
        "dashboard": "/dashboard"
    }

@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard HTML"""
    return FileResponse("api/static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline and pipeline.initialized:
        return {
            "status": "healthy",
            "rag_initialized": pipeline.rag_system.initialized,
            "llm_loaded": pipeline.llm_handler.model_loaded,
            "features": ["standard", "streaming", "lora-finetuned"]
        }
    return {"status": "initializing"}

@app.post("/query", response_model=QueryResponse)
async def query_sensors(request: QueryRequest):
    """Query IoT sensor data with natural language"""
    
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.query(
            user_query=request.query,
            n_results=request.n_results,
            category_filter=request.category,
            mode=request.mode,
            max_tokens=request.max_tokens
        )
        
        return QueryResponse(
            query=result['query'],
            mode=result['mode'],
            response=result['response'],
            retrieved_documents=result['retrieved_documents'],
            metadata=result['metadata']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
async def stream_query(
    query: str,
    category: Optional[str] = None,
    n_results: int = 5
):
    """Stream query response in real-time (SSE)"""
    
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    async def generate():
        # Step 1: Retrieve
        yield f"data: {json.dumps({'status': 'retrieving', 'message': 'Searching...'})}\n\n"
        await asyncio.sleep(0.1)
        
        docs = pipeline.rag_system.query(query, n_results, category)
        yield f"data: {json.dumps({'status': 'retrieved', 'count': len(docs)})}\n\n"
        
        # Step 2: Generate
        yield f"data: {json.dumps({'status': 'generating', 'message': 'Analyzing...'})}\n\n"
        
        result = pipeline.query(query, n_results, category, "analysis", 150)
        
        # Stream response word by word for demo
        words = result['response'].split()
        for word in words:
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.05)
        
        yield f"data: {json.dumps({'status': 'complete', 'metadata': result['metadata']})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get RAG system statistics"""
    
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = pipeline.rag_system.get_stats()
    return StatsResponse(**stats)

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

# Add monitoring imports
from api.monitoring import monitor

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return monitor.get_metrics()
