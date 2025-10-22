from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rag.rag_llm_pipeline import RAGLLMPipeline
from agents.multi_agent_system import MultiAgentSystem
from api.monitoring import monitor
from contextlib import asynccontextmanager
import uvicorn
import os
import json
import asyncio
import time

# Global instances
pipeline = None
multi_agent_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global pipeline, multi_agent_system
    
    # Startup
    print("\n🚀 Starting Edge LLM IoT Monitor API...")
    pipeline = RAGLLMPipeline()
    pipeline.initialize(load_llm=True)
    
    # Initialize multi-agent system
    multi_agent_system = MultiAgentSystem(
        pipeline.llm_handler,
        pipeline.rag_system
    )
    print("✓ Multi-Agent System initialized")
    print("✓ API Ready with Full Features!\n")
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down...")
    if pipeline and pipeline.llm_handler.model_loaded:
        pipeline.llm_handler.unload_model()
    print("✓ Cleanup complete")

# Create FastAPI app
app = FastAPI(
    title="Edge LLM IoT Monitor",
    description="AI-powered IoT monitoring with RAG, LLM, Multi-Agent, and Real-time Streaming",
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
        "features": ["RAG", "LLM", "Streaming", "Multi-Agent", "LoRA Fine-tuned"],
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline and pipeline.initialized:
        return {
            "status": "healthy",
            "rag_initialized": pipeline.rag_system.initialized,
            "llm_loaded": pipeline.llm_handler.model_loaded,
            "multi_agent_ready": multi_agent_system is not None,
            "features": ["standard", "streaming", "multi-agent", "lora-finetuned"]
        }
    return {"status": "initializing"}

@app.post("/query", response_model=QueryResponse)
async def query_sensors(request: QueryRequest):
    """Query IoT sensor data with natural language"""
    
    if not pipeline or not pipeline.initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start_time = time.time()
        
        result = pipeline.query(
            user_query=request.query,
            n_results=request.n_results,
            category_filter=request.category,
            mode=request.mode,
            max_tokens=request.max_tokens
        )
        
        monitor.record_query(time.time() - start_time, success=True)
        
        return QueryResponse(
            query=result['query'],
            mode=result['mode'],
            response=result['response'],
            retrieved_documents=result['retrieved_documents'],
            metadata=result['metadata']
        )
    
    except Exception as e:
        monitor.record_query(0, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-agent-query")
async def multi_agent_query(request: QueryRequest):
    """Query using multi-agent system for complex analysis"""
    
    if not multi_agent_system:
        raise HTTPException(status_code=503, detail="Multi-agent system not initialized")
    
    try:
        start_time = time.time()
        
        result = multi_agent_system.analyze_with_agents(
            query=request.query,
            category=request.category,
            n_results=request.n_results
        )
        
        monitor.record_query(time.time() - start_time, success=True)
        
        return {
            "query": result['query'],
            "mode": "multi-agent",
            "agents_used": result['agents_used'],
            "agent_analyses": result['agent_analyses'],
            "final_response": result['final_synthesis'],
            "retrieved_documents": result['retrieved_documents'],
            "metadata": result['metadata']
        }
    
    except Exception as e:
        monitor.record_query(0, success=False)
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
        
        # Stream response word by word
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

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return monitor.get_metrics()

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

# Add A/B testing import
from api.ab_testing import ab_tester

# A/B Testing Endpoints
@app.post("/ab-test-query")
async def ab_test_query(request: QueryRequest):
    """Query with automatic A/B testing between standard and multi-agent"""
    
    experiment_id = "standard_vs_multiagent"
    variant = ab_tester.get_variant(experiment_id)
    
    try:
        start_time = time.time()
        
        if variant == "variant_a":
            # Standard query
            result = pipeline.query(
                user_query=request.query,
                n_results=request.n_results,
                category_filter=request.category,
                mode=request.mode,
                max_tokens=request.max_tokens
            )
            response = {
                "variant": "standard",
                "query": result['query'],
                "response": result['response'],
                "metadata": result['metadata']
            }
        else:
            # Multi-agent query
            result = multi_agent_system.analyze_with_agents(
                query=request.query,
                category=request.category,
                n_results=request.n_results
            )
            response = {
                "variant": "multi-agent",
                "query": result['query'],
                "agents_used": result['agents_used'],
                "response": result['final_synthesis'],
                "metadata": result['metadata']
            }
        
        inference_time = time.time() - start_time
        ab_tester.record_result(experiment_id, variant, inference_time, success=True)
        
        return response
        
    except Exception as e:
        ab_tester.record_result(experiment_id, variant, 0, success=False)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-tests")
async def list_ab_tests():
    """List all A/B tests and their results"""
    return ab_tester.list_experiments()

@app.get("/ab-test/{experiment_id}")
async def get_ab_test_results(experiment_id: str):
    """Get results for a specific A/B test"""
    return ab_tester.get_experiment_results(experiment_id)

@app.post("/ab-test/{experiment_id}/stop")
async def stop_ab_test(experiment_id: str):
    """Stop an A/B test"""
    ab_tester.stop_experiment(experiment_id)
    return {"status": "stopped", "experiment_id": experiment_id}
