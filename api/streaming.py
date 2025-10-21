from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio
from models.llm_handler import LLMHandler
from rag.rag_system import RAGSystem
import torch

class StreamingPipeline:
    """Handle streaming responses for real-time inference"""
    
    def __init__(self, llm_handler: LLMHandler, rag_system: RAGSystem):
        self.llm_handler = llm_handler
        self.rag_system = rag_system
    
    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 200
    ) -> AsyncGenerator[str, None]:
        """Stream tokens as they're generated"""
        
        if not self.llm_handler.model_loaded:
            yield json.dumps({"error": "Model not loaded"})
            return
        
        # Tokenize input
        inputs = self.llm_handler.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.llm_handler.device)
        
        # Stream generation
        with torch.no_grad():
            for i in range(max_new_tokens):
                outputs = self.llm_handler.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_handler.tokenizer.pad_token_id
                )
                
                # Get new token
                new_token_id = outputs[0][-1].item()
                new_token = self.llm_handler.tokenizer.decode([new_token_id])
                
                # Check for end of sequence
                if new_token_id == self.llm_handler.tokenizer.eos_token_id:
                    break
                
                # Yield token
                yield json.dumps({"token": new_token}) + "\n"
                
                # Update inputs for next iteration
                inputs = {
                    'input_ids': outputs,
                    'attention_mask': torch.ones_like(outputs)
                }
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
        
        # Send completion signal
        yield json.dumps({"done": True}) + "\n"
    
    async def stream_rag_query(
        self,
        user_query: str,
        category: str = None,
        n_results: int = 5
    ) -> AsyncGenerator[str, None]:
        """Stream RAG query with progressive results"""
        
        # Step 1: Retrieve documents
        yield json.dumps({
            "status": "retrieving",
            "message": "Searching sensor data..."
        }) + "\n"
        
        await asyncio.sleep(0.1)
        
        retrieved_docs = self.rag_system.query(
            user_query,
            n_results=n_results,
            category_filter=category
        )
        
        yield json.dumps({
            "status": "retrieved",
            "count": len(retrieved_docs),
            "message": f"Found {len(retrieved_docs)} relevant documents"
        }) + "\n"
        
        # Step 2: Build context
        yield json.dumps({
            "status": "building_context",
            "message": "Analyzing sensor data..."
        }) + "\n"
        
        context = self._build_context(retrieved_docs)
        prompt = self._build_prompt(user_query, context)
        
        # Step 3: Stream generation
        yield json.dumps({
            "status": "generating",
            "message": "Generating response..."
        }) + "\n"
        
        async for chunk in self.stream_generate(prompt, max_new_tokens=200):
            yield chunk
    
    def _build_context(self, docs):
        """Build context from documents"""
        context = "Relevant IoT Sensor Data:\n\n"
        for i, doc in enumerate(docs, 1):
            context += f"--- Record {i} ---\n{doc['document']}\n"
        return context
    
    def _build_prompt(self, query, context):
        """Build prompt for LLM"""
        return f"""You are an IoT monitoring assistant. Analyze the sensor data and respond concisely.

{context}

User Query: {query}

Analysis:"""


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()