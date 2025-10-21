from typing import Dict, List, Optional
from models.llm_handler import LLMHandler
from rag.rag_system import RAGSystem
import json

class RAGLLMPipeline:
    """Combines RAG retrieval with LLM generation for IoT monitoring"""
    
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.rag_system = RAGSystem()
        self.initialized = False
    
    def initialize(self, load_llm: bool = True):
        """Initialize both RAG and LLM systems"""
        print("Initializing RAG-LLM Pipeline...")
        
        # Initialize RAG
        self.rag_system.initialize()
        
        # Load LLM
        if load_llm:
            self.llm_handler.load_model()
        
        self.initialized = True
        print("✓ Pipeline ready!\n")
    
    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """Build context from retrieved documents"""
        
        context = "Relevant IoT Sensor Data:\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"--- Record {i} ---\n"
            context += doc['document']
            context += "\n"
        
        return context
    
    def _build_prompt(self, query: str, context: str, mode: str = "analysis") -> str:
        """Build prompt for LLM based on mode"""
        
        prompts = {
            "analysis": f"""You are an IoT monitoring assistant analyzing sensor data.

{context}

User Query: {query}

Provide a concise analysis of the sensor data relevant to the query. Focus on:
1. Key findings
2. Any anomalies or alerts
3. Recommendations if needed

Analysis:""",
            
            "alert": f"""You are an IoT alert system. Analyze the sensor data and determine if there are any issues.

{context}

Query: {query}

Determine if this situation requires attention. Respond with:
1. Alert Status (Normal/Warning/Critical)
2. Reason
3. Recommended Action (if needed)

Response:""",
            
            "summary": f"""You are an IoT data summarizer. Provide a brief summary of the sensor readings.

{context}

Summarize the above sensor data in 2-3 sentences, highlighting the most important information.

Summary:"""
        }
        
        return prompts.get(mode, prompts["analysis"])
    
    def query(
        self,
        user_query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        mode: str = "analysis",
        max_tokens: int = 200
    ) -> Dict:
        """
        Query the RAG-LLM pipeline
        
        Args:
            user_query: Natural language query
            n_results: Number of documents to retrieve
            category_filter: Filter by 'surveillance' or 'agriculture'
            mode: Response mode ('analysis', 'alert', 'summary')
            max_tokens: Max tokens to generate
        """
        
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized")
        
        # Step 1: Retrieve relevant documents
        print(f"🔍 Retrieving relevant sensor data...")
        retrieved_docs = self.rag_system.query(
            user_query,
            n_results=n_results,
            category_filter=category_filter
        )
        
        print(f"✓ Retrieved {len(retrieved_docs)} documents\n")
        
        # Step 2: Build context and prompt
        context = self._build_context(retrieved_docs)
        prompt = self._build_prompt(user_query, context, mode)
        
        # Step 3: Generate response with LLM
        print(f"🤖 Generating response...")
        llm_result = self.llm_handler.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7
        )
        
        print(f"✓ Generated in {llm_result['inference_time']:.2f}s\n")
        
        # Return complete result
        return {
            "query": user_query,
            "mode": mode,
            "retrieved_documents": len(retrieved_docs),
            "context": context,
            "response": llm_result['generated_text'],
            "metadata": {
                "inference_time": llm_result['inference_time'],
                "tokens_generated": llm_result['tokens_generated'],
                "tokens_per_second": llm_result['tokens_per_second']
            }
        }
    
    def analyze_anomalies(self, category: Optional[str] = None) -> Dict:
        """Detect anomalies in sensor data"""
        
        query = "Find unusual readings, alerts, or anomalies in the sensor data"
        return self.query(
            query,
            n_results=10,
            category_filter=category,
            mode="alert",
            max_tokens=150
        )
    
    def get_summary(self, category: Optional[str] = None) -> Dict:
        """Get summary of recent sensor activity"""
        
        query = "Summarize recent sensor activity and status"
        return self.query(
            query,
            n_results=8,
            category_filter=category,
            mode="summary",
            max_tokens=100
        )


# Test the pipeline
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGLLMPipeline()
    pipeline.initialize()
    
    # Test queries
    test_queries = [
        {
            "query": "Show me motion detection events in the parking area",
            "category": "surveillance",
            "mode": "analysis"
        },
        {
            "query": "Are there any temperature or humidity alerts in agriculture sensors?",
            "category": "agriculture",
            "mode": "alert"
        },
        {
            "query": "What's the overall status of surveillance cameras?",
            "category": "surveillance",
            "mode": "summary"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST QUERY {i}")
        print(f"{'='*60}")
        print(f"Query: {test['query']}")
        print(f"Category: {test['category']}")
        print(f"Mode: {test['mode']}\n")
        
        result = pipeline.query(
            test['query'],
            category_filter=test['category'],
            mode=test['mode'],
            n_results=3
        )
        
        print(f"{'─'*60}")
        print("RESPONSE:")
        print(f"{'─'*60}")
        print(result['response'])
        print(f"\n📊 Metadata:")
        print(f"  - Documents retrieved: {result['retrieved_documents']}")
        print(f"  - Inference time: {result['metadata']['inference_time']:.2f}s")
        print(f"  - Speed: {result['metadata']['tokens_per_second']:.2f} tok/s")
    
    # Cleanup
    pipeline.llm_handler.unload_model()
    print(f"\n{'='*60}")
    print("✓ Pipeline test complete!")