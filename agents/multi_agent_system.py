from typing import Dict, List, Optional
from models.llm_handler import LLMHandler
from rag.rag_system import RAGSystem
import json

class Agent:
    """Base agent class"""
    
    def __init__(self, name: str, role: str, llm_handler: LLMHandler):
        self.name = name
        self.role = role
        self.llm_handler = llm_handler
    
    def analyze(self, context: str, query: str) -> Dict:
        """Analyze with agent's perspective"""
        
        prompt = f"""You are a {self.role}.

Context:
{context}

Task: {query}

Provide your specialized analysis in 2-3 sentences:"""
        
        result = self.llm_handler.generate(prompt, max_new_tokens=100)
        
        return {
            "agent": self.name,
            "role": self.role,
            "analysis": result['generated_text'],
            "inference_time": result['inference_time']
        }


class SecurityAgent(Agent):
    """Specialized in surveillance and security analysis"""
    
    def __init__(self, llm_handler: LLMHandler):
        super().__init__(
            name="SecurityAgent",
            role="Security Expert specialized in surveillance systems, access control, and threat detection",
            llm_handler=llm_handler
        )


class AgricultureAgent(Agent):
    """Specialized in agriculture and environmental analysis"""
    
    def __init__(self, llm_handler: LLMHandler):
        super().__init__(
            name="AgricultureAgent",
            role="Agriculture Expert specialized in crop health, irrigation, and environmental monitoring",
            llm_handler=llm_handler
        )


class AnomalyAgent(Agent):
    """Specialized in anomaly detection and pattern analysis"""
    
    def __init__(self, llm_handler: LLMHandler):
        super().__init__(
            name="AnomalyAgent",
            role="Anomaly Detection Expert specialized in identifying unusual patterns and potential system failures",
            llm_handler=llm_handler
        )


class CoordinatorAgent(Agent):
    """Coordinates other agents and synthesizes their outputs"""
    
    def __init__(self, llm_handler: LLMHandler):
        super().__init__(
            name="CoordinatorAgent",
            role="System Coordinator that synthesizes insights from multiple agents",
            llm_handler=llm_handler
        )
    
    def synthesize(self, agent_results: List[Dict], original_query: str) -> Dict:
        """Synthesize results from multiple agents"""
        
        # Build synthesis prompt
        analyses = "\n\n".join([
            f"{r['agent']} ({r['role']}):\n{r['analysis']}"
            for r in agent_results
        ])
        
        prompt = f"""You are coordinating multiple expert agents. Synthesize their analyses into a comprehensive response.

Original Query: {original_query}

Agent Analyses:
{analyses}

Provide a unified, actionable response that combines the best insights from all agents:"""
        
        result = self.llm_handler.generate(prompt, max_new_tokens=150)
        
        return {
            "synthesis": result['generated_text'],
            "inference_time": result['inference_time'],
            "agents_consulted": len(agent_results)
        }


class MultiAgentSystem:
    """Multi-agent system for complex IoT analysis"""
    
    def __init__(self, llm_handler: LLMHandler, rag_system: RAGSystem):
        self.llm_handler = llm_handler
        self.rag_system = rag_system
        
        # Initialize agents
        self.security_agent = SecurityAgent(llm_handler)
        self.agriculture_agent = AgricultureAgent(llm_handler)
        self.anomaly_agent = AnomalyAgent(llm_handler)
        self.coordinator = CoordinatorAgent(llm_handler)
        
        self.all_agents = {
            "security": self.security_agent,
            "agriculture": self.agriculture_agent,
            "anomaly": self.anomaly_agent
        }
    
    def _determine_relevant_agents(self, query: str, category: Optional[str]) -> List[Agent]:
        """Determine which agents should analyze the query"""
        
        relevant_agents = []
        
        query_lower = query.lower()
        
        # Always include anomaly agent
        relevant_agents.append(self.anomaly_agent)
        
        # Category-based selection
        if category == "surveillance" or any(word in query_lower for word in ["motion", "camera", "door", "access", "security"]):
            relevant_agents.append(self.security_agent)
        
        if category == "agriculture" or any(word in query_lower for word in ["temperature", "humidity", "soil", "moisture", "crop"]):
            relevant_agents.append(self.agriculture_agent)
        
        # If no specific category, use both domain agents
        if not category and len(relevant_agents) == 1:
            relevant_agents.extend([self.security_agent, self.agriculture_agent])
        
        return relevant_agents
    
    def analyze_with_agents(
        self,
        query: str,
        category: Optional[str] = None,
        n_results: int = 5
    ) -> Dict:
        """Analyze query using multi-agent system"""
        
        print(f"\n🤖 Multi-Agent Analysis Started")
        print(f"Query: {query}")
        print("="*60)
        
        # Step 1: Retrieve relevant documents
        print("\n�� Retrieving relevant sensor data...")
        docs = self.rag_system.query(query, n_results, category)
        
        # Build context
        context = self._build_context(docs)
        print(f"✓ Retrieved {len(docs)} documents")
        
        # Step 2: Determine relevant agents
        relevant_agents = self._determine_relevant_agents(query, category)
        print(f"\n🎯 Consulting {len(relevant_agents)} specialized agents:")
        for agent in relevant_agents:
            print(f"  - {agent.name}")
        
        # Step 3: Get analyses from each agent
        print("\n🔍 Agent Analyses:")
        agent_results = []
        
        for agent in relevant_agents:
            print(f"\n  {agent.name} analyzing...")
            result = agent.analyze(context, query)
            agent_results.append(result)
            print(f"  ✓ {agent.name}: {result['inference_time']:.2f}s")
        
        # Step 4: Coordinator synthesizes results
        print(f"\n🎭 Coordinator synthesizing insights...")
        synthesis = self.coordinator.synthesize(agent_results, query)
        print(f"  ✓ Synthesis complete: {synthesis['inference_time']:.2f}s")
        
        # Calculate total time
        total_time = sum(r['inference_time'] for r in agent_results) + synthesis['inference_time']
        
        print(f"\n✓ Multi-Agent Analysis Complete!")
        print(f"  Total time: {total_time:.2f}s")
        print("="*60)
        
        return {
            "query": query,
            "retrieved_documents": len(docs),
            "agents_used": [r['agent'] for r in agent_results],
            "agent_analyses": agent_results,
            "final_synthesis": synthesis['synthesis'],
            "metadata": {
                "total_inference_time": total_time,
                "agents_consulted": len(agent_results),
                "documents_retrieved": len(docs)
            }
        }
    
    def _build_context(self, docs: List[Dict]) -> str:
        """Build context from retrieved documents"""
        context = "Relevant IoT Sensor Data:\n\n"
        for i, doc in enumerate(docs, 1):
            context += f"Record {i}:\n{doc['document']}\n"
        return context


# Test the multi-agent system
if __name__ == "__main__":
    from models.llm_handler import LLMHandler
    from rag.rag_system import RAGSystem
    
    print("="*60)
    print("Multi-Agent System Demo")
    print("="*60)
    
    # Initialize components
    llm_handler = LLMHandler()
    llm_handler.load_model()
    
    rag_system = RAGSystem()
    rag_system.initialize()
    
    # Create multi-agent system
    mas = MultiAgentSystem(llm_handler, rag_system)
    
    # Test queries
    test_queries = [
        {
            "query": "Are there any security concerns with motion detection and door access?",
            "category": "surveillance"
        },
        {
            "query": "Analyze temperature and humidity levels for crop health",
            "category": "agriculture"
        },
        {
            "query": "Detect any anomalies across all systems",
            "category": None
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n\n{'#'*60}")
        print(f"TEST {i}")
        print(f"{'#'*60}")
        
        result = mas.analyze_with_agents(
            test['query'],
            test['category']
        )
        
        print(f"\n{'─'*60}")
        print("AGENT ANALYSES:")
        print(f"{'─'*60}")
        for analysis in result['agent_analyses']:
            print(f"\n{analysis['agent']}:")
            print(analysis['analysis'])
        
        print(f"\n{'─'*60}")
        print("FINAL SYNTHESIS:")
        print(f"{'─'*60}")
        print(result['final_synthesis'])
        
        print(f"\n📊 Metadata:")
        print(f"  Agents: {', '.join(result['agents_used'])}")
        print(f"  Total time: {result['metadata']['total_inference_time']:.2f}s")
        print(f"  Documents: {result['metadata']['documents_retrieved']}")
    
    # Cleanup
    llm_handler.unload_model()
    print("\n✓ Demo complete!")
