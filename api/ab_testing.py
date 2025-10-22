from typing import Dict, List, Optional
import random
import json
from datetime import datetime
from collections import defaultdict
import os

class ABTestingFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self, results_file: str = "ab_test_results.json"):
        self.results_file = results_file
        self.experiments = {}
        self.results = defaultdict(lambda: {
            "variant_a": {"count": 0, "total_time": 0, "successes": 0},
            "variant_b": {"count": 0, "total_time": 0, "successes": 0}
        })
        self.load_results()
    
    def create_experiment(
        self,
        experiment_id: str,
        variant_a: Dict,
        variant_b: Dict,
        traffic_split: float = 0.5
    ):
        """Create a new A/B test experiment"""
        
        self.experiments[experiment_id] = {
            "created_at": datetime.now().isoformat(),
            "variant_a": variant_a,
            "variant_b": variant_b,
            "traffic_split": traffic_split,
            "status": "active"
        }
        
        print(f"✓ Created experiment: {experiment_id}")
        print(f"  Variant A: {variant_a['name']}")
        print(f"  Variant B: {variant_b['name']}")
        print(f"  Traffic split: {traffic_split*100:.0f}% / {(1-traffic_split)*100:.0f}%")
    
    def get_variant(self, experiment_id: str) -> str:
        """Determine which variant to use for this request"""
        
        if experiment_id not in self.experiments:
            return "variant_a"  # Default to A if experiment doesn't exist
        
        experiment = self.experiments[experiment_id]
        
        if experiment["status"] != "active":
            return "variant_a"  # Use A if experiment is not active
        
        # Random assignment based on traffic split
        if random.random() < experiment["traffic_split"]:
            return "variant_a"
        else:
            return "variant_b"
    
    def record_result(
        self,
        experiment_id: str,
        variant: str,
        inference_time: float,
        success: bool = True
    ):
        """Record the result of an experiment"""
        
        self.results[experiment_id][variant]["count"] += 1
        self.results[experiment_id][variant]["total_time"] += inference_time
        if success:
            self.results[experiment_id][variant]["successes"] += 1
        
        # Save periodically (every 10 requests)
        if self.results[experiment_id][variant]["count"] % 10 == 0:
            self.save_results()
    
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get results for an experiment"""
        
        if experiment_id not in self.results:
            return {"error": "Experiment not found"}
        
        results = self.results[experiment_id]
        
        # Calculate metrics for variant A
        a_count = results["variant_a"]["count"]
        a_avg_time = results["variant_a"]["total_time"] / a_count if a_count > 0 else 0
        a_success_rate = results["variant_a"]["successes"] / a_count if a_count > 0 else 0
        
        # Calculate metrics for variant B
        b_count = results["variant_b"]["count"]
        b_avg_time = results["variant_b"]["total_time"] / b_count if b_count > 0 else 0
        b_success_rate = results["variant_b"]["successes"] / b_count if b_count > 0 else 0
        
        # Calculate improvements
        time_improvement = ((a_avg_time - b_avg_time) / a_avg_time * 100) if a_avg_time > 0 else 0
        success_improvement = ((b_success_rate - a_success_rate) * 100) if a_count > 0 else 0
        
        # Determine winner
        winner = None
        if a_count >= 30 and b_count >= 30:  # Minimum sample size
            if b_avg_time < a_avg_time * 0.9 and b_success_rate >= a_success_rate * 0.95:
                winner = "variant_b"
            elif a_avg_time < b_avg_time * 0.9 and a_success_rate >= b_success_rate * 0.95:
                winner = "variant_a"
        
        return {
            "experiment_id": experiment_id,
            "experiment": self.experiments.get(experiment_id, {}),
            "variant_a": {
                "requests": a_count,
                "avg_inference_time": round(a_avg_time, 3),
                "success_rate": round(a_success_rate * 100, 2)
            },
            "variant_b": {
                "requests": b_count,
                "avg_inference_time": round(b_avg_time, 3),
                "success_rate": round(b_success_rate * 100, 2)
            },
            "comparison": {
                "time_improvement_pct": round(time_improvement, 2),
                "success_improvement_pct": round(success_improvement, 2),
                "winner": winner,
                "confidence": "high" if (a_count >= 100 and b_count >= 100) else "medium" if (a_count >= 30 and b_count >= 30) else "low"
            }
        }
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments"""
        return [
            {
                "experiment_id": exp_id,
                **self.get_experiment_results(exp_id)
            }
            for exp_id in self.experiments.keys()
        ]
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = "stopped"
            self.save_results()
            print(f"✓ Stopped experiment: {experiment_id}")
    
    def save_results(self):
        """Save results to file"""
        data = {
            "experiments": self.experiments,
            "results": dict(self.results)
        }
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_results(self):
        """Load results from file"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                self.experiments = data.get("experiments", {})
                self.results = defaultdict(lambda: {
                    "variant_a": {"count": 0, "total_time": 0, "successes": 0},
                    "variant_b": {"count": 0, "total_time": 0, "successes": 0}
                }, data.get("results", {}))


# Global A/B testing instance
ab_tester = ABTestingFramework()

# Create default experiments
ab_tester.create_experiment(
    experiment_id="standard_vs_multiagent",
    variant_a={
        "name": "Standard RAG-LLM",
        "description": "Single model inference with RAG",
        "endpoint": "/query"
    },
    variant_b={
        "name": "Multi-Agent System",
        "description": "Multiple specialized agents with synthesis",
        "endpoint": "/multi-agent-query"
    },
    traffic_split=0.7  # 70% to standard, 30% to multi-agent
)

ab_tester.create_experiment(
    experiment_id="base_vs_finetuned",
    variant_a={
        "name": "Base Model",
        "description": "Qwen 2.5 1.5B base model",
        "model": "base"
    },
    variant_b={
        "name": "LoRA Fine-tuned",
        "description": "Fine-tuned with IoT domain data",
        "model": "lora"
    },
    traffic_split=0.5  # 50/50 split
)