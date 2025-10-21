import json
from typing import List, Dict
import random

class IoTTrainingDataGenerator:
    """Generate training data for fine-tuning on IoT domain"""
    
    def __init__(self):
        self.surveillance_scenarios = [
            {
                "instruction": "Analyze this surveillance sensor reading",
                "context": "Motion detected: True, Zone: parking, Confidence: 0.95, Device: CAM_101",
                "response": "High-confidence motion detected in parking area by CAM_101. This indicates active movement in the monitored zone. Recommend verifying the activity and checking recorded footage for security assessment."
            },
            {
                "instruction": "Interpret this access control event",
                "context": "Door status: open, Access granted: False, User ID: USER_5432, Device: DOOR_203",
                "response": "Unauthorized access attempt detected at DOOR_203 by USER_5432. Access was denied but door status shows open, indicating a potential security breach. Immediate investigation required and security team should be notified."
            },
            {
                "instruction": "Evaluate camera system status",
                "context": "Camera status: recording, FPS: 30, Resolution: 4K, Device: CAM_045",
                "response": "CAM_045 is operating optimally with 4K resolution at 30 FPS in recording mode. System performance is excellent for high-quality surveillance monitoring. No maintenance required."
            },
            {
                "instruction": "Assess window sensor alert",
                "context": "Window status: open, Break detected: True, Device: WIN_012",
                "response": "Critical security alert: Window break detected at WIN_012 with window status showing open. This indicates forced entry or damage. Immediate security response required and authorities should be contacted."
            },
        ]
        
        self.agriculture_scenarios = [
            {
                "instruction": "Analyze soil moisture reading",
                "context": "Soil moisture: 35%, Status: dry, Device: AGR_201",
                "response": "Soil moisture at 35% indicates dry conditions detected by AGR_201. This is below optimal range for most crops (40-60%). Recommend increasing irrigation frequency to prevent crop stress and yield reduction."
            },
            {
                "instruction": "Evaluate temperature sensor data",
                "context": "Temperature: 32°C, Status: alert, Device: AGR_150",
                "response": "Temperature alert at 32°C from AGR_150 exceeds optimal growing range (18-28°C). High temperature may cause heat stress in plants. Recommend activating cooling systems or shade structures to protect crops."
            },
            {
                "instruction": "Interpret humidity levels",
                "context": "Humidity: 75%, Status: alert, Device: AGR_089",
                "response": "Elevated humidity at 75% detected by AGR_089 exceeds optimal range (40-70%). High humidity increases risk of fungal diseases and mold growth. Recommend improving ventilation and monitoring for disease symptoms."
            },
            {
                "instruction": "Assess light exposure data",
                "context": "Light: 45000 lux, Status: day, Device: AGR_312",
                "response": "Light exposure at 45,000 lux indicates strong daylight conditions suitable for photosynthesis. AGR_312 shows optimal lighting for plant growth. Current conditions support healthy crop development."
            },
        ]
        
        self.anomaly_scenarios = [
            {
                "instruction": "Detect anomalies in sensor pattern",
                "context": "Multiple motion sensors triggered simultaneously across different zones within 2 seconds",
                "response": "Anomaly detected: Simultaneous motion alerts across multiple zones suggest either a system malfunction or coordinated unauthorized activity. Recommend immediate system diagnostics and security verification."
            },
            {
                "instruction": "Identify unusual sensor behavior",
                "context": "Temperature sensor showing 45°C reading, which is 15°C above historical average",
                "response": "Critical anomaly: Temperature reading significantly exceeds historical patterns. This could indicate sensor malfunction, equipment failure, or actual environmental emergency. Verify sensor calibration and investigate heat source immediately."
            },
        ]
    
    def generate_training_dataset(self, num_samples: int = 100) -> List[Dict]:
        """Generate training dataset with variations"""
        
        dataset = []
        all_scenarios = (
            self.surveillance_scenarios * 10 + 
            self.agriculture_scenarios * 10 + 
            self.anomaly_scenarios * 5
        )
        
        # Add base scenarios
        dataset.extend(all_scenarios[:num_samples])
        
        # Create variations with different values
        for _ in range(num_samples - len(dataset)):
            scenario_type = random.choice(['surveillance', 'agriculture', 'anomaly'])
            
            if scenario_type == 'surveillance':
                base = random.choice(self.surveillance_scenarios)
            elif scenario_type == 'agriculture':
                base = random.choice(self.agriculture_scenarios)
            else:
                base = random.choice(self.anomaly_scenarios)
            
            dataset.append(base.copy())
        
        return dataset[:num_samples]
    
    def format_for_training(self, dataset: List[Dict], format_type: str = "alpaca") -> List[Dict]:
        """Format dataset for different training frameworks"""
        
        if format_type == "alpaca":
            # Alpaca format: instruction, input, output
            formatted = []
            for item in dataset:
                formatted.append({
                    "instruction": item["instruction"],
                    "input": item["context"],
                    "output": item["response"]
                })
            return formatted
        
        elif format_type == "chat":
            # Chat format for Qwen
            formatted = []
            for item in dataset:
                formatted.append({
                    "messages": [
                        {"role": "system", "content": "You are an expert IoT monitoring assistant specialized in surveillance and agriculture systems."},
                        {"role": "user", "content": f"{item['instruction']}\n\nContext: {item['context']}"},
                        {"role": "assistant", "content": item['response']}
                    ]
                })
            return formatted
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = "iot_training_data.json"):
        """Save dataset to file"""
        with open(f"training/{filename}", 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Saved {len(dataset)} training examples to training/{filename}")


if __name__ == "__main__":
    generator = IoTTrainingDataGenerator()
    
    # Generate training data
    print("Generating IoT domain training data...")
    raw_dataset = generator.generate_training_dataset(num_samples=150)
    
    # Format for training
    alpaca_format = generator.format_for_training(raw_dataset, format_type="alpaca")
    chat_format = generator.format_for_training(raw_dataset, format_type="chat")
    
    # Save datasets
    generator.save_dataset(alpaca_format, "iot_alpaca_format.json")
    generator.save_dataset(chat_format, "iot_chat_format.json")
    
    # Show samples
    print("\n" + "="*60)
    print("Sample Training Example (Alpaca Format):")
    print("="*60)
    print(json.dumps(alpaca_format[0], indent=2))
    
    print("\n" + "="*60)
    print("Sample Training Example (Chat Format):")
    print("="*60)
    print(json.dumps(chat_format[0], indent=2))
    
    print(f"\n✓ Generated {len(alpaca_format)} training examples")
    print("✓ Ready for LoRA fine-tuning!")