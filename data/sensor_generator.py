import random
import json
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

class IoTSensorDataGenerator:
    """Generates realistic IoT sensor data for surveillance and agriculture"""
    
    def __init__(self):
        self.sensor_types = {
            "surveillance": ["motion", "camera", "door", "window"],
            "agriculture": ["soil_moisture", "temperature", "humidity", "light"]
        }
        
    def generate_surveillance_event(self, timestamp: datetime) -> Dict:
        """Generate surveillance sensor event"""
        sensor_type = random.choice(self.sensor_types["surveillance"])
        
        events = {
            "motion": {
                "detected": random.choice([True, False]),
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "zone": random.choice(["entrance", "parking", "warehouse", "office"])
            },
            "camera": {
                "status": random.choice(["active", "recording", "idle"]),
                "fps": random.choice([15, 24, 30]),
                "resolution": random.choice(["1080p", "720p", "4K"])
            },
            "door": {
                "status": random.choice(["open", "closed", "locked"]),
                "access_granted": random.choice([True, False]),
                "user_id": f"USER_{random.randint(1000, 9999)}"
            },
            "window": {
                "status": random.choice(["open", "closed"]),
                "break_detected": random.choice([True] + [False]*9)  # 10% chance
            }
        }
        
        return {
            "timestamp": timestamp.isoformat(),
            "sensor_type": sensor_type,
            "category": "surveillance",
            "data": events[sensor_type],
            "device_id": f"CAM_{random.randint(100, 999)}"
        }
    
    def generate_agriculture_reading(self, timestamp: datetime) -> Dict:
        """Generate agriculture sensor reading"""
        sensor_type = random.choice(self.sensor_types["agriculture"])
        
        readings = {
            "soil_moisture": {
                "value": round(random.uniform(20, 80), 2),
                "unit": "%",
                "status": random.choice(["optimal", "dry", "wet"])
            },
            "temperature": {
                "value": round(random.uniform(15, 35), 2),
                "unit": "°C",
                "status": "normal" if 18 <= random.uniform(15, 35) <= 28 else "alert"
            },
            "humidity": {
                "value": round(random.uniform(30, 90), 2),
                "unit": "%",
                "status": "normal" if 40 <= random.uniform(30, 90) <= 70 else "alert"
            },
            "light": {
                "value": round(random.uniform(0, 100000), 2),
                "unit": "lux",
                "status": "day" if random.uniform(0, 100000) > 1000 else "night"
            }
        }
        
        return {
            "timestamp": timestamp.isoformat(),
            "sensor_type": sensor_type,
            "category": "agriculture",
            "data": readings[sensor_type],
            "device_id": f"AGR_{random.randint(100, 999)}"
        }
    
    def generate_dataset(
        self,
        num_records: int = 100,
        category: str = "both"
    ) -> List[Dict]:
        """Generate a dataset of sensor readings"""
        
        data = []
        start_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_records):
            timestamp = start_time + timedelta(minutes=i*10)
            
            if category == "surveillance":
                data.append(self.generate_surveillance_event(timestamp))
            elif category == "agriculture":
                data.append(self.generate_agriculture_reading(timestamp))
            else:  # both
                if random.random() > 0.5:
                    data.append(self.generate_surveillance_event(timestamp))
                else:
                    data.append(self.generate_agriculture_reading(timestamp))
        
        return data
    
    def save_to_json(self, data: List[Dict], filename: str = "sensor_data.json"):
        """Save data to JSON file"""
        with open(f"data/{filename}", "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(data)} records to data/{filename}")
    
    def save_to_csv(self, data: List[Dict], filename: str = "sensor_data.csv"):
        """Save data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(f"data/{filename}", index=False)
        print(f"✓ Saved {len(data)} records to data/{filename}")


# Test the generator
if __name__ == "__main__":
    generator = IoTSensorDataGenerator()
    
    print("Generating IoT sensor data...")
    data = generator.generate_dataset(num_records=200, category="both")
    
    # Show sample
    print("\nSample records:")
    print(json.dumps(data[:3], indent=2))
    
    # Save
    generator.save_to_json(data)
    
    print(f"\n✓ Generated {len(data)} total records")
    surveillance = sum(1 for d in data if d['category'] == 'surveillance')
    agriculture = sum(1 for d in data if d['category'] == 'agriculture')
    print(f"  - Surveillance: {surveillance}")
    print(f"  - Agriculture: {agriculture}")