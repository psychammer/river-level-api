# test_client.py
"""
Example client for testing the Water Level Prediction API
"""

import requests
import json
from datetime import datetime
from typing import Dict, List
import time

class WaterLevelClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """Check if API is running"""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except:
            return False
    
    def get_predictions(self, lookback_hours: int = 72, model_type: str = "full") -> Dict:
        """Get water level predictions for all stations"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"lookback_hours": lookback_hours, "model_type": model_type}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting predictions: {e}")
            return None
    
    def compare_models(self, lookback_hours: int = 72) -> Dict:
        """Compare predictions from both models"""
        try:
            response = self.session.post(
                f"{self.base_url}/compare_models",
                json={"lookback_hours": lookback_hours}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error comparing models: {e}")
            return None
    
    def get_stations(self) -> List[Dict]:
        """Get list of available stations"""
        try:
            response = self.session.get(f"{self.base_url}/stations")
            response.raise_for_status()
            return response.json()["stations"]
        except requests.exceptions.RequestException as e:
            print(f"Error getting stations: {e}")
            return []
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model_info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting model info: {e}")
            return {}
    
    def display_predictions(self, predictions: Dict):
        """Display predictions in a formatted way"""
        if not predictions:
            print("No predictions available")
            return
        
        print("\n" + "="*80)
        print(f"WATER LEVEL PREDICTIONS - {predictions['timestamp']}")
        print(f"Model Type: {predictions.get('model_type', 'Unknown')}")
        print("="*80)
        print(f"\nData Range: {predictions['data_range']['start']} to {predictions['data_range']['end']}")
        print("\n")
        
        for station_data in predictions['predictions']:
            print(f"📍 Station: {station_data['station']}")
            print(f"   Current Level: {station_data['current_level']:.2f} m" if station_data['current_level'] else "   Current Level: N/A")
            print("   Predictions:")
            
            for horizon, value in station_data['predictions'].items():
                # Check alert levels
                alert_status = ""
                if value >= station_data['alert_levels']['critical']:
                    alert_status = " ⚠️ CRITICAL"
                elif value >= station_data['alert_levels']['alarm']:
                    alert_status = " ⚠️ ALARM"
                elif value >= station_data['alert_levels']['alert']:
                    alert_status = " ⚠️ ALERT"
                
                print(f"      +{horizon}: {value:.2f} m{alert_status}")
            
            print(f"   Alert Thresholds:")
            print(f"      Alert: {station_data['alert_levels']['alert']:.2f} m")
            print(f"      Alarm: {station_data['alert_levels']['alarm']:.2f} m")
            print(f"      Critical: {station_data['alert_levels']['critical']:.2f} m")
            print()
    
    def display_comparison(self, comparison_data: Dict):
        """Display model comparison results"""
        if not comparison_data:
            print("No comparison data available")
            return
        
        print("\n" + "="*80)
        print(f"MODEL COMPARISON - {comparison_data['timestamp']}")
        print("="*80)
        print(f"\nData Range: {comparison_data['data_range']['start']} to {comparison_data['data_range']['end']}")
        print("\n")
        
        for station, horizons in comparison_data['comparison'].items():
            print(f"📍 Station: {station}")
            print("   Horizon | Full Model | Ablated Model | Difference | % Diff")
            print("   " + "-"*60)
            
            for horizon, values in horizons.items():
                print(f"   {horizon:7s} | {values['full_model']:10.2f} | {values['ablated_model']:13.2f} | "
                      f"{values['difference']:10.2f} | {values['percent_diff']:6.1f}%")
            print()
    
    def monitor_predictions(self, interval_minutes: int = 60, model_type: str = "full"):
        """Continuously monitor predictions"""
        print(f"Starting continuous monitoring (updating every {interval_minutes} minutes)")
        print(f"Using model: {model_type}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                predictions = self.get_predictions(model_type=model_type)
                if predictions:
                    self.display_predictions(predictions)
                    
                    # Save to file for logging
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    with open(f"predictions_{model_type}_{timestamp}.json", "w") as f:
                        json.dump(predictions, f, indent=2)
                    print(f"\n💾 Predictions saved to predictions_{model_type}_{timestamp}.json")
                else:
                    print(f"⚠️ Failed to get predictions at {datetime.now()}")
                
                print(f"\nNext update in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

def main():
    # Initialize client
    client = WaterLevelClient("http://localhost:8000")
    
    # Check API health
    print("Checking API status...")
    if not client.check_health():
        print("❌ API is not running. Please start the server first.")
        return
    
    print("✅ API is online!\n")
    
    # Get model information
    print("Model Information:")
    model_info = client.get_model_info()
    if model_info:
        print(f"  Available Models: {model_info.get('available_models')}")
        print(f"  Forecast Horizons: {model_info.get('forecast_horizons')}")
        print(f"  Lookback Window: {model_info.get('lookback_window')} hours")
        print(f"  Number of Stations: {model_info.get('num_stations')}")
        print(f"  Device: {model_info.get('device')}")
        
        print("\n  Model Details:")
        for model_key, model_details in model_info.get('models', {}).items():
            print(f"    {model_key}: {model_details['name']}")
            print(f"      - {model_details['description']}")
    
    # Get available stations
    print("\nAvailable Stations:")
    stations = client.get_stations()
    for station in stations:
        print(f"  - {station['name']}: ({station['coordinates']['lat']}, {station['coordinates']['lon']})")
    
    # Check available models
    available_models = model_info.get('available_models', [])
    if not available_models:
        print("\n⚠️ No models available. Please ensure model files are properly loaded.")
        return
    
    # Ask user to choose model
    print("\n" + "="*50)
    print("SELECT OPERATION MODE")
    print("="*50)
    print("1. Predict with Full Model (STGAE-GAT-Transformer)")
    print("2. Predict with Ablated Model (GAT-Transformer)")
    print("3. Compare Both Models")
    print("4. Continuous Monitoring")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1":
        if "full" not in available_models:
            print("❌ Full model not available")
            return
        print("\nFetching predictions with FULL model...")
        predictions = client.get_predictions(lookback_hours=72, model_type="full")
        if predictions:
            client.display_predictions(predictions)
            
    elif choice == "2":
        if "ablated" not in available_models:
            print("❌ Ablated model not available")
            return
        print("\nFetching predictions with ABLATED model...")
        predictions = client.get_predictions(lookback_hours=72, model_type="ablated")
        if predictions:
            client.display_predictions(predictions)
            
    elif choice == "3":
        if not all(m in available_models for m in ["full", "ablated"]):
            print("❌ Both models must be available for comparison")
            return
        print("\nComparing models...")
        comparison = client.compare_models(lookback_hours=72)
        if comparison:
            client.display_comparison(comparison)
            
    elif choice == "4":
        # Ask for monitoring preferences
        model_choice = input("Which model to use for monitoring? (full/ablated): ").lower()
        if model_choice not in available_models:
            print(f"❌ Model '{model_choice}' not available")
            return
        
        interval = input("Enter monitoring interval in minutes (default 60): ")
        interval = int(interval) if interval else 60
        client.monitor_predictions(interval, model_type=model_choice)
    
    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    main()