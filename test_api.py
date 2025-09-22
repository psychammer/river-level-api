# test_api.py - Test script for the River Level Prediction API
import requests
import json
from datetime import datetime

# API base URL (update this based on deployment)
BASE_URL = "http://localhost:8000"

def test_health():
    """Test API health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_stations():
    """Test stations endpoint"""
    print("Testing stations endpoint...")
    response = requests.get(f"{BASE_URL}/stations")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available stations: {data['stations']}\n")

def test_prediction(model_type="full"):
    """Test prediction endpoint"""
    print(f"Testing prediction with {model_type} model...")
    
    payload = {
        "model_type": model_type,
        "include_current_levels": True
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Timestamp: {data['timestamp']}")
        print(f"Model Type: {data['model_type']}")
        
        # Display predictions for each station
        for station in data['stations']:
            print(f"\n{station['station']}:")
            if station['current_level']:
                print(f"  Current Level: {station['current_level']:.2f} m")
            print("  Predictions:")
            for horizon, value in station['predictions'].items():
                print(f"    +{horizon}h: {value:.2f} m")
    else:
        print(f"Error: {response.text}")

def test_single_station(station="MONTALBAN", model_type="full"):
    """Test single station prediction"""
    print(f"Testing prediction for {station} with {model_type} model...")
    
    response = requests.get(f"{BASE_URL}/predict/station/{station}?model_type={model_type}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Station: {data['station']}")
        print(f"Timestamp: {data['timestamp']}")
        print("Predictions:")
        for horizon, value in data['predictions'].items():
            print(f"  +{horizon}h: {value:.2f} m")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("="*50)
    print("River Level Prediction API Test Suite")
    print("="*50 + "\n")
    
    # Run all tests
    test_health()
    test_stations()
    test_prediction("full")
    print("\n" + "="*50 + "\n")
    test_prediction("ablated")
    print("\n" + "="*50 + "\n")
    test_single_station("NANGKA", "full")
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)