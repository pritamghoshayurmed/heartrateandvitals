"""
Test script for PPG Vitals Microservice
Tests the API endpoints with sample data
"""
import requests
import base64
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import io

SERVICE_URL = "http://localhost:8001"

def create_test_image(width=640, height=480, intensity=128):
    """Create a simple test image"""
    img_array = np.full((height, width, 3), intensity, dtype=np.uint8)
    # Add some variation
    img_array[height//2-50:height//2+50, width//2-50:width//2+50] = intensity + 20
    
    img = Image.fromarray(img_array, 'RGB')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

def test_health_check():
    """Test health check endpoint"""
    print("\n[TEST] Health Check")
    print("-" * 50)
    
    try:
        response = requests.get(f"{SERVICE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✓ Health check passed")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False

def test_signal_quality():
    """Test signal quality endpoint"""
    print("\n[TEST] Signal Quality Check")
    print("-" * 50)
    
    try:
        # Create test frames
        frames = [create_test_image(intensity=120 + i*2) for i in range(10)]
        
        payload = {
            "frames": frames
        }
        
        print(f"Sending {len(frames)} test frames...")
        response = requests.post(
            f"{SERVICE_URL}/test/signal-quality",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        assert response.status_code == 200
        assert "quality_score" in result
        print("✓ Signal quality check passed")
        return True
    except Exception as e:
        print(f"✗ Signal quality check failed: {str(e)}")
        return False

def test_analyze_frames():
    """Test frame analysis endpoint"""
    print("\n[TEST] Analyze Frames")
    print("-" * 50)
    
    try:
        # Create realistic test frames (simulate PPG signal)
        num_frames = 90  # 3 seconds at 30fps
        frames = []
        
        print(f"Generating {num_frames} test frames with simulated PPG signal...")
        
        for i in range(num_frames):
            # Simulate heartbeat at 70 BPM (70/60 = 1.17 Hz)
            # PPG signal has both DC and AC components
            t = i / 30.0  # Time in seconds
            ppg_signal = 128 + 10 * np.sin(2 * np.pi * 1.17 * t)  # Simulate heartbeat
            ppg_signal += 2 * np.random.randn()  # Add noise
            
            intensity = int(np.clip(ppg_signal, 0, 255))
            frames.append(create_test_image(intensity=intensity))
        
        payload = {
            "frames": frames,
            "frame_rate": 30,
            "metadata": {
                "test": True,
                "timestamp": time.time()
            }
        }
        
        print(f"Sending {len(frames)} frames for analysis...")
        start_time = time.time()
        
        response = requests.post(
            f"{SERVICE_URL}/analyze/frames",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        processing_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {processing_time:.2f}s")
        
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("success"):
            vitals = result.get("vitals", {})
            print(f"\n✓ Analysis successful!")
            print(f"  Heart Rate: {vitals.get('heart_rate', 'N/A')} BPM")
            print(f"  Respiratory Rate: {vitals.get('respiratory_rate', 'N/A')} BrPM")
            print(f"  HRV: {vitals.get('heart_rate_variability', 'N/A')} ms")
            print(f"  Beats Detected: {vitals.get('beats_detected', 'N/A')}")
            return True
        else:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (processing taking too long)")
        return False
    except Exception as e:
        print(f"✗ Frame analysis failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\n[TEST] Error Handling")
    print("-" * 50)
    
    try:
        # Test with too few frames
        print("Testing with insufficient frames...")
        payload = {
            "frames": [create_test_image() for _ in range(5)],
            "frame_rate": 30
        }
        
        response = requests.post(
            f"{SERVICE_URL}/analyze/frames",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        assert response.status_code == 400
        print("✓ Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("PPG Vitals Microservice Test Suite")
    print("=" * 50)
    
    results = {
        "Health Check": test_health_check(),
        "Signal Quality": test_signal_quality(),
        "Analyze Frames": test_analyze_frames(),
        "Error Handling": test_error_handling()
    }
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print("-" * 50)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print("=" * 50)
    
    return all(results.values())

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
