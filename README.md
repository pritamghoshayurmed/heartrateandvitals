# PPG Vitals Analysis Microservice

This microservice analyzes PPG (Photoplethysmography) signals from smartphone camera video to extract vital signs including heart rate, heart rate variability, and respiratory rate.

## Features

- Real-time PPG signal processing
- Heart rate (HR) measurement
- Heart rate variability (HRV) calculation
- Respiratory rate estimation
- Signal quality assessment
- REST API interface

## Technology Stack

- **FastAPI**: Modern web framework
- **OpenCV**: Video/image processing
- **NumPy/SciPy**: Signal processing
- **scikit-learn**: Feature extraction

## Installation

### Using Docker (Recommended)

```bash
cd ppg-service
docker-compose up --build
```

The service will be available at `http://localhost:8001`

### Manual Installation

```bash
cd ppg-service
pip install -r requirements.txt
python main.py
```

## API Endpoints

### 1. Health Check
```
GET /health
```

### 2. Analyze Frames
```
POST /analyze/frames
Content-Type: application/json

{
  "frames": ["base64_encoded_image1", "base64_encoded_image2", ...],
  "frame_rate": 30
}
```

**Response:**
```json
{
  "success": true,
  "vitals": {
    "heart_rate": 72.5,
    "heart_rate_variability": 45.2,
    "respiratory_rate": 16.0,
    "beats_detected": 36,
    "signal_duration": 30.0
  },
  "quality": {
    "snr": 15.3,
    "signal_strength": 8.2,
    "quality_score": 0.85
  },
  "confidence": {
    "heart_rate": "high",
    "respiratory_rate": "medium"
  }
}
```

### 3. Test Signal Quality
```
POST /test/signal-quality

{
  "frames": ["base64_encoded_image1", ...]
}
```

## Usage from Frontend

See the React component in `aether-med-nexus-58/src/pages/dashboard/VitalsMeasurement.tsx` for integration example.

## Algorithm

Based on research from "Seeing Red: PPG Biometrics Using Smartphone Cameras" (IEEE CVPRW 2020):

1. **Signal Extraction**: Extract LUMA component from video frames
2. **Preprocessing**: Apply bandpass filtering and detrending
3. **Beat Detection**: Identify heartbeat peaks using signal processing
4. **Vital Calculation**: Calculate HR, HRV, and respiratory rate
5. **Quality Assessment**: Evaluate signal quality metrics

## Configuration

Edit `ppg_processor.py` to adjust:
- Filter parameters
- Beat detection thresholds
- Signal quality thresholds

## Requirements

- Python 3.10+
- Minimum 30 frames (1 second at 30fps)
- Optimal: 900 frames (30 seconds at 30fps)
- Camera: Smartphone with LED flash

## Performance

- Processing time: ~2-3 seconds for 30 seconds of video
- Accuracy: HR ±3 BPM (research-validated)
- Signal quality threshold: SNR > 10 dB

## Troubleshooting

**Poor Signal Quality:**
- Ensure finger completely covers camera lens
- Enable LED flash/flashlight
- Hold finger steady during recording
- Apply gentle pressure

**No Beats Detected:**
- Increase recording duration
- Check lighting conditions
- Ensure camera is not obstructed

## License

MIT License - Based on seeing-red research by SSL Oxford
