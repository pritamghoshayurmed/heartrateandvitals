# PPG Service - Quick Start Guide

## 🚀 Starting the PPG Service

### Method 1: Using the Batch File (Easiest - Windows)
1. Navigate to `d:\KabirajAI\ppg-service`
2. Double-click `start.bat`
3. The service will start on port 8001

### Method 2: Using Command Line
```cmd
cd d:\KabirajAI\ppg-service
python app.py
```

## ✅ Verify Service is Running

Open your browser and go to: http://localhost:8001

You should see:
```json
{
  "service": "PPG Vitals Analysis",
  "status": "running",
  "version": "1.0.0"
}
```

## 📊 What Can PPG Measure?

### ✅ ACCURATELY Measurable
- **Heart Rate (HR)** - Beats per minute
- **Heart Rate Variability (HRV)** - SDNN in milliseconds

### 📈 ESTIMATED
- **Respiratory Rate (RR)** - Derived from signal amplitude modulation

### ❌ CANNOT Measure
- Body Temperature
- Blood Pressure
- Blood Oxygen (SpO2) - Requires dual-wavelength LED
- Blood Glucose

## 🔧 Troubleshooting

### Error: "Failed to fetch" / "ERR_CONNECTION_REFUSED"
**Problem:** PPG service is not running

**Solution:**
1. Open a new terminal/cmd window
2. Navigate to: `cd d:\KabirajAI\ppg-service`
3. Run: `python app.py`
4. Keep this terminal open while using the app

### Error: "Module not found"
**Problem:** Missing Python packages

**Solution:**
```cmd
cd d:\KabirajAI\ppg-service
pip install -r requirements.txt
```

### Port Already in Use
**Problem:** Port 8001 is occupied

**Solution:**
```cmd
# Kill process on port 8001
netstat -ano | findstr :8001
taskkill /PID <process_id> /F
```

## 📱 Frontend Configuration

The frontend (VitalsMeasurement.tsx) connects to:
```typescript
const PPG_SERVICE_URL = 'http://localhost:8001';
```

Make sure this matches where your service is running!

## 🧪 Testing the Service

### Test Signal Quality Endpoint
```bash
curl -X POST http://localhost:8001/test/signal-quality ^
  -H "Content-Type: application/json" ^
  -d "{\"frames\": []}"
```

### Health Check
```bash
curl http://localhost:8001/health
```
