"""
PPG Analysis Microservice
FastAPI service for processing PPG signals from video/images
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import base64
import logging
import io
from PIL import Image

from ppg_processor import PPGSignalProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PPG Vitals Analysis Service",
    description="Microservice for analyzing PPG signals from smartphone camera",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PPG processor
ppg_processor = PPGSignalProcessor(frame_rate=30)


class FrameData(BaseModel):
    """Model for receiving base64 encoded frames"""
    frames: List[str]
    frame_rate: Optional[int] = 30
    metadata: Optional[dict] = {}


class VitalsResponse(BaseModel):
    """Response model for vitals analysis"""
    success: bool
    vitals: dict
    quality: Optional[dict] = None
    confidence: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "PPG Vitals Analysis",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "ppg-analysis",
        "processor_ready": True
    }


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (OpenCV format)"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array (RGB)
        img_array = np.array(img)
        
        # Convert RGB to BGR (OpenCV format)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_array
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        raise ValueError(f"Failed to decode image: {str(e)}")


@app.post("/analyze/frames", response_model=VitalsResponse)
async def analyze_frames(data: FrameData):
    """
    Analyze PPG signal from a sequence of video frames
    
    Expects base64 encoded images captured at regular intervals
    """
    try:
        logger.info(f"Received {len(data.frames)} frames for analysis")
        
        if len(data.frames) < 30:
            raise HTTPException(
                status_code=400,
                detail="Insufficient frames. Need at least 30 frames (1 second at 30fps)"
            )
        
        if len(data.frames) > 1200:
            raise HTTPException(
                status_code=400,
                detail="Too many frames. Maximum 1200 frames (40 seconds at 30fps)"
            )
        
        # Decode frames
        frames = []
        for i, frame_b64 in enumerate(data.frames):
            try:
                frame = decode_base64_image(frame_b64)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to decode frame {i}: {str(e)}")
                continue
        
        if len(frames) < 30:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode enough valid frames"
            )
        
        logger.info(f"Successfully decoded {len(frames)} frames")
        
        # Update processor frame rate
        if data.frame_rate:
            ppg_processor.frame_rate = data.frame_rate
            ppg_processor.sample_rate = data.frame_rate
        
        # Process frames
        result = ppg_processor.process_video_frames(frames)
        
        return VitalsResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_frames: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/analyze/video")
async def analyze_video(video: UploadFile = File(...), frame_rate: int = Form(30)):
    """
    Analyze PPG signal from uploaded video file
    
    Alternative endpoint for video file upload
    """
    try:
        logger.info(f"Received video file: {video.filename}")
        
        # Read video file
        video_bytes = await video.read()
        
        # Save temporarily
        temp_path = f"/tmp/{video.filename}"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(temp_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        
        if len(frames) < 30:
            raise HTTPException(
                status_code=400,
                detail="Video too short. Need at least 1 second of footage"
            )
        
        # Update processor frame rate
        ppg_processor.frame_rate = frame_rate
        ppg_processor.sample_rate = frame_rate
        
        # Process frames
        result = ppg_processor.process_video_frames(frames)
        
        return VitalsResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/test/signal-quality")
async def test_signal_quality(data: FrameData):
    """
    Test endpoint to check signal quality before full analysis
    Returns quality metrics without full processing
    """
    try:
        if len(data.frames) < 10:
            raise HTTPException(
                status_code=400,
                detail="Need at least 10 frames for quality check"
            )
        
        # Decode first 10 frames
        frames = []
        for frame_b64 in data.frames[:10]:
            try:
                frame = decode_base64_image(frame_b64)
                frames.append(frame)
            except:
                continue
        
        if len(frames) < 5:
            return {
                "quality_score": 0.0,
                "message": "Failed to decode frames",
                "recommendation": "Check camera access and image format"
            }
        
        # Extract signal from sample
        signal = ppg_processor.extract_luma_signal(frames, skip_seconds=0)
        quality = ppg_processor.analyze_signal_quality(signal)
        
        message = "Good signal quality"
        recommendation = "Continue recording"
        
        if quality["quality_score"] < 0.3:
            message = "Poor signal quality"
            recommendation = "Ensure finger completely covers camera lens and flashlight is on"
        elif quality["quality_score"] < 0.6:
            message = "Fair signal quality"
            recommendation = "Try to hold finger more steadily"
        
        return {
            "quality_score": quality["quality_score"],
            "snr": quality["snr"],
            "signal_strength": quality["signal_strength"],
            "message": message,
            "recommendation": recommendation
        }
        
    except Exception as e:
        logger.error(f"Error in test_signal_quality: {str(e)}")
        return {
            "quality_score": 0.0,
            "message": "Error checking quality",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
