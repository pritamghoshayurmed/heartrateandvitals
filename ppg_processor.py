"""
PPG Signal Processor - Adapted from seeing-red research
Processes video frames to extract vital signs using photoplethysmography
"""
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, argrelmin, argrelmax
from scipy import interpolate
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPGSignalProcessor:
    """Process PPG signals from video frames to extract vital signs"""
    
    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.sample_rate = frame_rate
        
    def extract_luma_signal(self, frames: List[np.ndarray], skip_seconds: int = 1) -> np.ndarray:
        """Extract LUMA component mean from video frames"""
        signal = []
        for frame_bgr in frames:
            img_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            mean_of_luma = img_ycrcb[..., 0].mean()
            signal.append(mean_of_luma)
        
        signal = np.array(signal)
        samples_to_skip = skip_seconds * self.sample_rate
        signal = signal[samples_to_skip:]  # Skip first second due to auto-exposure
        return signal
    
    def butter_bandpass_filter(self, signal: np.ndarray, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth bandpass filter"""
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y
    
    def butter_lowpass_filter(self, signal: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
        """Apply Butterworth lowpass filter"""
        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, signal)
        return y
    
    def rolling_average(self, signal: np.ndarray, window_seconds: float = 2.0) -> np.ndarray:
        """Apply rolling average to smooth signal"""
        window_size = int(window_seconds * self.sample_rate)
        if window_size % 2 == 0:
            window_size += 1
        y = np.convolve(signal, np.ones(window_size), 'valid') / window_size
        y = np.pad(y, [((window_size - 1) // 2, (window_size - 1) // 2)], mode='edge')
        return y
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess PPG signal with filtering pipeline"""
        # Apply rolling average to remove slow trends
        processed = self.rolling_average(signal, window_seconds=2.0)
        
        # Subtract from original to get AC component
        processed = signal - processed
        
        # Apply lowpass filter to remove high-frequency noise
        processed = self.butter_lowpass_filter(processed, cutoff=3.0, order=4)
        
        return processed
    
    def detect_beats(self, signal: np.ndarray, max_bpm: int = 180) -> List[int]:
        """Detect heartbeat peaks in the signal"""
        # Smooth signal for peak detection
        smooth_window = 13
        if smooth_window % 2 == 0:
            smooth_window += 1
        smoothed = np.convolve(signal, np.ones(smooth_window), 'valid') / smooth_window
        smoothed = np.pad(smoothed, [((smooth_window - 1) // 2, (smooth_window - 1) // 2)], mode='edge')
        
        # Find peaks
        peaks = argrelmax(smoothed, order=2)[0]
        
        # Filter peaks based on minimum distance (max BPM constraint)
        min_frame_gap = self.frame_rate / (max_bpm / 60)
        
        filtered_peaks = []
        if len(peaks) > 0:
            filtered_peaks.append(peaks[0])
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= min_frame_gap:
                    filtered_peaks.append(peak)
        
        return filtered_peaks
    
    def calculate_heart_rate(self, beat_indices: List[int]) -> Tuple[float, float]:
        """Calculate heart rate and HRV from beat indices"""
        if len(beat_indices) < 2:
            return 0.0, 0.0
        
        # Calculate inter-beat intervals (IBI) in seconds
        ibis = np.diff(beat_indices) / self.frame_rate
        
        # Heart rate in BPM
        heart_rate = 60.0 / np.mean(ibis)
        
        # Heart rate variability (standard deviation of IBI)
        hrv = np.std(ibis) * 1000  # Convert to milliseconds
        
        return heart_rate, hrv
    
    def estimate_respiratory_rate(self, signal: np.ndarray, beat_indices: List[int]) -> float:
        """Estimate respiratory rate from PPG signal amplitude modulation"""
        if len(beat_indices) < 3:
            return 0.0
        
        # Extract peak amplitudes
        peak_amplitudes = signal[beat_indices]
        
        # Apply lowpass filter to extract respiratory component
        if len(peak_amplitudes) > 10:
            filtered_amp = self.butter_lowpass_filter(peak_amplitudes, cutoff=0.5, order=2)
            
            # Find respiratory peaks
            resp_peaks = argrelmax(filtered_amp, order=2)[0]
            
            if len(resp_peaks) > 1:
                # Calculate respiratory rate
                resp_intervals = np.diff(resp_peaks) * np.mean(np.diff(beat_indices)) / self.frame_rate
                resp_rate = 60.0 / np.mean(resp_intervals)
                
                # Clamp to reasonable range (8-30 breaths per minute)
                resp_rate = np.clip(resp_rate, 8, 30)
                return float(resp_rate)
        
        return 15.0  # Default respiratory rate
    
    def extract_beat_waveforms(self, signal: np.ndarray, beat_indices: List[int]) -> List[np.ndarray]:
        """Extract individual heartbeat waveforms"""
        beats = []
        
        for i in range(len(beat_indices) - 1):
            start_idx = beat_indices[i]
            end_idx = beat_indices[i + 1]
            beat = signal[start_idx:end_idx]
            
            # Normalize beat
            if len(beat) > 0:
                beat = beat - beat.min()
                if beat.max() > 0:
                    beat = beat / beat.max()
                beats.append(beat)
        
        return beats
    
    def analyze_signal_quality(self, signal: np.ndarray) -> Dict[str, float]:
        """Analyze signal quality metrics"""
        # Calculate signal-to-noise ratio
        signal_power = np.var(signal)
        noise_estimate = np.var(np.diff(signal))
        snr = 10 * np.log10(signal_power / max(noise_estimate, 1e-10))
        
        # Calculate signal strength
        signal_strength = np.std(signal)
        
        return {
            "snr": float(snr),
            "signal_strength": float(signal_strength),
            "quality_score": float(min(max(snr / 20.0, 0.0), 1.0))  # Normalize to 0-1
        }
    
    def process_video_frames(self, frames: List[np.ndarray]) -> Dict:
        """Main processing pipeline for video frames"""
        try:
            logger.info(f"Processing {len(frames)} frames at {self.frame_rate} FPS")
            
            # Extract raw PPG signal
            raw_signal = self.extract_luma_signal(frames, skip_seconds=1)
            
            if len(raw_signal) < 30:  # Need at least 1 second of data
                return {
                    "success": False,
                    "error": "Insufficient signal length",
                    "vitals": {}
                }
            
            # Preprocess signal
            processed_signal = self.preprocess_signal(raw_signal)
            
            # Analyze signal quality
            quality_metrics = self.analyze_signal_quality(processed_signal)
            
            if quality_metrics["quality_score"] < 0.3:
                return {
                    "success": False,
                    "error": "Poor signal quality. Please ensure finger covers camera completely.",
                    "vitals": {},
                    "quality": quality_metrics
                }
            
            # Detect heartbeats
            beat_indices = self.detect_beats(processed_signal, max_bpm=180)
            
            if len(beat_indices) < 3:
                return {
                    "success": False,
                    "error": "Could not detect enough heartbeats. Please hold finger steady.",
                    "vitals": {},
                    "quality": quality_metrics
                }
            
            # Calculate heart rate and HRV
            heart_rate, hrv = self.calculate_heart_rate(beat_indices)
            
            # Estimate respiratory rate
            respiratory_rate = self.estimate_respiratory_rate(processed_signal, beat_indices)
            
            # Extract beat waveforms for further analysis
            beat_waveforms = self.extract_beat_waveforms(processed_signal, beat_indices)
            
            logger.info(f"Extracted vitals - HR: {heart_rate:.1f} BPM, RR: {respiratory_rate:.1f} BrPM")
            
            return {
                "success": True,
                "vitals": {
                    "heart_rate": round(heart_rate, 1),
                    "heart_rate_variability": round(hrv, 2),
                    "respiratory_rate": round(respiratory_rate, 1),
                    "beats_detected": len(beat_indices),
                    "signal_duration": len(raw_signal) / self.frame_rate
                },
                "quality": quality_metrics,
                "confidence": {
                    "heart_rate": "high" if quality_metrics["quality_score"] > 0.7 else "medium",
                    "respiratory_rate": "medium"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing video frames: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Processing error: {str(e)}",
                "vitals": {}
            }
