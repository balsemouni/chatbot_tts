
import torch
import numpy as np
import time

from agent.dsp.agc import SimpleAGC
from agent.dsp.deepfilter import DeepFilterNoiseReducer
from agent.dsp.buffer import SpeechBuffer

class VoiceActivityDetector:
    """
    High-sensitivity VAD for catching ALL speech
    """
    
    def __init__(
        self,
        sample_rate=16000,
        device="cpu",
        idle_threshold=0.25,
        barge_in_threshold=0.50,
        min_rms=0.005,
        silence_limit_ms=600,
        enable_noise_reduction=True,
        min_chunk_samples=512,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.silence_limit_ms = silence_limit_ms
        self.enable_noise_reduction = enable_noise_reduction
        self.min_chunk_samples = min_chunk_samples
        
        # --- Silero VAD ---
        print("🎯 Loading high-sensitivity VAD...")
        try:
            self.vad_model, _ = torch.hub.load(
                "snakers4/silero-vad",
                "silero_vad",
                force_reload=False,
                verbose=False
            )
            self.vad_model.to(self.device).eval()
            print("✅ VAD model loaded")
        except Exception as e:
            print(f"❌ Failed to load VAD: {e}")
            raise
        
        # --- DSP chain ---
        self.agc = SimpleAGC(target_rms=0.015)
        self.buffer = SpeechBuffer(sample_rate, min_speech_ms=0)  # NO minimum!
        
        # Noise reduction
        if self.enable_noise_reduction:
            try:
                self.denoiser = DeepFilterNoiseReducer(
                    sample_rate=sample_rate,
                    device=device
                )
            except Exception as e:
                print(f"⚠️ Could not load DeepFilter: {e}")
                self.denoiser = None
        else:
            self.denoiser = None
        
        # --- Thresholds (LOW for sensitivity) ---
        self.idle_threshold = idle_threshold
        self.barge_in_threshold = barge_in_threshold
        self.min_rms = min_rms
        
        # --- State ---
        self.vad_accumulator = np.array([], dtype=np.float32)
        self.last_vad_prob = 0.0
        self.consecutive_voice = 0
        self.consecutive_silence = 0
        
        print("✅ VAD ready (high sensitivity mode)")
    
    @staticmethod
    def rms(audio: np.ndarray) -> float:
        """RMS with noise floor compensation"""
        if len(audio) == 0:
            return 0.0
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        return max(rms, 1e-5)
    
    def process_chunk(self, audio_chunk: np.ndarray, ai_is_speaking=False):
        """
        Process with HIGH sensitivity
        """
        if len(audio_chunk) == 0:
            return None, False, 0.0, 0.0
        
        start_time = time.time()
        
        try:
            # 1) AGC
            audio = self.agc.process(audio_chunk)
            
            # 2) Optional noise reduction
            if self.enable_noise_reduction and self.denoiser is not None:
                audio = self.denoiser.process(audio)
            
            # 3) VAD
            self.vad_accumulator = np.concatenate([self.vad_accumulator, audio])
            
            prob = self.last_vad_prob
            
            if len(self.vad_accumulator) >= self.min_chunk_samples:
                vad_chunk = self.vad_accumulator[:self.min_chunk_samples]
                
                tensor = torch.from_numpy(vad_chunk).to(
                    device=self.device,
                    dtype=torch.float32
                )
                
                with torch.no_grad():
                    prob = self.vad_model(tensor, self.sample_rate).item()
                
                self.vad_accumulator = self.vad_accumulator[self.min_chunk_samples:]
                self.last_vad_prob = prob
            
            # 4) RMS
            rms_val = self.rms(audio)
            
            # 5) VOICE DETECTION with HYSTERESIS
            threshold = self.barge_in_threshold if ai_is_speaking else self.idle_threshold
            
            # Very sensitive detection
            if prob > threshold + 0.05:
                is_voice = True
                self.consecutive_voice += 1
                self.consecutive_silence = max(0, self.consecutive_silence - 1)
            elif prob < threshold - 0.05:
                is_voice = False
                self.consecutive_silence += 1
                self.consecutive_voice = max(0, self.consecutive_voice - 1)
            else:
                # Borderline - favor voice if we've had recent voice
                is_voice = (self.consecutive_voice > self.consecutive_silence)
            
            # EXTREMELY LOW volume threshold
            volume_threshold = self.min_rms * 0.5 if ai_is_speaking else self.min_rms
            if rms_val < volume_threshold:
                is_voice = False
            
            # 6) Buffering (NO minimum length!)
            segment = self.buffer.push(audio, is_voice)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Only warn about really high latency
            if processing_time > 50:
                print(f"⚠️ High VAD latency: {processing_time:.1f}ms")
            
            return segment, is_voice, prob, rms_val
            
        except Exception as e:
            print(f"❌ VAD processing error: {e}")
            # Return safe defaults
            return None, False, 0.0, 0.0
    
    def get_state(self):
        """Get current detection state"""
        return {
            'prob': self.last_vad_prob,
            'voice_count': self.consecutive_voice,
            'silence_count': self.consecutive_silence
        }
    
    def reset(self):
        """Reset state"""
        self.vad_accumulator = np.array([], dtype=np.float32)
        self.last_vad_prob = 0.0
        self.consecutive_voice = 0
        self.consecutive_silence = 0
