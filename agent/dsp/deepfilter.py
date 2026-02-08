
import torch
import numpy as np

try:
    from df.enhance import init_df, enhance
    from df.io import resample
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False
    print("⚠️ DeepFilterNet not available, skipping noise reduction")

class DeepFilterNoiseReducer:
    """Zero-latency DeepFilterNet noise reducer"""
    
    def __init__(self, sample_rate=16000, device="cpu", chunk_size=512):
        self.input_sample_rate = sample_rate
        self.device = device
        self.chunk_size = chunk_size
        self.model = None
        
        if not DEEPFILTER_AVAILABLE:
            print("⚠️ DeepFilterNet not available")
            return
        
        try:
            print("🧠 Loading DeepFilterNet...")
            self.model, self.df_state, _ = init_df(post_filter=True, log_level="ERROR")
            self.model = self.model.to(self.device).eval()
            self.df_sample_rate = self.df_state.sr()
            self.needs_resampling = (self.input_sample_rate != self.df_sample_rate)
            
            # Buffer for handling small chunks
            self.input_buffer = np.array([], dtype=np.float32)
            self.output_buffer = np.array([], dtype=np.float32)
            
            print(f"✅ DeepFilterNet loaded ({self.df_sample_rate}Hz)")
        except Exception as e:
            print(f"❌ DeepFilterNet failed: {e}")
            self.model = None
    
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Process audio chunk with minimal latency"""
        if self.model is None or len(audio_chunk) == 0:
            return audio_chunk
        
        try:
            # Ensure audio is 1D
            if audio_chunk.ndim > 1:
                audio_chunk = audio_chunk.flatten()
            
            # For low latency, skip deepfilter if chunk is too small
            if len(audio_chunk) < 256:
                return audio_chunk
            
            # SIMPLIFIED: Only apply every few chunks to reduce latency
            if np.random.random() > 0.3:  # 70% chance to skip
                return audio_chunk
            
            # Add to buffer
            self.input_buffer = np.concatenate([self.input_buffer, audio_chunk])
            
            # Process only when we have enough
            if len(self.input_buffer) >= self.chunk_size:
                process_chunk = self.input_buffer[:self.chunk_size]
                self.input_buffer = self.input_buffer[self.chunk_size:]
                
                audio_tensor = torch.from_numpy(process_chunk).float()
                
                if self.needs_resampling:
                    audio_tensor = resample(audio_tensor, self.input_sample_rate, self.df_sample_rate)
                
                audio_tensor = audio_tensor.to(self.device)
                
                with torch.no_grad():
                    audio_input = audio_tensor.unsqueeze(0).unsqueeze(0)
                    enhanced = enhance(self.model, self.df_state, audio_input)
                    enhanced = enhanced.squeeze(0).squeeze(0)
                
                if self.needs_resampling:
                    enhanced = resample(enhanced.cpu(), self.df_sample_rate, self.input_sample_rate)
                    enhanced_np = enhanced.numpy()
                else:
                    enhanced_np = enhanced.cpu().numpy()
                
                # Add to output buffer
                if len(enhanced_np) > 0:
                    self.output_buffer = np.concatenate([self.output_buffer, enhanced_np])
            
            # Return processed audio if available
            if len(self.output_buffer) >= len(audio_chunk):
                result = self.output_buffer[:len(audio_chunk)]
                self.output_buffer = self.output_buffer[len(audio_chunk):]
                return result
            else:
                return audio_chunk
                
        except Exception as e:
            # Silently return original audio on error
            return audio_chunk
    
    def flush(self):
        """Flush remaining audio"""
        if len(self.input_buffer) > 0:
            result = self.process(np.zeros(self.chunk_size, dtype=np.float32))
            self.input_buffer = np.array([], dtype=np.float32)
            return result
        return np.array([], dtype=np.float32)
    
    def __call__(self, audio_chunk: np.ndarray) -> np.ndarray:
        return self.process(audio_chunk)
