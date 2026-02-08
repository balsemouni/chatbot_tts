from faster_whisper import WhisperModel
import numpy as np

class StreamingSpeechRecognizer:
    """Transcribes audio to text with real-time word streaming"""
    
    def __init__(self, model_size="base.en", device="cuda"):
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Loading Whisper model on {device.upper()}...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    
    def transcribe_streaming(self, audio_data):
        """
        Generator that yields words as they are detected.
        Each word is immediately available for LLM processing.
        """
        segments, _ = self.model.transcribe(
            audio_data, 
            beam_size=1,
            word_timestamps=True  # Enable word-level timestamps
        )
        
        for segment in segments:
            # Yield each word individually as soon as it's detected
            if hasattr(segment, 'words') and segment.words:
                for word_info in segment.words:
                    yield word_info.word.strip()
            else:
                # Fallback if word timestamps not available
                words = segment.text.strip().split()
                for word in words:
                    yield word
    
    def transcribe(self, audio_data):
        """Legacy method for compatibility - returns full text"""
        segments, _ = self.model.transcribe(audio_data, beam_size=1)
        return " ".join(s.text for s in segments).strip()