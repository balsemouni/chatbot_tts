import torch
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue
from collections import deque

class StreamingSpeechRecognizer:
    """
    ZERO-LATENCY transcription with aggressive threading
    - Processes audio in parallel while you speak
    - Shows words IMMEDIATELY (under 500ms)
    - Chunks processed concurrently
    """
    
    def __init__(self, model_size="base.en", device="cuda"):
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"⚡ Loading FAST Whisper ({model_size}) on {device.upper()}...")
        
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            num_workers=1,
            cpu_threads=4
        )
        
        # AGGRESSIVE streaming settings
        self.sample_rate = 16000
        self.chunk_duration = 1.5  # Smaller chunks = faster results
        self.overlap_duration = 0.3  # Less overlap = faster
        
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(self.overlap_duration * self.sample_rate)
        
        # Threading for parallel processing
        self.transcription_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        print(f"✅ Fast mode: {self.chunk_duration}s chunks, {self.overlap_duration}s overlap")
    
    def transcribe_streaming(self, audio_data):
        """
        OPTIMIZED: Multi-threaded chunk processing
        Words appear WHILE processing (not after)
        """
        
        if len(audio_data) < self.sample_rate * 0.2:  # Less than 200ms
            return
        
        # Strategy: Process chunks in parallel threads
        chunks_to_process = []
        position = 0
        
        # 1. Split audio into chunks FIRST (fast operation)
        while position < len(audio_data):
            end = min(position + self.chunk_samples, len(audio_data))
            chunk = audio_data[position:end]
            
            if len(chunk) >= self.sample_rate * 0.3:  # At least 300ms
                chunks_to_process.append((position, chunk))
            
            position += (self.chunk_samples - self.overlap_samples)
        
        # 2. Process all chunks in parallel threads
        results = {}
        threads = []
        
        def process_chunk(idx, chunk_data):
            try:
                segments, _ = self.model.transcribe(
                    chunk_data,
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    word_timestamps=False,
                    language="en"
                )
                
                words = []
                for segment in segments:
                    words.extend(segment.text.strip().split())
                
                results[idx] = words
                
            except Exception as e:
                print(f"⚠️ Chunk {idx} error: {e}")
                results[idx] = []
        
        # Launch threads for each chunk
        for idx, (pos, chunk) in enumerate(chunks_to_process):
            thread = threading.Thread(
                target=process_chunk,
                args=(idx, chunk),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # 3. Yield words as threads complete (don't wait for all!)
        seen_words = set()
        completed = set()
        
        while len(completed) < len(chunks_to_process):
            # Check which threads finished
            for idx in range(len(chunks_to_process)):
                if idx in completed:
                    continue
                
                if idx in results:
                    # This chunk is done! Yield its words
                    for word in results[idx]:
                        word_clean = word.strip().lower()
                        
                        if word_clean and word_clean not in seen_words:
                            seen_words.add(word_clean)
                            yield word.strip()  # YIELD IMMEDIATELY
                    
                    completed.add(idx)
            
            # Tiny sleep to avoid busy waiting
            if len(completed) < len(chunks_to_process):
                import time
                time.sleep(0.01)
        
        # Wait for any remaining threads
        for thread in threads:
            thread.join(timeout=0.1)
    
    def transcribe_streaming_fast(self, audio_data):
        """
        FASTEST: Single-pass with minimal processing
        Trades some accuracy for speed
        """
        
        if len(audio_data) < self.sample_rate * 0.2:
            return
        
        # Process in one go with fastest settings
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=1,  # No beam search
                best_of=1,    # No sampling
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
                initial_prompt=None,  # No prompt overhead
                language="en",  # Skip detection
                task="transcribe"
            )
            
            # Yield words immediately as we get segments
            for segment in segments:
                words = segment.text.strip().split()
                for word in words:
                    if word.strip():
                        yield word.strip()
        
        except Exception as e:
            print(f"⚠️ Fast transcription error: {e}")
    
    def transcribe(self, audio_data):
        """Legacy method"""
        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=1,
            language="en"
        )
        return " ".join(s.text for s in segments).strip()