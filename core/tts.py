import asyncio
import threading
import queue
import time
from typing import Optional, List
import pyttsx3
import re

class InstantTTS:
    """
    FREE, OFFLINE, ZERO-LATENCY TTS
    Speaks IMMEDIATELY as tokens arrive from LLM
    Handles chunks intelligently for natural speech
    """
    
    def __init__(self, voice: str = None, rate: int = 180):
        """
        Args:
            voice: Voice name (e.g., "David", "Zira" on Windows)
            rate: Speech rate (words per minute)
        """
        self.is_speaking = False
        self.should_stop = False
        self.lock = threading.Lock()
        
        # Initialize pyttsx3 engine
        print("🔊 Initializing pyttsx3 TTS engine...")
        self.engine = pyttsx3.init()
        
        # Configure engine
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', 1.0)
        
        # Set voice if specified
        if voice:
            voices = self.engine.getProperty('voices')
            for v in voices:
                if voice.lower() in v.name.lower():
                    self.engine.setProperty('voice', v.id)
                    print(f"✅ Voice set to: {v.name}")
                    break
        
        # Smart buffering for natural speech
        self.text_buffer = ""
        self.buffer_lock = threading.Lock()
        self.last_speak_time = 0
        self.min_speak_interval = 0.05  # 50ms minimum between speaking
        
        # Queue for chunks
        self.chunk_queue = queue.Queue()
        self.is_running = True
        
        # Start TTS worker thread
        self.tts_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True
        )
        self.tts_thread.start()
        
        # Start buffer flusher thread (speaks buffered text at intervals)
        self.flush_thread = threading.Thread(
            target=self._buffer_flusher,
            daemon=True
        )
        self.flush_thread.start()
        
        print("✅ Instant TTS ready (FREE, offline, zero-latency)")
    
    def speak_now(self, token: str):
        """
        Speak token IMMEDIATELY with intelligent buffering
        
        Args:
            token: Text chunk from LLM (word, phrase, or punctuation)
        """
        if not token or self.should_stop:
            return
        
        with self.lock:
            self.is_speaking = True
        
        # Add to buffer for processing
        with self.buffer_lock:
            self.text_buffer += token
            
            # Speak immediately if we hit a sentence boundary
            should_flush = self._should_flush_buffer()
            
            if should_flush:
                text_to_speak = self.text_buffer.strip()
                self.text_buffer = ""
                
                if text_to_speak:
                    self._queue_for_speaking(text_to_speak)
    
    def _should_flush_buffer(self) -> bool:
        """Check if buffer should be flushed (spoken)"""
        if not self.text_buffer:
            return False
        
        # Flush on sentence endings
        if any(self.text_buffer.rstrip().endswith(char) for char in '.!?'):
            return True
        
        # Flush on comma/semicolon if buffer is long enough
        if any(self.text_buffer.rstrip().endswith(char) for char in ',;:'):
            if len(self.text_buffer.split()) >= 3:
                return True
        
        # Flush if buffer gets too long (prevent lag)
        if len(self.text_buffer) > 100:
            return True
        
        return False
    
    def _queue_for_speaking(self, text: str):
        """Queue text for immediate speaking"""
        try:
            self.chunk_queue.put_nowait(text)
        except queue.Full:
            print("⚠️ TTS queue full")
    
    def _buffer_flusher(self):
        """
        Background thread that flushes buffer at regular intervals
        Ensures small chunks don't get stuck in buffer
        """
        while self.is_running:
            time.sleep(0.3)  # Check every 300ms
            
            if self.should_stop:
                continue
            
            with self.buffer_lock:
                if self.text_buffer.strip():
                    # Flush if buffer has been idle
                    text_to_speak = self.text_buffer.strip()
                    self.text_buffer = ""
                    self._queue_for_speaking(text_to_speak)
    
    def _tts_worker(self):
        """Worker thread that speaks queued text immediately"""
        while self.is_running:
            try:
                # Get text from queue
                text = self.chunk_queue.get(timeout=0.05)
                
                if text is None:  # Stop signal
                    break
                
                if self.should_stop:
                    self.chunk_queue.task_done()
                    continue
                
                # ⚡ SPEAK IMMEDIATELY
                self._speak_text(text)
                
                self.chunk_queue.task_done()
                
            except queue.Empty:
                with self.lock:
                    if self.chunk_queue.empty():
                        self.is_speaking = False
                continue
            except Exception as e:
                print(f"⚠️ TTS worker error: {e}")
    
    def _speak_text(self, text: str):
        """Actually speak the text using pyttsx3"""
        try:
            # Clean text
            clean_text = self._clean_text(text)
            
            if not clean_text or len(clean_text.strip()) < 1:
                return
            
            # Rate limiting to prevent engine overload
            current_time = time.time()
            time_since_last = current_time - self.last_speak_time
            
            if time_since_last < self.min_speak_interval:
                time.sleep(self.min_speak_interval - time_since_last)
            
            # ⚡ SPEAK
            self.engine.say(clean_text)
            self.engine.runAndWait()
            
            self.last_speak_time = time.time()
            
        except Exception as e:
            print(f"⚠️ Speech error: {e}")
            # Try to restart engine
            try:
                self.engine.stop()
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 180)
            except:
                pass
    
    def _clean_text(self, text: str) -> str:
        """Clean text for TTS"""
        if not text:
            return ""
        
        # Remove markdown
        text = re.sub(r'[*_`#\[\]]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        text = text.strip()
        
        return text
    
    def flush(self):
        """Force flush any buffered text"""
        with self.buffer_lock:
            if self.text_buffer.strip():
                text_to_speak = self.text_buffer.strip()
                self.text_buffer = ""
                self._queue_for_speaking(text_to_speak)
    
    def stop(self):
        """Stop speaking IMMEDIATELY (called during barge-in)"""
        with self.lock:
            self.should_stop = True
            self.is_speaking = False
        
        # Clear buffer
        with self.buffer_lock:
            self.text_buffer = ""
        
        # Stop engine
        try:
            self.engine.stop()
        except:
            pass
        
        # Clear queue
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
                self.chunk_queue.task_done()
            except queue.Empty:
                break
        
        # Reset stop flag
        threading.Thread(
            target=self._reset_stop_flag,
            daemon=True
        ).start()
    
    def _reset_stop_flag(self):
        """Reset stop flag after interruption"""
        time.sleep(0.1)
        with self.lock:
            self.should_stop = False
    
    def get_speaking_state(self) -> bool:
        """Check if currently speaking"""
        with self.lock:
            return self.is_speaking or not self.chunk_queue.empty()
    
    def set_voice(self, voice_name: str):
        """Change TTS voice"""
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if voice_name.lower() in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"✅ Voice changed to: {voice.name}")
                return
        print(f"⚠️ Voice '{voice_name}' not found")
    
    def set_rate(self, rate: int):
        """Change speech rate"""
        self.engine.setProperty('rate', rate)
        print(f"✅ Speech rate changed to: {rate}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        voices = self.engine.getProperty('voices')
        return [voice.name for voice in voices]
    
    def shutdown(self):
        """Clean shutdown"""
        self.is_running = False
        
        # Flush any remaining text
        self.flush()
        time.sleep(0.5)
        
        self.stop()
        
        # Send stop signals
        try:
            self.chunk_queue.put_nowait(None)
        except:
            pass
        
        # Wait for threads
        if self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=1.0)


class StreamingVoiceAssistant:
    """
    Voice assistant with intelligent chunk handling
    Speaks immediately but buffers smartly for natural speech
    """
    
    def __init__(self, rate: int = 185):
        self.tts = InstantTTS(rate=rate)
        print("🎙️ Streaming Voice Assistant initialized")
    
    def process_llm_token(self, token: str):
        """
        Process a token from LLM and speak it intelligently
        
        Args:
            token: Single token from LLM
        """
        if not token:
            return
        
        # Pass to TTS - it handles buffering
        self.tts.speak_now(token)
    
    def flush_remaining(self):
        """Speak any remaining buffered text"""
        self.tts.flush()
    
    def stop_speaking(self):
        """Stop speaking immediately"""
        self.tts.stop()
    
    def is_speaking(self) -> bool:
        """Check if speaking"""
        return self.tts.get_speaking_state()


# ============================================================================
# DEMO
# ============================================================================

async def demo():
    """Demo with realistic LLM streaming"""
    print("\n" + "="*60)
    print("🎯 DEMO: Intelligent TTS Streaming")
    print("="*60)
    
    assistant = StreamingVoiceAssistant()
    
    # Simulate realistic LLM token stream
    llm_tokens = [
        "Hello", "!", " ", "I", "'", "m", " ", "speaking", " ", "in", " ", "real", "-", "time", ".",
        " ", "Each", " ", "chunk", " ", "is", " ", "spoken", " ", "immediately", ",",
        " ", "but", " ", "with", " ", "natural", " ", "pauses", ".",
        " ", "This", " ", "creates", " ", "a", " ", "smooth", " ", "listening", " ", "experience", "!"
    ]
    
    print("\n🤖 AI speaking:")
    for token in llm_tokens:
        print(token, end="", flush=True)
        assistant.process_llm_token(token)
        await asyncio.sleep(0.03)  # Simulate LLM generation time
    
    # Flush any remaining
    assistant.flush_remaining()
    
    print("\n\n⏳ Waiting for speech to finish...")
    while assistant.is_speaking():
        await asyncio.sleep(0.1)
    
    print("✅ Demo complete!")
    assistant.tts.shutdown()


if __name__ == "__main__":
    asyncio.run(demo())