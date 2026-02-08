import asyncio
import threading
import queue
import time
from typing import Optional, List
import pyttsx3
import re


class InstantTTS:
    """
    FREE, OFFLINE, ZERO-LATENCY TTS with ENHANCED VOICE QUALITY
    Speaks IMMEDIATELY as tokens arrive from LLM
    Handles chunks intelligently for natural speech
    """

    def __init__(self, voice: str = None, rate: int = 180, volume: float = 0.95):
        """
        Args:
            voice: Voice name (e.g., "David", "Zira" on Windows)
            rate: Speech rate (words per minute, 150-200 recommended)
            volume: Volume level (0.0 to 1.0)
        """
        self.is_speaking = False
        self.should_stop = False
        self.lock = threading.Lock()

        # Initialize pyttsx3 engine with better settings
        print("🔊 Initializing enhanced pyttsx3 TTS engine...")
        self.engine = pyttsx3.init()

        # Configure engine for better quality
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Try to select the best available voice
        self._select_best_voice(voice)

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
            target=self._tts_worker, daemon=True
        )
        self.tts_thread.start()

        # Start buffer flusher thread (speaks buffered text at intervals)
        self.flush_thread = threading.Thread(
            target=self._buffer_flusher, daemon=True
        )
        self.flush_thread.start()

        print("✅ Enhanced Instant TTS ready (FREE, offline, zero-latency)")

    def _select_best_voice(self, preferred_voice: Optional[str] = None):
        """Select the best available voice with preference for higher quality"""
        voices = self.engine.getProperty('voices')
        
        if not voices:
            print("⚠️ No voices available")
            return

        selected_voice = None

        # If user specified a voice, try to find it
        if preferred_voice:
            for v in voices:
                if preferred_voice.lower() in v.name.lower():
                    selected_voice = v
                    break

        # Otherwise, select best quality voice automatically
        if not selected_voice:
            # Priority order for better quality voices
            quality_priorities = [
                # Windows high-quality voices
                'hazel', 'zira', 'david', 'mark',
                # macOS high-quality voices  
                'samantha', 'alex', 'victoria', 'karen',
                # Linux espeak voices (less quality but available)
                'english', 'en-us', 'en-gb'
            ]

            for priority in quality_priorities:
                for v in voices:
                    if priority in v.name.lower():
                        selected_voice = v
                        break
                if selected_voice:
                    break

            # Fallback to first available voice
            if not selected_voice and voices:
                selected_voice = voices[0]

        if selected_voice:
            self.engine.setProperty('voice', selected_voice.id)
            print(f"✅ Voice: {selected_voice.name}")
            
            # Print additional voice info if available
            if hasattr(selected_voice, 'languages'):
                print(f"   Languages: {selected_voice.languages}")
            if hasattr(selected_voice, 'gender'):
                print(f"   Gender: {selected_voice.gender}")

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

        buffer_stripped = self.text_buffer.rstrip()

        # Flush on sentence endings
        if any(buffer_stripped.endswith(char) for char in '.!?'):
            return True

        # Flush on comma/semicolon if buffer is long enough
        if any(buffer_stripped.endswith(char) for char in ',;:'):
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
        """Actually speak the text using pyttsx3 with enhanced processing"""
        try:
            # Clean and enhance text
            clean_text = self._clean_and_enhance_text(text)
            
            if not clean_text or len(clean_text.strip()) < 1:
                return

            # Rate limiting to prevent engine overload
            current_time = time.time()
            time_since_last = current_time - self.last_speak_time
            
            if time_since_last < self.min_speak_interval:
                time.sleep(self.min_speak_interval - time_since_last)

            # ⚡ SPEAK with better quality
            self.engine.say(clean_text)
            self.engine.runAndWait()
            
            self.last_speak_time = time.time()

        except Exception as e:
            print(f"⚠️ Speech error: {e}")
            # Try to restart engine
            try:
                self.engine.stop()
                self._reinitialize_engine()
            except:
                pass

    def _reinitialize_engine(self):
        """Reinitialize engine after error"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 180)
            self.engine.setProperty('volume', 0.95)
            self._select_best_voice()
        except Exception as e:
            print(f"⚠️ Engine reinit error: {e}")

    def _clean_and_enhance_text(self, text: str) -> str:
        """Clean and enhance text for better TTS quality"""
        if not text:
            return ""

        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
        text = re.sub(r'__(.+?)__', r'\1', text)      # Underline
        text = re.sub(r'_(.+?)_', r'\1', text)        # Italic
        text = re.sub(r'`(.+?)`', r'\1', text)        # Code
        text = re.sub(r'[#]+\s*', '', text)           # Headers
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)  # Links

        # Improve number pronunciation
        text = re.sub(r'\b(\d+)%', r'\1 percent', text)
        text = re.sub(r'\$(\d+)', r'\1 dollars', text)
        
        # Add natural pauses
        text = text.replace(' - ', ', ')
        text = text.replace('...', '.')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Fix common abbreviations for better pronunciation
        abbreviations = {
            ' e.g. ': ' for example ',
            ' i.e. ': ' that is ',
            ' etc.': ' and so on',
            ' vs. ': ' versus ',
            ' Dr. ': ' Doctor ',
            ' Mr. ': ' Mister ',
            ' Mrs. ': ' Misses ',
            ' Ms. ': ' Miss ',
        }
        
        for abbr, replacement in abbreviations.items():
            text = text.replace(abbr, replacement)

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
            target=self._reset_stop_flag, daemon=True
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
        """
        Change speech rate
        
        Args:
            rate: Words per minute (recommended: 150-200)
                  150 = slower, clearer
                  180 = natural
                  200 = faster
        """
        self.engine.setProperty('rate', rate)
        print(f"✅ Speech rate: {rate} WPM")

    def set_volume(self, volume: float):
        """
        Change volume
        
        Args:
            volume: 0.0 to 1.0
        """
        volume = max(0.0, min(1.0, volume))
        self.engine.setProperty('volume', volume)
        print(f"✅ Volume: {int(volume * 100)}%")

    def get_available_voices(self) -> List[dict]:
        """Get detailed list of available voices"""
        voices = self.engine.getProperty('voices')
        voice_list = []
        
        for v in voices:
            voice_info = {
                'name': v.name,
                'id': v.id,
                'languages': getattr(v, 'languages', []),
                'gender': getattr(v, 'gender', 'unknown')
            }
            voice_list.append(voice_info)
        
        return voice_list

    def print_available_voices(self):
        """Print formatted list of available voices"""
        print("\n" + "="*60)
        print("📢 Available Voices:")
        print("="*60)
        
        voices = self.get_available_voices()
        for i, v in enumerate(voices, 1):
            print(f"{i}. {v['name']}")
            if v['languages']:
                print(f"   Languages: {v['languages']}")
            if v['gender'] != 'unknown':
                print(f"   Gender: {v['gender']}")
            print()

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
    Voice assistant with intelligent chunk handling and enhanced voice quality
    Speaks immediately but buffers smartly for natural speech
    """

    def __init__(self, rate: int = 175, volume: float = 0.95, voice: str = None):
        """
        Args:
            rate: Speech rate in WPM (150-200 recommended)
            volume: Volume level 0.0-1.0
            voice: Specific voice name to use
        """
        self.tts = InstantTTS(rate=rate, volume=volume, voice=voice)
        print("🎙️ Enhanced Streaming Voice Assistant initialized")

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

    def list_voices(self):
        """Show available voices"""
        self.tts.print_available_voices()

    def change_voice(self, voice_name: str):
        """Change voice"""
        self.tts.set_voice(voice_name)

    def adjust_speed(self, rate: int):
        """Adjust speaking speed (150-200 recommended)"""
        self.tts.set_rate(rate)

    def adjust_volume(self, volume: float):
        """Adjust volume (0.0 to 1.0)"""
        self.tts.set_volume(volume)


# ============================================================================
# DEMO
# ============================================================================

async def demo():
    """Demo with realistic LLM streaming and enhanced voice"""
    print("\n" + "="*60)
    print("🎯 DEMO: Enhanced Intelligent TTS Streaming")
    print("="*60)

    # Create assistant with optimized settings
    assistant = StreamingVoiceAssistant(
        rate=175,      # Slightly slower for clarity
        volume=0.95    # High but not distorted
    )

    print("\n📋 Available voices:")
    assistant.list_voices()

    # Simulate realistic LLM token stream with various content
    llm_tokens = [
        "Hello", "!", " ", "I", "'", "m", " ", "speaking", " ", 
        "with", " ", "enhanced", " ", "voice", " ", "quality", ".", " ",
        "Notice", " ", "how", " ", "I", " ", "handle", " ", "numbers", " ",
        "like", " ", "100", "%", " ", "and", " ", "$", "50", ",", " ",
        "abbreviations", " ", "like", " ", "e.g.", " ", "and", " ", "i.e.", ",", " ",
        "and", " ", "natural", " ", "pauses", ".", " ",
        "The", " ", "text", " ", "is", " ", "cleaned", " ", "and", " ",
        "optimized", " ", "for", " ", "the", " ", "best", " ", "listening", " ",
        "experience", "!", " ",
        "This", " ", "creates", " ", "smooth", ",", " ", "natural", " ", 
        "sounding", " ", "speech", "."
    ]

    print("\n🤖 AI speaking with enhanced voice:")
    print("-" * 60)
    
    for token in llm_tokens:
        print(token, end="", flush=True)
        assistant.process_llm_token(token)
        await asyncio.sleep(0.03)  # Simulate LLM generation time

    # Flush any remaining
    assistant.flush_remaining()

    print("\n" + "-" * 60)
    print("\n⏳ Waiting for speech to finish...")
    
    while assistant.is_speaking():
        await asyncio.sleep(0.1)

    print("✅ Demo complete!")
    print("\n💡 Tips for better voice quality:")
    print("   • Adjust rate: assistant.adjust_speed(150-200)")
    print("   • Change voice: assistant.change_voice('name')")
    print("   • Adjust volume: assistant.adjust_volume(0.8-1.0)")
    
    assistant.tts.shutdown()


async def interactive_demo():
    """Interactive demo to test different voices and settings"""
    print("\n" + "="*60)
    print("🎛️  INTERACTIVE VOICE DEMO")
    print("="*60)

    assistant = StreamingVoiceAssistant()

    test_phrases = [
        "This is a test of the enhanced voice quality.",
        "Numbers like 100% and $50 are pronounced clearly.",
        "Abbreviations like e.g. and i.e. sound natural.",
        "The quick brown fox jumps over the lazy dog."
    ]

    print("\n📝 Testing different settings...\n")

    # Test different rates
    for rate in [160, 175, 190]:
        print(f"\n🔊 Testing rate: {rate} WPM")
        assistant.adjust_speed(rate)
        
        for phrase in test_phrases[:1]:
            for char in phrase:
                assistant.process_llm_token(char)
                await asyncio.sleep(0.02)
            
            assistant.flush_remaining()
            
            while assistant.is_speaking():
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(0.5)

    print("\n✅ Interactive demo complete!")
    assistant.tts.shutdown()


if __name__ == "__main__":
    # Run main demo
    asyncio.run(demo())
    
    # Uncomment to run interactive demo
    # asyncio.run(interactive_demo())