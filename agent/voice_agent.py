import os
import torch
import numpy as np
import asyncio
import threading
import queue
import time
from datetime import datetime
from core.memory import ConversationMemory
from core.microphone import MicrophoneHandler
from core.vad import VoiceActivityDetector
from core.asr import StreamingSpeechRecognizer
from core.llm import StreamingLLMHandler
from core.tts import InstantTTS, StreamingVoiceAssistant
from core.ui import UIHandler

class UltraLowLatencyVoiceAgent:
    """
    ZERO-LATENCY Voice Agent with INSTANT TTS
    - Transcribes speech in real-time (shows progress)
    - Sends COMPLETE utterance to LLM only when finished
    - Each LLM token spoken IMMEDIATELY as it arrives
    - FREE pyttsx3 backend (no API keys, no internet)
    """
    
    def __init__(self):
        # Audio settings
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 512
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"⚡ Initializing ULTRA-LOW-LATENCY Voice Agent on {self.device.upper()}")
        
        # Core components
        self.memory = ConversationMemory()
        self.microphone = MicrophoneHandler(self.SAMPLE_RATE, self.CHUNK_SIZE)
        
        # VAD with optimized settings
        self.vad = VoiceActivityDetector(
            sample_rate=self.SAMPLE_RATE,
            device=self.device,
            idle_threshold=0.30,
            barge_in_threshold=0.60,
            min_rms=0.008,
            silence_limit_ms=700,
            enable_noise_reduction=True,
            min_chunk_samples=512,
        )
        
        # Streaming components
        self.asr = StreamingSpeechRecognizer("base.en", self.device)
        self.llm = StreamingLLMHandler(model="llama3.2:3b")
        
        # ⚡ INSTANT TTS - FREE, OFFLINE, ZERO-LATENCY
        self.tts = InstantTTS(
            voice=None,  # Auto-select best voice
            rate=200  # Slightly faster for real-time feel
        )
        
        # Voice assistant for smart token handling
        self.voice_assistant = StreamingVoiceAssistant()
        
        self.ui = UIHandler()
        
        # Pipeline state
        self.pipeline_active = False
        self.pipeline_lock = threading.Lock()
        self.last_voice_time = 0
        self.silence_threshold_ms = 700
        
        # Queues
        self.audio_queue = queue.Queue(maxsize=10)
        
        # Current state
        self.current_audio_buffer = []
        self.user_is_talking = False
        self.ai_is_thinking = False
        self.ai_is_speaking = False
        
        # Thread control
        self.running = True
        self.threads = []
        
        # Event loop for LLM operations (shared across all LLM calls)
        self.llm_loop = None
        self.llm_thread = None
        
        print("✅ System ready")
    
    def _start_llm_loop(self):
        """Start dedicated event loop for LLM operations in separate thread"""
        def run_loop():
            self.llm_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.llm_loop)
            self.llm_loop.run_forever()
        
        self.llm_thread = threading.Thread(target=run_loop, daemon=True)
        self.llm_thread.start()
        
        # Wait for loop to be ready
        while self.llm_loop is None:
            time.sleep(0.01)
    
    def audio_capture_thread(self):
        """Capture audio from microphone"""
        with self.microphone.start():
            while self.running:
                chunk = self.microphone.get_audio_chunk(timeout=0.01)
                if chunk is not None:
                    try:
                        self.audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass
    
    def vad_processor_thread(self):
        """Process audio through VAD"""
        last_voice_time = time.time()
        recording_buffer = []
        
        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.01)
                
                # Check if AI is speaking
                ai_speaking = self.tts.get_speaking_state()
                
                # Process through VAD
                segment, is_voice, prob, rms = self.vad.process_chunk(
                    chunk, 
                    ai_is_speaking=ai_speaking
                )
                
                current_time = time.time()
                
                if is_voice:
                    # VOICE DETECTED
                    last_voice_time = current_time
                    self.last_voice_time = current_time
                    
                    # Start recording if not already
                    if not self.user_is_talking:
                        self.user_is_talking = True
                        recording_buffer = [chunk]
                        
                        # Stop AI if speaking (barge-in)
                        if ai_speaking:
                            self.tts.stop()
                            self.ui.print_interruption()
                            self.ai_is_thinking = False
                            self.ai_is_speaking = False
                    else:
                        # Continue recording
                        recording_buffer.append(chunk)
                    
                    # Update UI
                    self.ui.update_status("speaking", prob, rms)
                    
                elif self.user_is_talking:
                    # SILENCE during recording
                    silence_duration = (current_time - last_voice_time) * 1000
                    recording_buffer.append(chunk)
                    
                    # Update UI
                    self.ui.update_status("silent", prob, rms, int(silence_duration))
                    
                    # Check if silence long enough to stop
                    if silence_duration > self.silence_threshold_ms:
                        # END OF SPEECH - process complete utterance
                        if recording_buffer:
                            audio_segment = np.concatenate(recording_buffer)
                            self._process_complete_utterance(audio_segment)
                        
                        # Reset state
                        self.user_is_talking = False
                        recording_buffer = []
                        self.ui.update_status("idle", prob, rms)
                
                else:
                    # IDLE state
                    self.ui.update_status("idle", prob, rms)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.ui.print_error(f"VAD error: {e}")
    
    def _process_complete_utterance(self, audio_data):
        """
        Process complete user utterance:
        1. Transcribe full audio (show progress)
        2. Send COMPLETE text to LLM once
        3. Stream LLM response with instant TTS
        """
        if len(audio_data) < self.SAMPLE_RATE * 0.2:  # Less than 200ms
            return
        
        # Run in separate thread
        threading.Thread(
            target=self._transcribe_and_respond,
            args=(audio_data,),
            daemon=True
        ).start()
    
    def _transcribe_and_respond(self, audio_data):
        """Transcribe complete utterance, then send to LLM"""
        try:
            # Step 1: Transcribe (show words as they come)
            words = []
            
            for word in self.asr.transcribe_streaming(audio_data):
                if not self.running or self.user_is_talking:
                    # New speech started during transcription
                    return
                
                word = word.strip()
                if word:
                    words.append(word)
                    self.ui.print_user_speech(word)
            
            # Step 2: Complete transcription
            if not words:
                return
            
            complete_text = " ".join(words)
            
            # Don't process if new speech started
            if self.user_is_talking:
                return
            
            # Show complete user text
            self.ui.print_user_complete(complete_text)
            
            # Step 3: Send ONCE to LLM and stream response
            self._get_llm_response(complete_text)
            
        except Exception as e:
            self.ui.print_error(f"Transcription error: {e}")
    
    def _get_llm_response(self, user_text):
        """Get LLM response and speak it token-by-token"""
        if not user_text or self.ai_is_thinking:
            return
        
        try:
            self.ai_is_thinking = True
            self.ai_is_speaking = True
            
            # Get conversation context
            messages = self.memory.get_messages()
            messages.append({"role": "user", "content": user_text})
            
            # Show thinking indicator
            self.ui.print_thinking()
            
            # Run async streaming in the dedicated LLM loop
            future = asyncio.run_coroutine_threadsafe(
                self._stream_llm_tokens(messages),
                self.llm_loop
            )
            
            # Wait for completion
            result = future.result(timeout=30)
            response_text = result.get('text', '')
            token_count = result.get('tokens', 0)
            
            # Save conversation
            if response_text.strip() and not self.user_is_talking:
                self.memory.add_exchange(user_text, response_text)
            
            self.ai_is_thinking = False
            self.ai_is_speaking = False
            
        except Exception as e:
            self.ui.print_error(f"LLM error: {e}")
            self.ai_is_thinking = False
            self.ai_is_speaking = False
    
    async def _stream_llm_tokens(self, messages):
        """Stream LLM tokens and speak each one immediately"""
        response_text = ""
        token_count = 0
        started = False
        
        try:
            async for token in self.llm.stream_response_async(messages):
                # Check for interruption
                if not self.running or self.user_is_talking:
                    self.tts.stop()
                    break
                
                # Clear thinking indicator on first token
                if not started:
                    self.ui.clear_thinking()
                    self.ui.start_ai_response()
                    started = True
                
                response_text += token
                token_count += 1
                
                # ⚡ SPEAK TOKEN IMMEDIATELY
                self.tts.speak_now(token)
                
                # Print token
                self.ui.print_ai_token(token)
            
            # Finish response
            if started:
                self.ui.finish_ai_response()
            
            return {'text': response_text, 'tokens': token_count}
            
        except Exception as e:
            self.ui.print_error(f"Token streaming error: {e}")
            return {'text': response_text, 'tokens': token_count}
    
    async def run(self):
        """Main event loop"""
        self.ui.print_welcome()
        
        # Start dedicated LLM event loop
        self._start_llm_loop()
        
        # Start processing threads
        threads = [
            threading.Thread(target=self.audio_capture_thread, daemon=True),
            threading.Thread(target=self.vad_processor_thread, daemon=True),
        ]
        
        for thread in threads:
            thread.start()
        
        self.threads = threads
        
        try:
            # Keep main thread alive
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            self.running = False
            
            # Stop TTS
            self.tts.stop()
            
            # Stop LLM loop
            if self.llm_loop:
                self.llm_loop.call_soon_threadsafe(self.llm_loop.stop)
            
            # Wait for threads
            for thread in self.threads:
                thread.join(timeout=0.5)
            
            # Show conversation history
            self._show_history()
    
    def _show_history(self):
        """Show conversation history on exit"""
        history = self.memory.get_full_history()
        self.ui.print_conversation_history(history)
    
    def get_conversation_summary(self):
        """Get conversation summary"""
        return self.memory.get_summary()


# ============================================================================
# LAUNCHER
# ============================================================================

async def main():
    """Launch the voice agent"""
    agent = UltraLowLatencyVoiceAgent()
    await agent.run()

if __name__ == "__main__":
    # Install check
    try:
        import pyttsx3
    except ImportError:
        print("❌ pyttsx3 not installed! Run: pip install pyttsx3")
        exit(1)
    
    # Run the agent
    asyncio.run(main())