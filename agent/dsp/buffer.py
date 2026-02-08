
import numpy as np
import time

class SpeechBuffer:
    """
    Speech buffer with NO minimum length requirement
    """
    def __init__(self, sample_rate, pre_ms=200, post_ms=600, 
                 min_speech_ms=0, max_speech_ms=10000):  # min_speech_ms = 0!
        self.sample_rate = sample_rate
        self.pre_frames = int(sample_rate * pre_ms / 1000)
        self.post_frames = int(sample_rate * post_ms / 1000)
        self.min_speech_frames = int(sample_rate * min_speech_ms / 1000)  # ZERO!
        self.max_speech_frames = int(sample_rate * max_speech_ms / 1000)
        
        self.pre_buffer = []
        self.speech_frames = []
        self.post_counter = 0
        self.recording = False
        self.speech_start_time = 0
        
        print(f"📦 Buffer initialized: NO minimum length, accepts ALL speech")
    
    def push(self, frame, is_voice):
        """
        Accept ALL speech, even single frames
        """
        # Store pre-roll
        self.pre_buffer.append(frame)
        if len(self.pre_buffer) > self.pre_frames:
            self.pre_buffer.pop(0)
        
        if is_voice:
            if not self.recording:
                # START recording immediately
                self.recording = True
                self.speech_frames = self.pre_buffer.copy()
                self.speech_start_time = time.time()
                self.post_counter = 0
                
                duration_ms = (len(self.speech_frames) / self.sample_rate) * 1000
                print(f"🎤 Voice detected (pre-roll: {duration_ms:.0f}ms)")
            
            # Add frame
            self.speech_frames.append(frame)
            self.post_counter = 0
            
            # Check max length
            if len(self.speech_frames) > self.max_speech_frames:
                print("⏰ Max length reached, forcing segment")
                return self._finalize_segment()
        
        elif self.recording:
            # Silence during recording
            self.speech_frames.append(frame)
            self.post_counter += len(frame)
            
            # Check if enough post-silence
            if self.post_counter >= self.post_frames:
                # ALWAYS return segment, even if very short
                current_frames = len(self.speech_frames)
                current_ms = (current_frames / self.sample_rate) * 1000
                
                print(f"✅ Utterance complete: {current_ms:.0f}ms ({current_frames} frames)")
                return self._finalize_segment()
        
        return None
    
    def _finalize_segment(self):
        """Return whatever we have, no matter how short"""
        if not self.speech_frames:
            return None
        
        segment = np.concatenate(self.speech_frames)
        duration = len(segment) / self.sample_rate
        
        print(f"📤 Segment ready: {duration:.3f}s ({len(segment)} samples)")
        
        self._reset()
        return segment
    
    def _reset(self):
        """Reset buffer"""
        self.speech_frames = []
        self.post_counter = 0
        self.recording = False
        self.speech_start_time = 0
    
    def force_end(self):
        """Force end and return current segment"""
        if self.recording:
            print("⚡ Forcing segment end")
            return self._finalize_segment()
        return None
    
    def get_state(self):
        """Get current state"""
        current_frames = len(self.speech_frames)
        current_ms = (current_frames / self.sample_rate) * 1000 if current_frames > 0 else 0
        
        return {
            'recording': self.recording,
            'frames': current_frames,
            'duration_ms': current_ms,
            'post_counter': self.post_counter,
            'pre_buffer': len(self.pre_buffer)
        }
