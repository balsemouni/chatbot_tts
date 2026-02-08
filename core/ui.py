import sys
import time
from datetime import datetime

class UIHandler:
    """Clean, minimal UI for voice agent"""
    
    def __init__(self):
        self.last_update = 0
        self.last_status = ""
        self.showing_status = False
        self.current_user_text = []
        self.current_ai_text = ""
        
    def update_status(self, state, vad_prob, rms, silence_ms=None):
        """Update status display - only show important changes"""
        current_time = time.time()
        
        # Only show state transitions, not continuous updates
        if state == "speaking" and self.last_status != "speaking":
            self._print_clean("\n🎤 Listening...")
            self.showing_status = True
            self.last_status = "speaking"
            
        elif state == "idle" and self.last_status == "silent":
            # Transition from silent to idle means processing
            if self.showing_status:
                self._clear_status_line()
                self.showing_status = False
            self.last_status = "idle"
            
        elif state == "silent":
            self.last_status = "silent"
    
    def print_user_speech(self, word):
        """Accumulate user speech words"""
        self.current_user_text.append(word)
    
    def print_user_complete(self, full_text):
        """Print complete user utterance"""
        if self.showing_status:
            self._clear_status_line()
            self.showing_status = False
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 👤 You: {full_text}")
        self.current_user_text = []
    
    def start_ai_response(self):
        """Start AI response"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 🤖 AI: ", end="", flush=True)
        self.current_ai_text = ""
    
    def print_ai_token(self, token):
        """Print AI token as it streams"""
        # Clean display of tokens
        display_token = token
        # Add space before if not punctuation and previous char wasn't space
        if self.current_ai_text and not token[0] in ' \t\n.,!?;:' and self.current_ai_text[-1] not in ' \t\n':
            display_token = ' ' + token
        
        print(display_token, end="", flush=True)
        self.current_ai_text += token
    
    def finish_ai_response(self):
        """Finish AI response"""
        print()  # New line after AI completes
    
    def print_interruption(self):
        """Print interruption notice"""
        print(" [interrupted]")
    
    def print_thinking(self):
        """Show AI is thinking"""
        print("💭 Thinking...", end="", flush=True)
    
    def clear_thinking(self):
        """Clear thinking indicator"""
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()
    
    def print_error(self, msg):
        """Print error message"""
        print(f"\n❌ Error: {msg}")
    
    def print_info(self, msg):
        """Print info message"""
        print(f"\n💡 {msg}")
    
    def _print_clean(self, msg):
        """Print message cleanly"""
        if self.showing_status:
            self._clear_status_line()
        print(msg, flush=True)
    
    def _clear_status_line(self):
        """Clear the status line"""
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
    
    def print_conversation_history(self, history):
        """Print full conversation history in a clean format"""
        if not history:
            print("\n📭 No conversation history yet")
            return
        
        print("\n" + "="*70)
        print("📜 CONVERSATION HISTORY")
        print("="*70)
        
        for i, turn in enumerate(history, 1):
            timestamp = turn.get('timestamp', '')
            if hasattr(timestamp, 'strftime'):
                timestamp = timestamp.strftime('%H:%M:%S')
            
            print(f"\n[{timestamp}] Turn {i}")
            print(f"  👤 You: {turn['user']}")
            print(f"  🤖 AI:  {turn['ai']}")
        
        print("\n" + "="*70)
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*70)
        print("⚡ VOICE ASSISTANT - READY")
        print("="*70)
        print("💡 Speak naturally - I'll respond when you finish")
        print("🔇 Pause for 700ms to submit your message")
        print("⚡ Interrupt me anytime by speaking")
        print("🛑 Press Ctrl+C to exit and see history")
        print("="*70 + "\n")