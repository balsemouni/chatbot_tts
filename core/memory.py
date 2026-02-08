
from collections import deque
from datetime import datetime

class ConversationMemory:
    """Manages conversation history with streaming support"""
    
    def __init__(self, limit=20):
        self.history = deque(maxlen=limit)
        self.system_prompt = (
            "You are a helpful, witty AI voice assistant. "
            "Keep responses concise and conversational. "
            "Respond in real-time as the user speaks."
        )
    
    def add_exchange(self, user_text: str, ai_text: str, 
                    timestamp: datetime = None, interrupted: bool = False):
        """
        Add conversation exchange to history
        
        Args:
            user_text: User's speech
            ai_text: AI's response
            timestamp: When the exchange occurred
            interrupted: Whether the exchange was interrupted
        """
        if not timestamp:
            timestamp = datetime.now()
        
        exchange = {
            'user': user_text,
            'ai': ai_text,
            'timestamp': timestamp,
            'interrupted': interrupted
        }
        
        self.history.append(exchange)
    
    def get_messages(self) -> list:
        """Get messages in LLM format"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for exchange in self.history:
            messages.append({"role": "user", "content": exchange['user']})
            messages.append({"role": "assistant", "content": exchange['ai']})
        
        return messages
    
    def get_full_history(self) -> list:
        """Get full conversation history"""
        return list(self.history)
    
    def get_last_n(self, n: int) -> list:
        """Get last n exchanges"""
        return list(self.history)[-n:] if n > 0 else []
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
        print("🗑️  Conversation history cleared")
    
    def get_summary(self) -> str:
        """Get conversation summary"""
        if not self.history:
            return "No conversation history"
        
        summary = []
        for i, exchange in enumerate(self.history, 1):
            summary.append(f"Turn {i}:")
            summary.append(f"  User: {exchange['user'][:50]}...")
            summary.append(f"  AI: {exchange['ai'][:50]}...")
        
        return "\n".join(summary)
