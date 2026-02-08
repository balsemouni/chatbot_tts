"""
Streaming LLM Handler - Returns reusable model instance
Keeps same logic, just refactored for model reuse across files
"""
from langchain_ollama import ChatOllama
import asyncio


class StreamingLLMHandler:
    """LLM handler that streams tokens ONE BY ONE and provides reusable model"""
    
    def __init__(self, model="llama3.2:3b", num_ctx=17000, temperature=0.7):
        """
        Initialize handler with reusable LLM instance
        
        Args:
            model: Model name
            num_ctx: Context window size
            temperature: Sampling temperature
        """
        # Create the LLM instance - THIS IS WHAT OTHER FILES WILL USE
        self.llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=temperature,
            streaming=True,
            num_ctx=num_ctx,
            num_gpu=100,  # Keep all on GPU
        )
        
        self.model_name = model
        self.conversation_history = []
    
    # ========== CORE STREAMING METHODS (SAME LOGIC) ==========
    
    def stream_response(self, messages):
        """
        Stream LLM response token by token
        SAME LOGIC - streams character by character
        
        Args:
            messages: Conversation history
            
        Yields:
            Each token as it's generated
        """
        async def async_generator():
            buffer = ""
            
            try:
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        content = chunk.content
                        
                        # Stream character by character for lowest latency
                        for char in content:
                            buffer += char
                            
                            # Yield on word boundaries
                            if char in ' \t\n.,!?;':
                                if buffer.strip():
                                    yield buffer
                                buffer = ""
                        
                        # Yield any remaining
                        if buffer.strip():
                            yield buffer
                            buffer = ""
            except Exception as e:
                print(f"⚠️ LLM streaming error: {e}")
        
        # Convert async generator to sync generator
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            async_gen = async_generator()
            
            while True:
                try:
                    task = asyncio.wait_for(async_gen.__anext__(), timeout=5.0)
                    token = loop.run_until_complete(task)
                    yield token
                except asyncio.TimeoutError:
                    print("⏱️  LLM timeout")
                    break
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
    
    async def stream_response_async(self, messages):
        """Async version for use in async contexts - SAME LOGIC"""
        buffer = ""
        
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                
                for char in content:
                    buffer += char
                    
                    if char in ' \t\n.,!?;':
                        if buffer.strip():
                            yield buffer
                        buffer = ""
                
                if buffer.strip():
                    yield buffer
                    buffer = ""
    
    # ========== MODEL ACCESS METHODS ==========
    
    def get_model(self):
        """
        Get the underlying LLM instance for use in other files
        
        Returns:
            ChatOllama instance that can be used anywhere
        """
        return self.llm
    
    def invoke(self, messages):
        """
        Direct invoke (non-streaming) using the same model
        
        Args:
            messages: List of message dicts
            
        Returns:
            Response content as string
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.llm.ainvoke(messages))
            loop.close()
            return response.content
        except Exception as e:
            print(f"⚠️ Error invoking model: {e}")
            return None
    
    async def ainvoke(self, messages):
        """Async invoke"""
        response = await self.llm.ainvoke(messages)
        return response.content
    
    # ========== CONVENIENCE METHODS ==========
    
    def add_message(self, role, content):
        """Add a message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_messages(self):
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def chat_stream(self, user_input, system_prompt=None):
        """
        Stream chat response
        
        Args:
            user_input: User message
            system_prompt: Optional system prompt
            
        Yields:
            Tokens as they're generated
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # Add to history
        self.add_message("user", user_input)
        
        full_response = ""
        for token in self.stream_response(messages):
            yield token
            full_response += token
        
        # Add assistant response to history
        if full_response:
            self.add_message("assistant", full_response)
    
    def chat(self, user_input, system_prompt=None, stream=True, print_response=True):
        """
        Chat interface
        
        Args:
            user_input: User message
            system_prompt: Optional system prompt
            stream: Whether to stream response
            print_response: Whether to print response
            
        Returns:
            Full response string or streaming generator
        """
        if stream:
            generator = self.chat_stream(user_input, system_prompt)
            
            if print_response:
                full_response = ""
                print("\n🤖 Assistant: ", end="", flush=True)
                for token in generator:
                    print(token, end="", flush=True)
                    full_response += token
                print()
                return full_response
            else:
                return generator
        else:
            # Non-streaming response
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_input})
            
            response_text = self.invoke(messages)
            
            if response_text:
                # Add to history
                self.add_message("user", user_input)
                self.add_message("assistant", response_text)
                
                if print_response:
                    print(f"\n🤖 Assistant: {response_text}")
            
            return response_text


# ========== GLOBAL MODEL INSTANCE (FOR SHARING ACROSS FILES) ==========

_global_handler = None

def get_shared_handler(model="llama3.2:3b", num_ctx=17000, force_new=False):
    """
    Get or create a shared handler instance
    This allows multiple files to use the same loaded model
    
    Args:
        model: Model name
        num_ctx: Context window
        force_new: Force create new instance
        
    Returns:
        StreamingLLMHandler instance
    """
    global _global_handler
    
    if _global_handler is None or force_new:
        print(f"🔄 Loading model: {model}")
        _global_handler = StreamingLLMHandler(model=model, num_ctx=num_ctx)
        print(f"✅ Model loaded and ready")
    
    return _global_handler

def get_shared_model():
    """
    Get the shared LLM instance directly
    
    Returns:
        ChatOllama instance ready to use
    """
    handler = get_shared_handler()
    return handler.get_model()


# ========== QUICK USAGE FUNCTIONS ==========

def create_handler(model="llama3.2:3b", num_ctx=17000):
    """
    Quick function to create a StreamingLLMHandler instance
    
    Args:
        model: Model name
        num_ctx: Context window size
        
    Returns:
        StreamingLLMHandler instance
    """
    return StreamingLLMHandler(model=model, num_ctx=num_ctx)


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    print("="*60)
    print("StreamingLLMHandler - Reusable Model Demo")
    print("="*60)
    
    # METHOD 1: Create your own handler
    print("\n📝 Method 1: Create your own handler")
    handler = create_handler(model="llama3.2:3b")
    
    # Use it for streaming
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python in one sentence?"}
    ]
    
    print("🤖 Response: ", end="", flush=True)
    for token in handler.stream_response(messages):
        print(token, end="", flush=True)
    print()
    
    # METHOD 2: Get the model instance to use elsewhere
    print("\n📝 Method 2: Get model instance for reuse")
    llm = handler.get_model()
    print(f"✅ Got model instance: {type(llm).__name__}")
    print("   You can now pass 'llm' to other files/functions")
    
    # METHOD 3: Use shared global handler
    print("\n📝 Method 3: Use shared global handler")
    shared_handler = get_shared_handler()
    shared_model = get_shared_model()
    
    print(f"✅ Shared handler ready")
    print(f"✅ Shared model ready: {type(shared_model).__name__}")
    
    # Test it
    print("\n🧪 Testing shared handler:")
    response = shared_handler.chat("Say 'hello' in one word", stream=False, print_response=False)
    print(f"   Response: {response}")
    
    print("\n" + "="*60)
    print("✅ All methods work! You can now:")
    print("   1. Create handler: handler = create_handler()")
    print("   2. Get model: llm = handler.get_model()")
    print("   3. Use in other files: from streaming_llm_handler import get_shared_model")
    print("="*60)