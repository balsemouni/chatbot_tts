
# from langchain_ollama import ChatOllama
# import asyncio

# class StreamingLLMHandler:
#     """LLM handler that streams tokens ONE BY ONE"""
    
#     def __init__(self, model="llama3.1"):
#         self.llm = ChatOllama(
#             model=model,
#             base_url="http://localhost:11434",
#             temperature=0.7,
#             streaming=True,
#         )
    
#     def stream_response(self, messages):
#         """
#         Stream LLM response token by token
        
#         Args:
#             messages: Conversation history
            
#         Yields:
#             Each token as it's generated
#         """
#         # Create async generator
#         async def async_generator():
#             buffer = ""
            
#             try:
#                 async for chunk in self.llm.astream(messages):
#                     if hasattr(chunk, 'content') and chunk.content:
#                         content = chunk.content
                        
#                         # Stream character by character for lowest latency
#                         for char in content:
#                             buffer += char
                            
#                             # Yield on word boundaries
#                             if char in ' \t\n.,!?;':
#                                 if buffer.strip():
#                                     yield buffer
#                                 buffer = ""
                        
#                         # Yield any remaining
#                         if buffer.strip():
#                             yield buffer
#                             buffer = ""
#             except Exception as e:
#                 print(f"⚠️ LLM streaming error: {e}")
        
#         # Convert async generator to sync generator
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         try:
#             async_gen = async_generator()
            
#             while True:
#                 try:
#                     task = asyncio.wait_for(async_gen.__anext__(), timeout=5.0)
#                     token = loop.run_until_complete(task)
#                     yield token
#                 except asyncio.TimeoutError:
#                     print("⏱️  LLM timeout")
#                     break
#                 except StopAsyncIteration:
#                     break
#         finally:
#             loop.close()
    
#     async def stream_response_async(self, messages):
#         """Async version for use in async contexts"""
#         buffer = ""
        
#         async for chunk in self.llm.astream(messages):
#             if hasattr(chunk, 'content') and chunk.content:
#                 content = chunk.content
                
#                 for char in content:
#                     buffer += char
                    
#                     if char in ' \t\n.,!?;':
#                         if buffer.strip():
#                             yield buffer
#                         buffer = ""
                
#                 if buffer.strip():
#                     yield buffer
#                     buffer = ""
# from langchain_ollama import ChatOllama
# import asyncio

# class StreamingLLMHandler:
#     """LLM handler that streams tokens ONE BY ONE using StreamingLLMHandler"""
    
#     def __init__(self, model="llama3.2:3b"):
#         # Import and use the StreamingLLMHandler from your other file
#         from Llm_gpu.streaming_handler import StreamingLLM
        
#         # Initialize the handler which has GPU optimization built-in
#         self.handler = StreamingLLM(model=model)
    
#     def stream_response(self, messages):
#         """
#         Stream LLM response token by token
#         SAME LOGIC as before - just using different model source
        
#         Args:
#             messages: Conversation history
            
#         Yields:
#             Each token as it's generated
#         """
#         # Create async generator - SAME LOGIC
#         async def async_generator():
#             buffer = ""
            
#             try:
#                 # Use the handler's streaming method instead of direct llm.astream
#                 async for chunk in self.handler.llm.astream(messages):
#                     if hasattr(chunk, 'content') and chunk.content:
#                         content = chunk.content
                        
#                         # Stream character by character for lowest latency - SAME
#                         for char in content:
#                             buffer += char
                            
#                             # Yield on word boundaries - SAME
#                             if char in ' \t\n.,!?;':
#                                 if buffer.strip():
#                                     yield buffer
#                                 buffer = ""
                        
#                         # Yield any remaining - SAME
#                         if buffer.strip():
#                             yield buffer
#                             buffer = ""
#             except Exception as e:
#                 print(f"⚠️ LLM streaming error: {e}")
        
#         # Convert async generator to sync generator - SAME
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         try:
#             async_gen = async_generator()
            
#             while True:
#                 try:
#                     task = asyncio.wait_for(async_gen.__anext__(), timeout=5.0)
#                     token = loop.run_until_complete(task)
#                     yield token
#                 except asyncio.TimeoutError:
#                     print("⏱️  LLM timeout")
#                     break
#                 except StopAsyncIteration:
#                     break
#         finally:
#             loop.close()
    
#     async def stream_response_async(self, messages):
#         """Async version for use in async contexts - SAME LOGIC"""
#         buffer = ""
        
#         # Use the handler's llm instance
#         async for chunk in self.handler.llm.astream(messages):
#             if hasattr(chunk, 'content') and chunk.content:
#                 content = chunk.content
                
#                 for char in content:
#                     buffer += char
                    
#                     if char in ' \t\n.,!?;':
#                         if buffer.strip():
#                             yield buffer
#                         buffer = ""
                
#                 if buffer.strip():
#                     yield buffer
#                     buffer = ""
    
#     # Alternative: Use handler's own streaming method
#     def stream_response_via_handler(self, messages):
#         """
#         Alternative: Use the handler's built-in streaming method
#         This might have different chunking logic
#         """
#         return self.handler.stream_response(messages)
from Llm_gpu.streaming_handler import initialize_llm
import asyncio

class StreamingLLMHandler:
    """LLM handler that streams tokens ONE BY ONE using GPU-optimized LLM"""
    
    def __init__(self, model="llama3.2:3b", target_vram_mb=4096, model_base_size_mb=2200):
        """
        Initialize with GPU-managed LLM
        
        Args:
            model: Model name (default: "llama3.2:3b")
            target_vram_mb: Target VRAM allocation (default: 4096 for 4GB)
            model_base_size_mb: Base model size (default: 2200 for 3b model)
        """
        print("🚀 Initializing GPU-managed streaming LLM...")
        
        # Initialize LLM with GPU management
        self.llm = initialize_llm(
            model=model,
            target_vram_mb=target_vram_mb,
            model_base_size_mb=model_base_size_mb,
            auto_prepare_gpu=True,
            temperature=0.7
        )
        
        if not self.llm:
            raise RuntimeError("Failed to initialize GPU-managed LLM")
        
        # ChatOllama doesn't have a 'streaming' attribute
        # Streaming is controlled by using astream() method instead
        
        print("✅ Streaming LLM ready on GPU!")
    
    def stream_response(self, messages):
        """
        Stream LLM response token by token
        
        Args:
            messages: Conversation history (string or list of messages)
            
        Yields:
            Each token as it's generated
        """
        # Create async generator
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
        """Async version for use in async contexts"""
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


# ============================================================================
# DEMO USAGE
# ============================================================================

def demo_sync_streaming():
    """Demo: Synchronous streaming"""
    print("\n" + "="*60)
    print("DEMO: Synchronous Streaming")
    print("="*60)
    
    # Initialize handler (GPU management happens here)
    handler = StreamingLLMHandler(
        model="llama3.2:3b",
        target_vram_mb=4096,
        model_base_size_mb=2200
    )
    
    # Stream a response
    print("\n💬 Question: Tell me a short story about a robot")
    print("🤖 Response: ", end="", flush=True)
    
    for token in handler.stream_response("Tell me a short story about a robot"):
        print(token, end="", flush=True)
    
    print("\n\n✅ Done!")


async def demo_async_streaming():
    """Demo: Asynchronous streaming"""
    print("\n" + "="*60)
    print("DEMO: Asynchronous Streaming")
    print("="*60)
    
    # Initialize handler
    handler = StreamingLLMHandler()
    
    # Stream a response
    print("\n💬 Question: What are three benefits of exercise?")
    print("🤖 Response: ", end="", flush=True)
    
    async for token in handler.stream_response_async("What are three benefits of exercise?"):
        print(token, end="", flush=True)
    
    print("\n\n✅ Done!")


def demo_conversation():
    """Demo: Multi-turn conversation with streaming"""
    print("\n" + "="*60)
    print("DEMO: Interactive Conversation")
    print("="*60)
    
    handler = StreamingLLMHandler()
    
    conversation_history = []
    
    print("\n💬 Chat started! Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Stream response
        print("Assistant: ", end="", flush=True)
        response_text = ""
        
        for token in handler.stream_response(conversation_history):
            print(token, end="", flush=True)
            response_text += token
        
        print()  # New line after response
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": response_text
        })
    
    print("\nGoodbye!")


def demo_smaller_model():
    """Demo: Using smaller model (1B) for faster responses"""
    print("\n" + "="*60)
    print("DEMO: Smaller Model (llama3.2:1b)")
    print("="*60)
    
    # Use 1B model with 2GB VRAM
    handler = StreamingLLMHandler(
        model="llama3.2:1b",
        target_vram_mb=2048,
        model_base_size_mb=1300
    )
    
    print("\n💬 Question: What is Python?")
    print("🤖 Response: ", end="", flush=True)
    
    for token in handler.stream_response("What is Python?"):
        print(token, end="", flush=True)
    
    print("\n\n✅ Done!")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🎯 GPU-MANAGED STREAMING LLM HANDLER")
    print("="*60)
    
    print("\nAvailable demos:")
    print("1. Synchronous streaming")
    print("2. Asynchronous streaming")
    print("3. Interactive conversation")
    print("4. Smaller model (1B)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect demo (1-4): ").strip()
    
    if choice == "1":
        demo_sync_streaming()
    elif choice == "2":
        asyncio.run(demo_async_streaming())
    elif choice == "3":
        demo_conversation()
    elif choice == "4":
        demo_smaller_model()
    else:
        print("Invalid choice, running demo 1...")
        demo_sync_streaming()