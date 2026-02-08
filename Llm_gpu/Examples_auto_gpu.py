"""
Complete Usage Examples with Auto GPU Freeing
Shows all the ways to use the auto-optimized model
"""

# ========================================================================
# EXAMPLE 1: Simple Auto-Optimized Usage (RECOMMENDED)
# ========================================================================

def example_1_auto_optimized():
    """
    Simplest way - Auto GPU optimizer handles everything
    Automatically frees GPU processes if needed
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Auto-Optimized (Easiest)")
    print("="*60)
    
    from auto_gpu_optimizer import get_auto_optimized_handler
    
    # Get handler - it will:
    # 1. Check GPU memory
    # 2. Free non-critical processes if needed
    # 3. Load model on GPU
    # 4. Return ready-to-use handler
    handler = get_auto_optimized_handler(
        model="llama3.2:3b",
        target_mb=4096,
        auto_free=True  # ✅ This enables automatic process freeing
    )
    
    # Use it immediately
    handler.chat("What is machine learning?", stream=True)
    
    print("\n✅ That's it! GPU was optimized automatically")


# ========================================================================
# EXAMPLE 2: Get Model Only (For Other Files)
# ========================================================================

def example_2_get_model_only():
    """
    Get just the model instance for use in other files
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Get Model Instance")
    print("="*60)
    
    from auto_gpu_optimizer import get_auto_optimized_model
    
    # Get model (GPU auto-optimized)
    llm = get_auto_optimized_model(
        model="llama3.2:3b",
        target_mb=4096,
        auto_free=True
    )
    
    # Now use it
    import asyncio
    messages = [{"role": "user", "content": "Hello!"}]
    loop = asyncio.new_event_loop()
    response = loop.run_until_complete(llm.ainvoke(messages))
    loop.close()
    
    print(f"🤖 Response: {response.content}")


# ========================================================================
# EXAMPLE 3: Custom Configuration
# ========================================================================

def example_3_custom_config():
    """
    Customize GPU target and auto-free settings
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Configuration")
    print("="*60)
    
    from auto_gpu_optimizer import AutoGPUOptimizer
    
    # Create custom optimizer
    optimizer = AutoGPUOptimizer(
        model="llama3.2:3b",
        target_mb=2048,    # Target 2GB instead of 4GB
        auto_free=True     # Auto-free enabled
    )
    
    # Load model
    optimizer.load_model()
    
    # Use it
    optimizer.chat("Explain Python in one sentence", stream=False)


# ========================================================================
# EXAMPLE 4: Manual GPU Management (Advanced)
# ========================================================================

def example_4_manual_control():
    """
    Manual control over GPU freeing process
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Manual GPU Control")
    print("="*60)
    
    from auto_gpu_optimizer import AutoGPUOptimizer, free_gpu_memory, get_gpu_state
    
    # Check current GPU state
    used, on_gpu, total = get_gpu_state()
    print(f"\n📊 Current GPU: {used}MB / {total}MB")
    
    # Manually free 4GB
    print("\n🔧 Manually freeing GPU memory...")
    success, freed = free_gpu_memory(target_mb=4096)
    
    if success:
        print(f"✅ Freed {freed}MB")
        
        # Now create optimizer WITHOUT auto-free
        optimizer = AutoGPUOptimizer(
            model="llama3.2:3b",
            target_mb=4096,
            auto_free=False  # We already freed manually
        )
        
        optimizer.load_model()
        optimizer.chat("What is AI?", stream=True)
    else:
        print("❌ Could not free enough memory")


# ========================================================================
# EXAMPLE 5: Using Across Multiple Files
# ========================================================================

# Main file (loads model once)
"""
# main.py
from auto_gpu_optimizer import get_auto_optimized_model

# Load once with auto GPU freeing
llm = get_auto_optimized_model(
    model="llama3.2:3b",
    target_mb=4096,
    auto_free=True  # ✅ Frees GPU processes automatically
)

# Pass to modules
import data_processor
import text_analyzer

data_processor.process(llm)
text_analyzer.analyze(llm)
"""

# Module 1
"""
# data_processor.py
import asyncio

def process(llm_instance):
    '''Process data using the GPU-optimized model'''
    messages = [{"role": "user", "content": "Process this data..."}]
    
    loop = asyncio.new_event_loop()
    response = loop.run_until_complete(llm_instance.ainvoke(messages))
    loop.close()
    
    return response.content
"""

# Module 2
"""
# text_analyzer.py
import asyncio

def analyze(llm_instance):
    '''Analyze text using the same model'''
    messages = [{"role": "user", "content": "Analyze this text..."}]
    
    loop = asyncio.new_event_loop()
    response = loop.run_until_complete(llm_instance.ainvoke(messages))
    loop.close()
    
    return response.content
"""


# ========================================================================
# EXAMPLE 6: Different Memory Targets
# ========================================================================

def example_6_memory_targets():
    """
    Different VRAM targets for different models
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Different Memory Targets")
    print("="*60)
    
    from auto_gpu_optimizer import get_auto_optimized_handler
    
    # Small model (2GB)
    print("\n📝 Loading small model (2GB)...")
    handler_small = get_auto_optimized_handler(
        model="llama3.2:1b",
        target_mb=2048,
        auto_free=True
    )
    
    # Medium model (4GB)
    print("\n📝 Loading medium model (4GB)...")
    handler_medium = get_auto_optimized_handler(
        model="llama3.2:3b",
        target_mb=4096,
        auto_free=True,
        force_reload=True
    )
    
    print("\n✅ Both models loaded with optimal VRAM")


# ========================================================================
# EXAMPLE 7: Error Handling
# ========================================================================

def example_7_error_handling():
    """
    Proper error handling when GPU optimization fails
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling")
    print("="*60)
    
    from auto_gpu_optimizer import AutoGPUOptimizer
    
    try:
        optimizer = AutoGPUOptimizer(
            model="llama3.2:3b",
            target_mb=4096,
            auto_free=True
        )
        
        success = optimizer.load_model()
        
        if success:
            print("✅ Model loaded successfully")
            optimizer.chat("Hello!", stream=False)
        else:
            print("⚠️  Model loaded but may not be fully on GPU")
            print("💡 Options:")
            print("   1. Lower target_mb")
            print("   2. Use smaller model")
            print("   3. Manually close more applications")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Troubleshooting:")
        print("   1. Check if Ollama is installed")
        print("   2. Check if NVIDIA GPU is available")
        print("   3. Try running with auto_free=True")


# ========================================================================
# COMPARISON: With vs Without Auto-Free
# ========================================================================

def comparison_with_without_auto_free():
    """
    Compare behavior with and without auto-free
    """
    print("\n" + "="*60)
    print("COMPARISON: Auto-Free ON vs OFF")
    print("="*60)
    
    from auto_gpu_optimizer import get_gpu_state, AutoGPUOptimizer
    
    # Check initial state
    used, _, total = get_gpu_state()
    available = total - used
    
    print(f"\n📊 Initial State:")
    print(f"   Available: {available}MB")
    print(f"   Need: 4096MB")
    
    if available < 4096:
        print(f"\n⚠️  Not enough memory ({available}MB < 4096MB)")
        
        # WITHOUT AUTO-FREE
        print("\n❌ WITHOUT auto_free:")
        print("   Model loading would FAIL")
        print("   You would need to manually close apps")
        
        # WITH AUTO-FREE
        print("\n✅ WITH auto_free:")
        print("   Optimizer will:")
        print("   1. Detect shortage")
        print("   2. Find non-critical GPU processes")
        print("   3. Move them to CPU")
        print("   4. Free the GPU memory")
        print("   5. Load model successfully")
        
        # Demonstrate
        optimizer = AutoGPUOptimizer(
            model="llama3.2:3b",
            target_mb=4096,
            auto_free=True  # ✅ Magic happens here
        )
        
        success = optimizer.load_model()
        
        if success:
            print("\n✅ Model loaded thanks to auto_free!")
        else:
            print("\n⚠️  Still couldn't load (GPU may be too full)")
    else:
        print(f"\n✅ Already have enough memory!")
        print("   auto_free will not do anything (no need)")


# ========================================================================
# QUICK START GUIDE
# ========================================================================

def quick_start():
    """
    Quickest way to get started
    """
    print("\n" + "="*60)
    print("QUICK START - Copy & Paste This:")
    print("="*60)
    
    code = """
# ========== QUICK START ==========

from auto_gpu_optimizer import get_auto_optimized_handler

# Get handler (auto-frees GPU if needed)
handler = get_auto_optimized_handler(
    model="llama3.2:3b",
    target_mb=4096,
    auto_free=True  # ✅ Auto GPU management
)

# Use it!
handler.chat("What is AI?", stream=True)

# Or get model for other files
llm = handler.get_model()

# ========== END QUICK START ==========
"""
    
    print(code)


# ========================================================================
# RUN ALL EXAMPLES
# ========================================================================

if __name__ == "__main__":
    print("="*60)
    print("AUTO GPU OPTIMIZER - COMPLETE EXAMPLES")
    print("="*60)
    
    # Show quick start
    quick_start()
    
    # Show comparison
    comparison_with_without_auto_free()
    
    print("\n" + "="*60)
    print("Run individual examples:")
    print("="*60)
    print("1. example_1_auto_optimized()")
    print("2. example_2_get_model_only()")
    print("3. example_3_custom_config()")
    print("4. example_4_manual_control()")
    print("6. example_6_memory_targets()")
    print("7. example_7_error_handling()")
    print("="*60)
    
    # Uncomment to run an example:
    # example_1_auto_optimized()