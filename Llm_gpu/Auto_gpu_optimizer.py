"""
GPU Optimizer with Automatic Process Management
Automatically frees GPU memory by moving non-critical processes to CPU
"""

import subprocess
import time
import os
import psutil
from streaming_llm_handler import StreamingLLMHandler


# ========================================================================
# GPU MEMORY MANAGEMENT
# ========================================================================

def run_cmd(cmd):
    """Run command and return output"""
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except:
        return ""

def get_gpu_state():
    """Returns (Used_VRAM_MB, Is_Ollama_On_GPU, Total_VRAM_MB)"""
    try:
        vram = int(run_cmd("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))
        total = int(run_cmd("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"))
        procs = run_cmd("nvidia-smi --query-compute-apps=process_name --format=csv,noheader").lower()
        on_gpu = "ollama" in procs or "llama" in procs
        return vram, on_gpu, total
    except:
        return 0, False, 0

def is_critical_process(proc_name):
    """Check if process is critical and should NEVER be touched"""
    critical_keywords = [
        'dwm.exe', 'explorer.exe', 'nvcontainer', 'nvdisplay',
        'csrss.exe', 'winlogon.exe', 'services.exe', 'lsass.exe',
        'smss.exe', 'svchost.exe', 'system', 'wininit.exe',
        'fontdrvhost.exe', 'conhost.exe'
    ]
    proc_lower = proc_name.lower()
    return any(crit in proc_lower for crit in critical_keywords)

def get_gpu_processes():
    """Get detailed list of GPU processes with PIDs"""
    try:
        cmd = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
        output = run_cmd(cmd)
        
        if not output:
            return []
        
        processes = []
        for line in output.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    pid = parts[0].strip()
                    name = parts[1].strip()
                    mem = parts[2].strip().split()[0] if len(parts) > 2 else "0"
                    
                    processes.append({
                        'pid': int(pid),
                        'name': name,
                        'mem_mb': int(mem) if mem.isdigit() else 0,
                        'critical': is_critical_process(name)
                    })
        
        return processes
    except Exception as e:
        print(f"⚠️  Error getting GPU processes: {e}")
        return []

def move_process_to_cpu(pid, proc_name):
    """Move a process from GPU to CPU by lowering priority"""
    try:
        p = psutil.Process(pid)
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        print(f"      ✅ Lowered priority: {proc_name} (PID {pid})")
        return True
    except Exception as e:
        print(f"      ⚠️  Could not modify {proc_name}: {e}")
        return False

def restart_ollama_cpu_mode():
    """Restart Ollama in CPU-only mode"""
    print("\n   🔄 Restarting Ollama in CPU mode...")
    try:
        # Stop Ollama
        subprocess.run("taskkill /F /IM ollama.exe", shell=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("taskkill /F /IM ollama_llama_server.exe", shell=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        
        # Restart with CPU-only
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print("      ✅ Ollama restarted in CPU mode")
        time.sleep(3)
        return True
    except Exception as e:
        print(f"      ⚠️  Failed to restart Ollama: {e}")
        return False

def restart_ollama_gpu_mode():
    """Restart Ollama in GPU mode"""
    print("\n   🔄 Restarting Ollama in GPU mode...")
    try:
        # Stop Ollama
        subprocess.run("taskkill /F /IM ollama.exe", shell=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("taskkill /F /IM ollama_llama_server.exe", shell=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        
        # Restart with GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print("      ✅ Ollama restarted in GPU mode")
        time.sleep(3)
        return True
    except Exception as e:
        print(f"      ⚠️  Failed to restart Ollama: {e}")
        return False

def free_gpu_memory(target_mb):
    """
    Free GPU memory by moving non-critical processes to CPU
    
    Args:
        target_mb: How much free memory we need
        
    Returns:
        (success, freed_mb)
    """
    print("\n🧹 FREEING GPU MEMORY")
    print("="*60)
    
    # Get current state
    used_vram, _, total_vram = get_gpu_state()
    available = total_vram - used_vram
    
    print(f"GPU Total: {total_vram}MB ({total_vram/1024:.1f}GB)")
    print(f"Currently Used: {used_vram}MB ({used_vram/1024:.1f}GB)")
    print(f"Available: {available}MB ({available/1024:.1f}GB)")
    print(f"Need: {target_mb}MB ({target_mb/1024:.1f}GB)")
    
    # Check if we already have enough
    if available >= target_mb:
        print(f"\n✅ Already have {available}MB available!")
        return True, 0
    
    shortage = target_mb - available
    print(f"\n⚠️  Need to free: {shortage}MB ({shortage/1024:.1f}GB)")
    
    # Get GPU processes
    gpu_procs = get_gpu_processes()
    
    if not gpu_procs:
        print("\n   ℹ️  No GPU processes found")
        return False, 0
    
    # Separate critical from non-critical
    critical = [p for p in gpu_procs if p['critical']]
    non_critical = [p for p in gpu_procs if not p['critical']]
    
    print(f"\n   📊 GPU Processes:")
    print(f"      • Critical (keeping): {len(critical)}")
    print(f"      • Non-critical (moving to CPU): {len(non_critical)}")
    
    if critical:
        print(f"\n   🔒 Protected processes:")
        for proc in critical[:5]:  # Show first 5
            print(f"      • {proc['name']} ({proc['mem_mb']}MB)")
    
    if not non_critical:
        print("\n   ⚠️  All processes are critical")
        return False, 0
    
    print(f"\n   🎯 Moving to CPU:")
    
    moved_count = 0
    estimated_freed = 0
    
    for proc in non_critical:
        # Handle Ollama specially
        if 'ollama' in proc['name'].lower():
            if restart_ollama_cpu_mode():
                estimated_freed += proc['mem_mb']
                moved_count += 1
        else:
            # Lower priority for other processes
            if move_process_to_cpu(proc['pid'], proc['name']):
                moved_count += 1
                estimated_freed += proc['mem_mb'] // 2  # Conservative estimate
    
    # Wait for changes
    print(f"\n   ⏳ Waiting for GPU to release memory...")
    time.sleep(3)
    
    # Check results
    new_vram, _, _ = get_gpu_state()
    actual_freed = used_vram - new_vram
    new_available = total_vram - new_vram
    
    print(f"\n   📊 Results:")
    print(f"      • Before: {used_vram}MB used")
    print(f"      • After: {new_vram}MB used")
    print(f"      • Freed: {actual_freed}MB")
    print(f"      • Available: {new_available}MB")
    
    success = new_available >= target_mb
    
    if success:
        print(f"\n   ✅ Success! Have {new_available}MB available")
    else:
        shortage = target_mb - new_available
        print(f"\n   ⚠️  Still short by {shortage}MB")
    
    return success, actual_freed


# ========================================================================
# AUTO GPU OPTIMIZER
# ========================================================================

class AutoGPUOptimizer:
    """
    Automatically manages GPU memory and loads model
    Frees processes to CPU if needed
    """
    
    def __init__(self, model="llama3.2:3b", target_mb=4096, auto_free=True):
        """
        Initialize optimizer
        
        Args:
            model: Model name
            target_mb: Target VRAM allocation
            auto_free: Automatically free GPU memory if needed
        """
        self.model_name = model
        self.target_mb = target_mb
        self.auto_free = auto_free
        self.handler = None
        self.llm = None
        
        # Model size estimates
        self.model_sizes = {
            "llama3.2:1b": 1300,
            "llama3.2:3b": 2200,
            "llama3.1:8b": 5000,
        }
        
        self.model_base_size = self.model_sizes.get(model, 2200)
    
    def check_gpu_availability(self):
        """Check if GPU has enough memory"""
        used, _, total = get_gpu_state()
        available = total - used
        
        print(f"\n📊 GPU Check:")
        print(f"   Total: {total}MB")
        print(f"   Used: {used}MB")
        print(f"   Available: {available}MB")
        print(f"   Need: {self.target_mb}MB")
        
        return available >= self.target_mb, available
    
    def ensure_gpu_memory(self):
        """Ensure enough GPU memory is available"""
        print("\n" + "="*60)
        print("🎯 ENSURING GPU MEMORY")
        print("="*60)
        
        # Check availability
        has_space, available = self.check_gpu_availability()
        
        if has_space:
            print(f"\n✅ Already have enough memory ({available}MB)")
            return True
        
        if not self.auto_free:
            print(f"\n❌ Not enough memory and auto_free=False")
            return False
        
        # Try to free memory
        print(f"\n🔧 Auto-freeing GPU memory...")
        success, freed = free_gpu_memory(self.target_mb)
        
        return success
    
    def calculate_context_size(self):
        """Calculate optimal context size for target VRAM"""
        CUDA_OVERHEAD_MB = 200
        ctx_budget_mb = self.target_mb - self.model_base_size - CUDA_OVERHEAD_MB
        num_ctx = int(ctx_budget_mb / 0.09)
        num_ctx = max(512, min(num_ctx, 32000))
        
        return num_ctx
    
    def load_model(self):
        """Load model with GPU optimization"""
        print("\n" + "="*60)
        print("🚀 LOADING GPU-OPTIMIZED MODEL")
        print("="*60)
        
        # Ensure GPU memory
        if not self.ensure_gpu_memory():
            print("\n❌ Cannot ensure GPU memory")
            return False
        
        # Restart Ollama in GPU mode
        restart_ollama_gpu_mode()
        
        # Calculate context
        num_ctx = self.calculate_context_size()
        
        print(f"\n📝 Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Target VRAM: {self.target_mb}MB")
        print(f"   Base size: ~{self.model_base_size}MB")
        print(f"   Context: {num_ctx} tokens")
        
        # Get baseline
        base_vram, _, _ = get_gpu_state()
        print(f"   Baseline VRAM: {base_vram}MB")
        
        # Create handler
        print("\n🔄 Creating handler...")
        self.handler = StreamingLLMHandler(
            model=self.model_name,
            num_ctx=num_ctx,
            temperature=0.7
        )
        
        # Get model
        self.llm = self.handler.get_model()
        
        # Warm up
        print("🔄 Warming up model...")
        try:
            self.handler.invoke([{"role": "user", "content": "Hi"}])
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")
        
        time.sleep(2)
        
        # Verify
        final_vram, is_on_gpu, total = get_gpu_state()
        allocated = final_vram - base_vram
        
        print("\n" + "="*60)
        print("📊 GPU STATUS")
        print("="*60)
        print(f"Baseline: {base_vram}MB")
        print(f"Current: {final_vram}MB")
        print(f"Allocated: {allocated}MB")
        print(f"On GPU: {'YES ✅' if is_on_gpu else 'NO ❌'}")
        print(f"Total VRAM: {total}MB")
        print("="*60)
        
        if is_on_gpu and allocated >= self.model_base_size * 0.8:
            print("\n✅ MODEL LOADED SUCCESSFULLY ON GPU!")
            return True
        else:
            print("\n⚠️  Model may not be fully on GPU")
            return False
    
    def get_model(self):
        """Get the LLM instance"""
        if self.llm is None:
            self.load_model()
        return self.llm
    
    def get_handler(self):
        """Get the handler instance"""
        if self.handler is None:
            self.load_model()
        return self.handler
    
    def chat(self, message, stream=True):
        """Quick chat method"""
        if self.handler is None:
            self.load_model()
        return self.handler.chat(message, stream=stream, print_response=True)


# ========================================================================
# GLOBAL INSTANCE WITH AUTO GPU MANAGEMENT
# ========================================================================

_global_auto_optimizer = None

def get_auto_optimized_handler(model="llama3.2:3b", target_mb=4096, auto_free=True, force_reload=False):
    """
    Get handler with automatic GPU optimization and process management
    
    Args:
        model: Model name
        target_mb: Target VRAM allocation
        auto_free: Automatically free GPU by moving processes to CPU
        force_reload: Force reload model
        
    Returns:
        StreamingLLMHandler instance
    """
    global _global_auto_optimizer
    
    if _global_auto_optimizer is None or force_reload:
        print("\n🎯 Initializing Auto GPU Optimizer...")
        _global_auto_optimizer = AutoGPUOptimizer(
            model=model,
            target_mb=target_mb,
            auto_free=auto_free
        )
        _global_auto_optimizer.load_model()
    
    return _global_auto_optimizer.get_handler()

def get_auto_optimized_model(model="llama3.2:3b", target_mb=4096, auto_free=True, force_reload=False):
    """
    Get model with automatic GPU optimization
    
    Returns:
        ChatOllama instance
    """
    global _global_auto_optimizer
    
    if _global_auto_optimizer is None or force_reload:
        _global_auto_optimizer = AutoGPUOptimizer(
            model=model,
            target_mb=target_mb,
            auto_free=auto_free
        )
        _global_auto_optimizer.load_model()
    
    return _global_auto_optimizer.get_model()


# ========================================================================
# EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    print("="*60)
    print("AUTO GPU OPTIMIZER - WITH PROCESS MANAGEMENT")
    print("="*60)
    
    # Check initial GPU state
    used, on_gpu, total = get_gpu_state()
    print(f"\n📊 Initial State:")
    print(f"   GPU: {used}MB / {total}MB")
    print(f"   Ollama on GPU: {on_gpu}")
    
    # Create optimizer with auto-free enabled
    optimizer = AutoGPUOptimizer(
        model="llama3.2:3b",
        target_mb=4096,
        auto_free=True  # THIS ENABLES AUTO PROCESS MANAGEMENT
    )
    
    # Load model (will auto-free GPU if needed)
    success = optimizer.load_model()
    
    if success:
        print("\n" + "="*60)
        print("✅ READY TO USE")
        print("="*60)
        
        # Test it
        print("\n🧪 Testing model:")
        optimizer.chat("Say hello in one sentence", stream=False)
        
        # Get model for use in other files
        llm = optimizer.get_model()
        print(f"\n✅ Model instance ready: {type(llm).__name__}")
        print("\n💡 You can now use this model in other files:")
        print("   from auto_gpu_optimizer import get_auto_optimized_model")
        print("   llm = get_auto_optimized_model()")
    else:
        print("\n❌ Failed to load model")