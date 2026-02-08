import subprocess
import time
import os
import psutil
from langchain_ollama import ChatOllama

def run_cmd(cmd):
    """Run command and return output"""
    try:
        return subprocess.check_output(cmd, shell=True).decode().strip()
    except:
        return ""

def get_gpu_state():
    """Returns (Used_VRAM_MB, Is_Ollama_On_GPU_Flag, Total_VRAM_MB)"""
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
        'fontdrvhost.exe', 'conhost.exe',
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
        print(f"Error getting GPU processes: {e}")
        return []

def move_process_to_cpu(pid, proc_name):
    """Move a process from GPU to CPU by setting GPU affinity"""
    try:
        try:
            p = psutil.Process(pid)
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print(f"      ✅ Lowered priority: {proc_name} (PID {pid})")
            return True
        except:
            pass
        
        return True
        
    except Exception as e:
        print(f"      ⚠️  Could not modify {proc_name}: {e}")
        return False

def force_apps_to_cpu():
    """Force non-critical GPU processes to prefer CPU"""
    print("\n🔄 Moving GPU processes to CPU...")
    
    gpu_procs = get_gpu_processes()
    
    if not gpu_procs:
        print("   ℹ️  No GPU processes found")
        return 0
    
    critical = [p for p in gpu_procs if p['critical']]
    non_critical = [p for p in gpu_procs if not p['critical']]
    
    print(f"\n   Found {len(gpu_procs)} GPU processes:")
    print(f"   • Critical (keeping): {len(critical)}")
    print(f"   • Non-critical (moving to CPU): {len(non_critical)}")
    
    if critical:
        print(f"\n   🔒 Protected processes:")
        for proc in critical:
            print(f"      • {proc['name']} ({proc['mem_mb']}MB)")
    
    if not non_critical:
        print("\n   ℹ️  All processes are critical - cannot free memory safely")
        return 0
    
    print(f"\n   🎯 Moving to CPU:")
    
    moved_count = 0
    estimated_freed = 0
    
    for proc in non_critical:
        if 'ollama' in proc['name'].lower():
            print(f"      🔄 Restarting Ollama in CPU mode: {proc['name']}")
            try:
                subprocess.run("taskkill /F /IM ollama.exe", shell=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run("taskkill /F /IM ollama_llama_server.exe", shell=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
                
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = '-1'
                subprocess.Popen(
                    ["ollama", "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                print(f"      ✅ Ollama restarted in CPU mode")
                estimated_freed += proc['mem_mb']
                moved_count += 1
            except Exception as e:
                print(f"      ⚠️  Could not restart Ollama: {e}")
        else:
            if move_process_to_cpu(proc['pid'], proc['name']):
                moved_count += 1
                estimated_freed += proc['mem_mb'] // 2
    
    print(f"\n   📊 Summary:")
    print(f"      • Processes modified: {moved_count}")
    print(f"      • Estimated freed: ~{estimated_freed}MB")
    
    return estimated_freed

def restart_chrome_cpu_mode():
    """Restart Chrome browsers in CPU-only mode"""
    print("\n   🌐 Handling Chrome browsers...")
    
    try:
        chrome_procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'chrome.exe' in proc.info['name'].lower():
                    chrome_procs.append(proc)
            except:
                pass
        
        if not chrome_procs:
            print("      ℹ️  No Chrome processes found")
            return
        
        print(f"      Found {len(chrome_procs)} Chrome processes")
        choice = input("      Restart Chrome in CPU mode? (y/n): ")
        
        if choice.lower() != 'y':
            print("      ⏭️  Skipped Chrome restart")
            return
        
        for proc in chrome_procs:
            try:
                proc.terminate()
            except:
                pass
        
        time.sleep(2)
        
        chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        if os.path.exists(chrome_path):
            subprocess.Popen(
                [chrome_path, "--disable-gpu", "--disable-software-rasterizer"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("      ✅ Chrome restarted in CPU mode")
        else:
            print("      ℹ️  Chrome not found at default location")
            
    except Exception as e:
        print(f"      ⚠️  Error handling Chrome: {e}")

def aggressive_gpu_cleanup():
    """Comprehensive GPU memory cleanup without killing critical processes"""
    print("\n🧹 AGGRESSIVE GPU CLEANUP (Safe Mode)")
    print("="*60)
    
    base_vram, _, total_vram = get_gpu_state()
    print(f"Initial GPU usage: {base_vram}MB / {total_vram}MB")
    
    estimated_freed = force_apps_to_cpu()
    
    print("\n   ⏳ Waiting for GPU to release memory...")
    time.sleep(3)
    
    new_vram, _, _ = get_gpu_state()
    actual_freed = base_vram - new_vram
    
    print(f"\n   📊 Results:")
    print(f"      • Before: {base_vram}MB used")
    print(f"      • After: {new_vram}MB used")
    print(f"      • Freed: {actual_freed}MB")
    
    if actual_freed < estimated_freed:
        restart_chrome_cpu_mode()
        time.sleep(2)
        final_vram, _, _ = get_gpu_state()
        total_freed = base_vram - final_vram
        print(f"\n   📊 Final: {total_freed}MB freed total")
    
    return new_vram, total_vram

def prepare_gpu_memory(required_vram_mb=4000):
    """
    Prepare GPU memory for model loading.
    
    Args:
        required_vram_mb: Amount of VRAM needed in MB (default: 4000 for 4GB)
    
    Returns:
        tuple: (success: bool, available_mb: int)
    """
    print("\n🎯 PREPARING GPU MEMORY")
    print("="*60)
    
    base_vram, _, total_vram = get_gpu_state()
    available = total_vram - base_vram
    
    print(f"GPU Total: {total_vram}MB ({total_vram/1024:.1f}GB)")
    print(f"Currently Used: {base_vram}MB ({base_vram/1024:.1f}GB)")
    print(f"Available: {available}MB ({available/1024:.1f}GB)")
    print(f"Required: {required_vram_mb}MB ({required_vram_mb/1024:.1f}GB)")
    
    if total_vram < required_vram_mb:
        print(f"\n❌ GPU too small ({total_vram}MB total)")
        print("\n💡 OPTIONS:")
        print("   1. Use smaller model like 'llama3.2:1b' (~1.3GB)")
        print("   2. Run on CPU with num_gpu=0")
        return False, 0
    
    if available >= required_vram_mb:
        print(f"\n✅ Already have {available}MB available!")
        return True, available
    
    shortage = required_vram_mb - available
    print(f"\n⚠️  Need to free {shortage}MB ({shortage/1024:.1f}GB)")
    
    current_vram, total_vram = aggressive_gpu_cleanup()
    available = total_vram - current_vram
    
    if available >= required_vram_mb:
        print(f"\n✅ Success! {available}MB now available")
        return True, available
    else:
        print(f"\n⚠️  Only freed to {available}MB (need {required_vram_mb}MB)")
        shortage = required_vram_mb - available
        print(f"   Still short: {shortage}MB")
        return False, available

def initialize_llm(
    model="llama3.2:3b",
    target_vram_mb=4096,
    model_base_size_mb=2200,
    auto_prepare_gpu=True,
    temperature=0
):
    """
    Initialize ChatOllama LLM with smart GPU allocation.
    
    Args:
        model: Model name (default: "llama3.2:3b")
        target_vram_mb: Target VRAM allocation in MB (default: 4096 for 4GB)
        model_base_size_mb: Base model size in MB (default: 2200 for llama3.2:3b)
        auto_prepare_gpu: Automatically try to free GPU memory if needed
        temperature: Model temperature
    
    Returns:
        ChatOllama: Configured LLM instance, or None if failed
    """
    print("\n" + "="*60)
    print("🚀 INITIALIZING LLM")
    print("="*60)
    
    # Prepare GPU if requested
    if auto_prepare_gpu:
        success, available = prepare_gpu_memory(target_vram_mb)
        if not success:
            print("\n❌ Insufficient GPU memory")
            return None
        
        # Adjust target if needed
        actual_allocation = min(target_vram_mb, available - 300)  # 300MB safety buffer
        if actual_allocation < target_vram_mb:
            print(f"\n⚠️  Using {actual_allocation}MB instead of {target_vram_mb}MB")
            target_vram_mb = actual_allocation
    
    # Get baseline GPU state
    base_vram, _, total_vram = get_gpu_state()
    
    # Calculate context size
    CUDA_OVERHEAD_MB = 200
    ctx_budget_mb = target_vram_mb - model_base_size_mb - CUDA_OVERHEAD_MB
    num_ctx = int(ctx_budget_mb / 0.09)  # ~0.09 MB per token
    num_ctx = max(512, min(num_ctx, 32000))
    
    print(f"\n📊 Configuration:")
    print(f"   Model: {model}")
    print(f"   Base size: ~{model_base_size_mb}MB")
    print(f"   Context: {num_ctx} tokens (~{ctx_budget_mb}MB)")
    print(f"   Target VRAM: {target_vram_mb}MB")
    
    # Ensure GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Stop any CPU-mode Ollama
    print("\n🔄 Restarting Ollama in GPU mode...")
    try:
        subprocess.run("taskkill /F /IM ollama_llama_server.exe", shell=True,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
    except:
        pass
    
    # Create LLM instance
    print("\n⚙️  Creating LLM instance...")
    try:
        llm = ChatOllama(
            model=model,
            num_gpu=100,
            num_ctx=num_ctx,
            num_batch=512,
            temperature=temperature,
        )
        
        # Test the model
        print("🔄 Testing model...")
        response = llm.invoke("Hi")
        print(f"   ✅ Response: {response.content[:50]}...")
        
        # Verify GPU usage
        time.sleep(2)
        final_vram, is_on_gpu, _ = get_gpu_state()
        vram_used = final_vram - base_vram
        
        print(f"\n📊 Verification:")
        print(f"   Baseline: {base_vram}MB")
        print(f"   Current: {final_vram}MB")
        print(f"   Allocated: {vram_used}MB")
        print(f"   On GPU: {'YES ✅' if is_on_gpu else 'NO ❌'}")
        
        if is_on_gpu and vram_used >= model_base_size_mb * 0.8:
            print("\n✅ LLM initialized successfully on GPU!")
            return llm
        else:
            print("\n⚠️  LLM may not be using GPU optimally")
            return llm
            
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM: {e}")
        return None


def demo_usage():
    """Demo showing how to use the refactored code"""
    # Option 1: Let the function handle everything automatically
    llm = initialize_llm(
        model="llama3.2:3b",
        target_vram_mb=4096,
        auto_prepare_gpu=True
    )
    
    if llm:
        # Now you can use the LLM
        print("\n" + "="*60)
        print("💬 Testing LLM")
        print("="*60)
        
        response = llm.invoke("What is the capital of France?")
        print(f"\nQ: What is the capital of France?")
        print(f"A: {response.content}")
        
        response = llm.invoke("Write a haiku about coding")
        print(f"\nQ: Write a haiku about coding")
        print(f"A: {response.content}")
    else:
        print("\n❌ Could not initialize LLM")


if __name__ == "__main__":
    import ctypes
    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    print("="*60)
    if is_admin:
        print("✅ Running as Administrator")
    else:
        print("⚠️  Not Administrator (some features limited)")
    print("="*60)
    
    # Check dependencies
    try:
        import psutil
    except ImportError:
        print("\n⚠️  Missing dependency!")
        print("   Run: pip install psutil")
        exit(1)
    
    # Run demo
    demo_usage()