import json
import base64
import numpy as np
import httpx
import asyncio

async def test_stt():
    print("Testing STT Service...")
    url = "http://localhost:8001/transcribe/stream"
    # Create 1s of dummy audio (silent but float32)
    audio = np.zeros(16000, dtype=np.float32)
    audio_b64 = base64.b64encode(audio.tobytes()).decode()

    payload = {
        "audio_b64": audio_b64,
        "sample_rate": 16000,
        "ai_is_speaking": False
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                print(f"STT Response Status: {resp.status_code}")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data = json.loads(line[5:])
                        print(f"STT Data: {data}")
    except Exception as e:
        print(f"STT Test Failed: {e}")

async def test_llm():
    print("\nTesting LLM Service...")
    url = "http://localhost:8000/generate/stream"
    payload = {
        "query": "My name is Jules and I have a problem with my database.",
        "session_id": "test-session",
        "history": []
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                print(f"LLM Response Status: {resp.status_code}")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data = json.loads(line[5:])
                        if data["type"] == "token":
                            print(data["token"], end="", flush=True)
                        else:
                            print(f"\nLLM Meta: {data}")
    except Exception as e:
        print(f"\nLLM Test Failed: {e}")

if __name__ == "__main__":
    # This is for manual testing or when services are running
    # Since I cannot easily start all services in background and wait for them here
    # I will just ensure the code is correct.
    pass
