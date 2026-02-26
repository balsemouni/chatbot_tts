import base64
import json
import logging
import os
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

from asr import StreamingSpeechRecognizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-service")

# -- Global State --
asr_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_engine
    logger.info("Initializing ASR engine...")
    asr_engine = StreamingSpeechRecognizer(
        model_size=os.getenv("WHISPER_MODEL", "base.en"),
        device="cpu", # Defaulting to CPU for safety in sandbox, though asr.py tries to use CUDA
        enable_ai_filtering=os.getenv("ENABLE_AI_FILTERING", "true").lower() == "true"
    )
    logger.info("ASR engine ready.")
    yield
    logger.info("Shutting down STT service...")

app = FastAPI(title="STT Microservice", lifespan=lifespan)

class TranscribeRequest(BaseModel):
    audio_b64: str
    sample_rate: int = 16000
    ai_is_speaking: bool = False

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

@app.post("/transcribe/stream")
async def transcribe_stream(req: TranscribeRequest):
    if asr_engine is None:
        raise HTTPException(status_code=503, detail="ASR engine not initialized")

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(req.audio_b64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to decode audio: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio data")

    async def event_generator():
        try:
            # We wrap the synchronous generator from asr_engine
            for word in asr_engine.transcribe_streaming(audio_data, sample_rate=req.sample_rate):
                yield _sse({"type": "word", "word": word})

            yield _sse({"type": "done"})
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {
        "status": "ok" if asr_engine is not None else "initializing",
        "backend": "faster-whisper",
        "device": getattr(asr_engine, "device", "unknown") if asr_engine else "unknown"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
