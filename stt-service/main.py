"""
STT Microservice — GPU-accelerated Whisper ASR
==============================================
Provides real-time streaming transcription via Whisper.
"""

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from asr import StreamingSpeechRecognizer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stt_service")

# ── Global State ─────────────────────────────────────────────────────────────
recognizer: StreamingSpeechRecognizer = None


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    logger.info("🚀 Initializing ASR model...")

    # Load model settings from env
    model_size = os.getenv("WHISPER_MODEL", "base.en")
    device = os.getenv("DEVICE", "cpu")

    try:
        recognizer = StreamingSpeechRecognizer(
            model_size=model_size,
            device=device,
        )
        logger.info("✅ STT Service ready (model=%s, device=%s)", model_size, device)
    except Exception as e:
        logger.error("❌ Failed to initialize ASR: %s", e)
        # We don't raise here so the health check can report the failure

    yield
    logger.info("🛑 Shutting down STT Service...")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="STT Microservice",
    description="Zero-latency streaming ASR with Whisper",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64 encoded float32 PCM audio")
    sample_rate: int = Field(16000)
    ai_is_speaking: bool = Field(False)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if recognizer is None:
        return {"status": "loading", "ready": False}
    return {
        "status": "ok",
        "ready": True,
        "backend": "whisper",
        "model_size": recognizer._model_size,
        "device": str(recognizer.device),
    }


@app.post("/transcribe/stream")
async def transcribe_stream(req: TranscribeRequest):
    """
    SSE stream of transcribed words.

    Expected payload:
      - audio_b64: base64 string of raw float32 PCM
      - sample_rate: 16000
      - ai_is_speaking: boolean (for echo cancellation / barge-in)
    """
    if recognizer is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Decode audio
            try:
                audio_bytes = base64.b64decode(req.audio_b64)
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            except Exception as e:
                logger.error("Audio decoding error: %s", e)
                yield _sse({"type": "error", "message": "Invalid audio data"})
                return

            if len(audio_data) == 0:
                yield _sse({"type": "done"})
                return

            # Perform transcription (Whisper is CPU/GPU heavy, run in executor)
            loop = asyncio.get_event_loop()

            # Simple wrapper to run generator in thread
            def _run_asr():
                # Note: ai_is_speaking is passed to handle echo cancellation or
                # skipping processing if needed, though Whisper is mostly for
                # user speech.
                return recognizer.transcribe_streaming(audio_data, req.sample_rate)

            # We use run_in_executor for the blocking generator call
            # Whisper segments are yielded one-by-one
            gen = await loop.run_in_executor(None, _run_asr)

            word_count = 0
            while True:
                # Advance generator in executor to avoid blocking event loop
                word = await loop.run_in_executor(None, next, gen, None)
                if word is None:
                    break
                word_count += 1
                yield _sse({"type": "word", "word": word})

            yield _sse({"type": "done", "word_count": word_count})

        except Exception as e:
            logger.error("Transcription error: %s", e)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
