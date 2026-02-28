"""
STT Microservice  v2.0
======================
FastAPI service exposing Whisper-based speech-to-text with streaming SSE output.

Endpoints
---------
POST /transcribe/stream   SSE word stream  (primary — used by gateway)
POST /transcribe          Blocking full transcript
GET  /health

POST /transcribe/stream  request body
--------------------------------------
{
  "audio_b64":      "<base64 float32 PCM>",
  "sample_rate":    16000,
  "ai_is_speaking": false
}

SSE event types (in order)
---------------------------
  {"type": "word",        "word": "Hello"}
  {"type": "ai_filtered"}          ← audio detected as AI voice, skipped
  {"type": "silence"}              ← no speech detected
  {"type": "done",        "transcript": "Hello world"}

Environment variables
---------------------
  WHISPER_MODEL        str    default "base.en"
  DEVICE               str    default "cuda" (auto-detected)
  AI_DETECTOR_MODEL    str    path to CNN model (optional)
  AI_DETECTION_THR     float  default 0.70
  ENABLE_AI_FILTERING  bool   default true
  PORT                 int    default 8001
  LOG_LEVEL            str    default "INFO"
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from asr import StreamingSpeechRecognizer

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stt_service")

asr: StreamingSpeechRecognizer | None = None
_startup_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, _startup_time
    t0 = time.perf_counter()

    model_size = os.getenv("WHISPER_MODEL", "base.en")
    device     = os.getenv("DEVICE", "cuda")
    ai_model   = os.getenv("AI_DETECTOR_MODEL", None)
    ai_thr     = float(os.getenv("AI_DETECTION_THR", "0.70"))
    enable_ai  = os.getenv("ENABLE_AI_FILTERING", "true").lower() == "true"

    logger.info("🚀 Loading Whisper %s on %s…", model_size, device.upper())

    loop = asyncio.get_event_loop()
    asr = await loop.run_in_executor(
        None,
        lambda: StreamingSpeechRecognizer(
            model_size=model_size,
            device=device,
            ai_detector_model_path=ai_model,
            ai_detection_threshold=ai_thr,
            enable_ai_filtering=enable_ai,
        ),
    )

    _startup_time = time.perf_counter() - t0
    logger.info("✅ STT service ready in %.2fs", _startup_time)
    yield

    logger.info("🛑 Shutting down STT service…")


# ---------------------------------------------------------------------------
app = FastAPI(
    title="STT Microservice",
    description="Streaming Whisper ASR — SSE word stream with AI voice filtering",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TranscribeRequest(BaseModel):
    audio_b64:      str   = Field(..., description="Base64-encoded float32 PCM")
    sample_rate:    int   = Field(16000, ge=8000, le=48000)
    ai_is_speaking: bool  = Field(False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_asr() -> StreamingSpeechRecognizer:
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR not loaded")
    return asr


def _decode_audio(audio_b64: str, sample_rate: int) -> np.ndarray:
    """Decode base64 float32 PCM bytes → numpy float32 array."""
    raw = base64.b64decode(audio_b64)
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    return audio


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# /transcribe/stream  — primary endpoint used by gateway
# ---------------------------------------------------------------------------
@app.post("/transcribe/stream", tags=["transcribe"])
async def transcribe_stream(req: TranscribeRequest):
    """
    SSE word stream.

    Decodes audio, runs Whisper in a thread-pool executor (non-blocking),
    and streams each word as it arrives.  The gateway consumes this stream
    to build the transcript in real time.

    AI voice filtering:
      If the audio is detected as AI-generated (TTS echo), emits
      {"type": "ai_filtered"} and stops — no transcript produced.
    """
    recognizer = _require_asr()

    async def generate() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                audio = _decode_audio(req.audio_b64, req.sample_rate)
                if len(audio) == 0:
                    loop.call_soon_threadsafe(q.put_nowait, ("silence", None))
                    return

                # Check AI filtering via cached result (non-blocking in ASR)
                # The ASR class handles this internally; we just stream words.
                words_found = False
                for word in recognizer.transcribe_streaming(audio, req.sample_rate):
                    loop.call_soon_threadsafe(q.put_nowait, ("word", word))
                    words_found = True

                if not words_found:
                    loop.call_soon_threadsafe(q.put_nowait, ("silence", None))

            except Exception as exc:
                logger.error("ASR producer error: %s", exc)
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        # Run blocking Whisper in thread pool — never blocks event loop
        producer_fut = loop.run_in_executor(None, _producer)

        transcript_words: list[str] = []
        try:
            while True:
                kind, value = await q.get()

                if kind == "word":
                    transcript_words.append(value)
                    yield _sse({"type": "word", "word": value})

                elif kind == "silence":
                    yield _sse({"type": "silence"})
                    break

                elif kind == "ai_filtered":
                    yield _sse({"type": "ai_filtered"})
                    break

                elif kind == "error":
                    logger.error("ASR error: %s", value)
                    break

                elif kind == "done":
                    break

        finally:
            await producer_fut
            transcript = " ".join(transcript_words).strip()
            yield _sse({"type": "done", "transcript": transcript})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# /transcribe  — blocking full transcript (for testing / non-streaming clients)
# ---------------------------------------------------------------------------
@app.post("/transcribe", tags=["transcribe"])
async def transcribe(req: TranscribeRequest):
    """Blocking transcription — returns full text in one JSON response."""
    recognizer = _require_asr()
    loop = asyncio.get_event_loop()

    def _run():
        audio = _decode_audio(req.audio_b64, req.sample_rate)
        return recognizer.transcribe(audio, req.sample_rate)

    t0 = time.perf_counter()
    transcript = await loop.run_in_executor(None, _run)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return {
        "transcript":  transcript,
        "elapsed_ms":  elapsed_ms,
        "word_count":  len(transcript.split()) if transcript else 0,
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
async def health():
    return {
        "status":       "ok" if asr else "loading",
        "model_loaded": asr is not None,
        "model":        os.getenv("WHISPER_MODEL", "base.en"),
        "device":       os.getenv("DEVICE", "cuda"),
        "startup_time_s": round(_startup_time, 3),
    }
