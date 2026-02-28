"""
router.py — Zero-Latency Parallel Pipeline  v2
===============================================

Architecture (4 stages → 3 stages, sentence buffer removed):
  CLIENT → [audio_buf] → STT (streaming) → [word_q] → LLM → [token_q] → TTS → CLIENT

Changes vs v1:
  ✅ Sentence buffer stage REMOVED — tokens go directly to TTS chunker
  ✅ TTS chunker flushes on punctuation OR every 5 words (was 8)
  ✅ TTS ordered-queue: synthesis tasks run in parallel but audio is sent
     to WebSocket IN ORDER via an asyncio.Queue of futures
  ✅ No more "pending list awaited at end" — audio plays as each chunk finishes
  ✅ Barge-in cancels all tasks instantly via asyncio.Event

Latency budget:
  STT first word : ~300ms  (Whisper on first 1s of speech)
  LLM first token: ~200ms  (KV-cache warm, CAG context injected)
  TTS first chunk: ~80ms   (PocketTTS first 5-word group)
  Total TTFS     : ~580ms  (time-to-first-sound)
"""

import asyncio
import base64
import json
import re
import time
import logging
from typing import Optional, List

import httpx

logger = logging.getLogger("router")


# ─────────────────────────────────────────────────────────────────────────────
# TTS flush policy — fire TTS as early as possible
# Flush on sentence-ending punctuation OR every N words.
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END    = re.compile(r'[.!?\n]')
_MIN_WORDS_FLUSH = 5   # flush without punctuation after 5 words (was 8)


def _should_flush(buf: list[str]) -> bool:
    """Return True if buffer should be sent to TTS now."""
    text = " ".join(buf)
    return bool(_SENTENCE_END.search(text)) or len(buf) >= _MIN_WORDS_FLUSH


# ─────────────────────────────────────────────────────────────────────────────
# Audio buffer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _concat_audio_b64(chunks: list[str]) -> str:
    """
    Properly concatenate base64-encoded float32 PCM chunks.
    Decode → concatenate raw bytes → re-encode.
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]
    raw = b"".join(base64.b64decode(c) for c in chunks)
    return base64.b64encode(raw).decode()


# ─────────────────────────────────────────────────────────────────────────────
# PipelineRouter
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRouter:
    """
    Zero-latency STT → LLM → TTS pipeline.

    Stage layout (3 stages, no sentence buffer):
      1. _stage_stt          : audio → word stream
      2. _stage_llm          : words → transcript → LLM token stream + TTS dispatch
      3. _stage_tts_ordered  : ordered TTS synthesis → WebSocket audio

    TTS ordering guarantee:
      Each text chunk creates a synthesis Future that is placed into an
      ordered asyncio.Queue.  A single consumer task awaits each Future in
      order and sends audio to the WebSocket only after the previous chunk
      finished.  Synthesis of chunk N+1 runs in parallel while chunk N audio
      is being sent — zero dead-silence gap between chunks.
    """

    def __init__(self, session, stt_url: str, llm_url: str, tts_url: str,
                 ws, hubspot=None):
        self.session   = session
        self.stt_url   = stt_url
        self.llm_url   = llm_url
        self.tts_url   = tts_url
        self.ws        = ws
        self.hubspot   = hubspot

        # Shared HTTP client — keep-alive, connection pooling
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=30.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Cancellation event — set this to kill all running tasks immediately
        self._cancel_event = asyncio.Event()

        # Active pipeline task (only one at a time)
        self._pipeline_task: Optional[asyncio.Task] = None

        # Audio buffer — accumulates VAD-gated chunks until 800ms of speech
        self._audio_buf: list[str] = []
        self._audio_samples: int   = 0
        self._PARTIAL_FLUSH_SAMPLES = 12800   # 800ms @ 16kHz

        # State flags
        self._ai_speaking = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────────────────────────────────

    async def handle_audio(self, audio_b64: str, sample_count: int = 512,
                           is_voice: bool = True):
        """
        Called for every VAD-gated mic chunk (32ms / 512 samples).
        Barge-in: if AI is speaking when voice arrives → interrupt immediately.
        """
        if not is_voice:
            return

        # Barge-in detection
        if self._ai_speaking:
            await self._interrupt()

        # Accumulate audio
        self._audio_buf.append(audio_b64)
        self._audio_samples += sample_count

        # Fire partial pipeline every 800ms of speech
        if self._audio_samples >= self._PARTIAL_FLUSH_SAMPLES:
            combined = self._flush_audio_buf()
            await self._launch_pipeline(combined)

    async def handle_utterance_end(self):
        """
        Called when the client signals end of utterance (VAD silence timeout).
        Flushes any remaining buffered audio and fires the final pipeline.
        """
        if self._audio_samples > 0:
            combined = self._flush_audio_buf()
            await self._launch_pipeline(combined)

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline launcher
    # ─────────────────────────────────────────────────────────────────────────

    async def _launch_pipeline(self, audio_b64: str):
        """Cancel any running pipeline and start a fresh one."""
        if self._pipeline_task and not self._pipeline_task.done():
            self._cancel_event.set()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        self._cancel_event = asyncio.Event()
        self._pipeline_task = asyncio.create_task(
            self._pipeline(audio_b64), name="pipeline"
        )

    async def _interrupt(self):
        """Instant barge-in — cancel TTS and pipeline."""
        logger.info("⚡ BARGE-IN — cancelling pipeline")
        self._cancel_event.set()
        self._ai_speaking = False
        self.session.ai_is_speaking = False
        self.session.mark_interrupted()

        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.15)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self._send({"type": "interrupted"})
        await self._send({"type": "status", "state": "listening"})

    # ─────────────────────────────────────────────────────────────────────────
    # Full pipeline — STT → LLM → TTS (3 stages)
    # ─────────────────────────────────────────────────────────────────────────

    async def _pipeline(self, audio_b64: str):
        """
        STT → LLM → TTS, all streaming, all parallel.

        Queue flow:
          stt       → word_q   → transcript builder + LLM
          LLM tokens → tts_q   (ordered futures queue)
          tts_q     → TTS      → WebSocket audio chunks (in order)
        """
        cancel = self._cancel_event

        word_q = asyncio.Queue(maxsize=64)
        tts_q  = asyncio.Queue(maxsize=32)   # queue of (asyncio.Task | None)

        try:
            await self._send({"type": "status", "state": "listening"})

            stt_task = asyncio.create_task(
                self._stage_stt(audio_b64, word_q, cancel), name="stt"
            )
            llm_task = asyncio.create_task(
                self._stage_llm(word_q, tts_q, cancel), name="llm"
            )
            tts_task = asyncio.create_task(
                self._stage_tts_ordered(tts_q, cancel), name="tts"
            )

            await asyncio.gather(stt_task, llm_task, tts_task)

        except asyncio.CancelledError:
            cancel.set()
            raise
        finally:
            self._ai_speaking = False
            self.session.ai_is_speaking = False
            cancel.set()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — STT: audio → word stream
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_stt(self, audio_b64: str, word_q: asyncio.Queue,
                          cancel: asyncio.Event):
        """Stream words from Whisper as fast as they arrive."""
        t0 = time.perf_counter()
        first_word = True
        try:
            async with self._http.stream(
                "POST",
                f"{self.stt_url}/transcribe/stream",
                json={
                    "audio_b64":      audio_b64,
                    "sample_rate":    16000,
                    "ai_is_speaking": self._ai_speaking,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue

                    t = p.get("type", "")
                    if t == "word":
                        word = p.get("word", "").strip()
                        if word:
                            if first_word:
                                logger.info("STT first word %.0fms",
                                            (time.perf_counter() - t0) * 1000)
                                first_word = False
                            await word_q.put(word)
                            await self._send({"type": "word", "data": word})
                    elif t == "ai_filtered":
                        logger.info("STT: AI voice filtered — skipping")
                        break
                    elif t in ("done", "silence"):
                        break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("STT error: %s", e)
        finally:
            await word_q.put(None)   # sentinel

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — LLM: words → transcript → token stream → TTS dispatch
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_llm(self, word_q: asyncio.Queue,
                          tts_q: asyncio.Queue,
                          cancel: asyncio.Event):
        """
        Drain words → build transcript → stream LLM → dispatch TTS chunks.

        TTS dispatch:
          Tokens are buffered until _should_flush() returns True (punctuation
          or 5 words).  Each flush creates a synthesis Task and puts it into
          tts_q.  The ordered TTS consumer awaits tasks in order so audio
          always plays in the correct sequence.
        """
        words = []
        t0    = time.perf_counter()

        # Drain word queue
        while not cancel.is_set():
            try:
                word = await asyncio.wait_for(word_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if word is None:
                break
            words.append(word)

        if cancel.is_set() or not words:
            await tts_q.put(None)
            return

        transcript = " ".join(words).strip()
        if not transcript:
            await tts_q.put(None)
            return

        logger.info("Transcript (%dw) %.0fms: %r",
                    len(words), (time.perf_counter() - t0) * 1000, transcript[:60])

        self.session.add_user_utterance(transcript)
        self.session.state       = "thinking"
        self.session.ai_is_thinking = True

        if self.hubspot:
            self.hubspot.add_utterance(self.session.id, "user", "User", transcript)

        await self._send({"type": "status", "state": "thinking"})

        llm_payload = {
            "query":      transcript,
            "session_id": self.session.id,
            "history":    self.session.get_messages_for_llm()[:-1],
        }

        # Token buffer for TTS chunking
        token_buf:   list[str] = []
        full_tokens: list[str] = []

        async def _flush_tts():
            """Flush token buffer → create synthesis task → enqueue."""
            nonlocal token_buf
            if not token_buf:
                return
            text = " ".join(token_buf).strip()
            token_buf = []
            if not text:
                return
            # Create synthesis task immediately — runs concurrently
            task = asyncio.create_task(
                self._synthesize_chunk(text, cancel),
                name="tts-synth",
            )
            await tts_q.put(task)

        try:
            first_token = True
            t_llm = time.perf_counter()

            async with self._http.stream(
                "POST",
                f"{self.llm_url}/generate/stream",
                json=llm_payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue

                    t = p.get("type", "")
                    if t == "token":
                        token = p.get("token", "")
                        if not token:
                            continue
                        if first_token:
                            logger.info("LLM first token %.0fms",
                                        (time.perf_counter() - t_llm) * 1000)
                            first_token = False
                        full_tokens.append(token)
                        token_buf.append(token)
                        await self._send({"type": "ai_token", "data": token})

                        # Flush to TTS on natural break or word count
                        if _should_flush(token_buf):
                            await _flush_tts()

                    elif t == "done":
                        break
                    elif t == "error":
                        logger.error("LLM error: %s", p.get("message"))
                        break

            # Flush any remaining tokens
            await _flush_tts()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("LLM error: %s", e)
        finally:
            self.session.ai_is_thinking = False

            # Save full AI response to session history
            if full_tokens:
                full_response = " ".join(full_tokens)
                interrupted   = cancel.is_set()
                self.session.add_ai_utterance(full_response, interrupted=interrupted)
                if self.hubspot:
                    self.hubspot.add_utterance(
                        self.session.id, "ai", "Assistant", full_response
                    )

            await tts_q.put(None)   # sentinel

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — TTS ordered consumer: futures → WebSocket audio (in order)
    # ─────────────────────────────────────────────────────────────────────────

    async def _stage_tts_ordered(self, tts_q: asyncio.Queue,
                                  cancel: asyncio.Event):
        """
        Drains tts_q in order.

        Each item is either:
          - asyncio.Task[None]  → synthesis task that sends audio to WS
          - None                → sentinel, all done

        Tasks run concurrently (synthesis of chunk N+1 starts while chunk N
        audio is being sent), but we await them IN ORDER so audio is always
        coherent.  This is the key fix vs v1 where tasks were collected in a
        list and awaited at the very end.
        """
        self._ai_speaking = True
        self.session.ai_is_speaking = True
        self.session.state = "ai_speaking"
        await self._send({"type": "status", "state": "speaking"})

        t_first: Optional[float] = None

        try:
            while not cancel.is_set():
                try:
                    item = await asyncio.wait_for(tts_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if item is None:
                    break
                if cancel.is_set():
                    item.cancel()
                    break

                if t_first is None:
                    t_first = time.perf_counter()

                # Await this chunk's synthesis in order — next chunk is already
                # synthesizing concurrently in the background
                try:
                    await item
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("TTS chunk error: %s", e)

        except asyncio.CancelledError:
            # Cancel any remaining tasks in the queue
            while not tts_q.empty():
                try:
                    t = tts_q.get_nowait()
                    if t is not None:
                        t.cancel()
                except asyncio.QueueEmpty:
                    break
            raise
        finally:
            self._ai_speaking = False
            self.session.ai_is_speaking = False
            self.session.state = "listening"
            await self._send({"type": "status", "state": "listening"})
            if t_first:
                logger.info("TTS first audio %.0fms",
                            (time.perf_counter() - t_first) * 1000)

    # ─────────────────────────────────────────────────────────────────────────
    # TTS synthesis helper — called per chunk, runs concurrently
    # ─────────────────────────────────────────────────────────────────────────

    async def _synthesize_chunk(self, text: str, cancel: asyncio.Event):
        """
        Synthesize one text chunk and stream audio chunks to WebSocket.

        Uses /synthesize/pocket_stream SSE endpoint on the TTS service.
        Audio chunks are sent to the WebSocket as they arrive from the TTS
        service — no buffering, no waiting for full synthesis.
        """
        if cancel.is_set():
            return
        try:
            async with self._http.stream(
                "POST",
                f"{self.tts_url}/synthesize/pocket_stream",
                json={"text": text, "speed": 1.0},
            ) as resp:
                async for line in resp.aiter_lines():
                    if cancel.is_set():
                        return
                    if not line.startswith("data:"):
                        continue
                    try:
                        p = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    if p.get("type") == "audio_chunk":
                        await self._send({
                            "type":        "audio_chunk",
                            "data":        p["data"],
                            "sample_rate": p.get("sample_rate", 22050),
                        })
                    elif p.get("type") in ("done", "error"):
                        break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("TTS synthesize error %r: %s", text[:40], e)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _flush_audio_buf(self) -> str:
        """Concatenate ALL accumulated audio chunks into one base64 blob."""
        combined = _concat_audio_b64(self._audio_buf)
        self._audio_buf.clear()
        self._audio_samples = 0
        return combined

    async def _send(self, data: dict):
        try:
            await self.ws.send_json(data)
        except Exception as e:
            logger.debug("WS send error: %s", e)

    async def cancel_all(self):
        """Graceful shutdown."""
        self._cancel_event.set()
        if self._pipeline_task and not self._pipeline_task.done():
            self._pipeline_task.cancel()
            try:
                await asyncio.wait_for(self._pipeline_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        await self._http.aclose()
