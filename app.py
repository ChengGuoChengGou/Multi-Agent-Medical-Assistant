import os
import logging
import uuid
import tempfile
from typing import Dict, Union, Optional, List
import glob
import threading
import time
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import uvicorn
import requests
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
from agents.agent_decision import process_query
from agents.mcp_client import get_mcp_client, shutdown_mcp_client
from agents.memory_module import get_memory_store
from edge_tts_service import edge_tts_generate, list_chinese_voices, get_voice_id
from sse_utils import sse_stream_chat
from exceptions import MedicalAssistantError, AgentError, ValidationError, FileUploadError

# Security middleware
import secrets as _secrets
import uuid as _uuid
from middleware.security import (
    SecurityHeadersMiddleware,
    RequestLoggingMiddleware,
    CSRFProtection,
    sanitize_input,
    sanitize_filename as sec_sanitize_filename,
    validate_mime_type,
    secure_error_response,
)

# Rate limiting
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    logger.warning("[SlowAPI] Not installed, rate limiting disabled. Install: pip install slowapi")

# Load configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")


# ─── Observability: Request tracking middleware + /metrics ──────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request count and latency for all endpoints."""
    import time as _time
    path = request.url.path
    method = request.method
    metrics.inc("http_requests_total", labels={"method": method, "path": path})
    # Request tracing: generate and propagate X-Request-ID
    request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())[:12]
    start = _time.monotonic()
    try:
        response = await call_next(request)
        elapsed = _time.monotonic() - start
        metrics.observe("http_request_duration_seconds", elapsed, labels={"method": method, "path": path})
        metrics.inc("http_responses_total", labels={"status": str(response.status_code)})
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
        return response
    except Exception as exc:
        elapsed = _time.monotonic() - start
        metrics.observe("http_request_duration_seconds", elapsed, labels={"method": method, "path": path})
        metrics.inc("http_errors_total", labels={"path": path})
        raise

@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")



# Rate limiter: 30 requests/minute per IP
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please wait.", "retry_after": str(exc.detail)}
        )
else:
    # No-op decorator when slowapi not installed
    def limiter(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# --- Security Middleware Registration ---
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
if SLOWAPI_AVAILABLE:
    from slowapi.middleware import SlowAPIMiddleware
    app.add_middleware(SlowAPIMiddleware)

# CSRF protection
CSRF_SECRET = os.environ.get("CSRF_SECRET_KEY", _secrets.token_hex(32))
csrf_protection = CSRFProtection(CSRF_SECRET)

# MCP Agent instance (initialized on startup)
mcp_client = None  # MCP client manager, initialized on startup

# Set up directories
UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"

# Create directories if they don't exist
for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files directory
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize ElevenLabs client
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'dcm', 'nii', 'nii.gz', 'mha'}

# ==================== MCP Lifecycle ====================
@app.on_event("startup")
async def startup_mcp():
    """Initialize MCP connections on application startup."""
    global mcp_client
    try:
        mcp_client = get_mcp_client()
        await mcp_client.initialize()
        logger.info("[MCP] MCP Client initialized successfully")
    except Exception as e:
        logger.warning(f"[MCP] Failed to initialize MCP Client: {e}")
        mcp_client = None

@app.on_event("shutdown")
async def shutdown_mcp():
    """Clean up MCP connections on application shutdown."""
    # Phase 17: Initialize structured JSON logging
    setup_json_logging()
    import logging as _logging
    _logging.getLogger(__name__).info('Observability initialized: JSON logging + /metrics')
    global mcp_client
    if mcp_client:
        try:
            await shutdown_mcp_client()
            logger.info("[MCP] MCP Client cleaned up")
        except Exception as e:
            logger.warning(f"[MCP] MCP cleanup error: {e}")
        mcp_client = None
# ==================== End MCP Lifecycle ====================

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            logger.info("Cleaned up old speech files.")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes

# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

# Models imported from models.py (Phase 20: centralized API contracts)
from models import QueryRequest, SpeechRequest, ChatResponse, ErrorResponse, HealthResponse, api_success, api_error

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.exception_handler(MedicalAssistantError)
async def medical_error_handler(request: Request, exc: MedicalAssistantError):
    """Global handler for structured application errors."""
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with dependency status."""
    import time as _time
    checks = {}
    
    # LLM config check
    try:
        api_key = config.main_llm_config.api_key
        checks["llm_config"] = "ok" if api_key and len(api_key) > 10 else "missing_api_key"
    except Exception:
        checks["llm_config"] = "error"
    
    # MCP client check
    checks["mcp_client"] = "connected" if mcp_client is not None else "not_initialized"
    
    # Upload directories check
    for name, path in [("uploads_backend", UPLOAD_FOLDER), ("skin_lesion_output", SKIN_LESION_OUTPUT)]:
        checks[name] = "ok" if os.path.isdir(path) else "missing"
    
    # Overall status
    has_critical_error = any(v in ("error", "missing") for k, v in checks.items() if k != "mcp_client")
    status = "degraded" if (mcp_client is None or has_critical_error) else "healthy"
    
    return {
        "status": status,
        "version": "3.3.0",
        "checks": checks,
        "timestamp": int(_time.time()),
    }


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe - is the process alive?"""
    return {"status": "alive", "timestamp": int(time.time())}

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe - can we serve traffic?"""
    checks = {}
    
    # LLM config
    try:
        api_key = config.main_llm_config.api_key
        checks["llm_config"] = bool(api_key and len(api_key) > 10)
    except Exception:
        checks["llm_config"] = False
    
    # Upload dirs accessible
    checks["uploads"] = os.path.isdir(UPLOAD_FOLDER)
    
    ready = all(checks.values())
    return Response(
        content=json.dumps({"ready": ready, "checks": checks}),
        status_code=200 if ready else 503,
        media_type="application/json"
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: QueryRequest, 
    response: Response, 
    req: Request,
    session_id: Optional[str] = Cookie(None)
):
    """Process user text query through the multi-agent system (async with caching)."""
    import asyncio
    import hashlib
    import time as _time
from observability import metrics, setup_json_logging
    
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Session fingerprint header for client-side tracking
    response.headers["X-Session-ID"] = session_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Response cache (TTL=300s, max 256 entries)
    if not hasattr(chat, "_cache"):
        chat._cache = {}  # {hash: (timestamp, result)}
        chat._cache_ttl = 300
        chat._cache_max = 256
    
    # Check cache for identical query (same session + same text)
    cache_key = hashlib.sha256(f"{session_id}:{request.query}".encode()).hexdigest()[:16]
    cached = chat._cache.get(cache_key)
    if cached and (_time.time() - cached[0]) < chat._cache_ttl:
        result = cached[1].copy()
        result["cached"] = True
        return result
    
    try:
        # --- Memory recall: inject relevant long-term memories into query context ---
        memory_context = ""
        try:
            mem_store = get_memory_store()
            recalled = mem_store.recall(request.query, user_id=session_id, limit=3)
            if recalled:
                memory_context = "\n\n[Patient Memory Context]\n" + "\n".join(
                    f"- {m}" for m in recalled
                )
                logger.info(f"[chat] Recalled {len(recalled)} memories for session {session_id[:8]}...")
        except Exception as mem_err:
            logger.warning(f"[chat] Memory recall failed (non-fatal): {mem_err}")
        
        enhanced_query = request.query + memory_context if memory_context else request.query
        
        # Run synchronous process_query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(None, process_query, enhanced_query, session_id)
        response_text = response_data['messages'][-1].content
        
        # --- Memory store: extract and persist medical facts from conversation ---
        try:
            mem_store = get_memory_store()
            # Store the Q&A pair for long-term memory
            mem_store.remember(
                f"Q: {request.query}\nA: {response_text}",
                user_id=session_id,
                metadata={"agent": response_data.get("agent_name", "unknown")},
            )
        except Exception as mem_err:
            logger.warning(f"[chat] Memory store failed (non-fatal): {mem_err}")
        
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] in ("SKIN_LESION_AGENT", "HUMAN_VALIDATION"):
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                logger.warning("Skin Lesion Output path does not exist.")
        
        # Store in cache (evict oldest if full)
        if len(chat._cache) >= chat._cache_max:
            oldest_key = min(chat._cache, key=lambda k: chat._cache[k][0])
            del chat._cache[oldest_key]
        chat._cache[cache_key] = (_time.time(), result.copy())
        
        return result
    except Exception as e:
        logger.error(f"[chat] Internal error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")


@app.post("/chat/stream")
async def chat_stream(request: QueryRequest, req: Request):
    """SSE streaming endpoint for chat responses.

    Returns Server-Sent Events with incremental response tokens.
    """
    return await sse_stream_chat(
        query=request.query,
        conversation_history=request.conversation_history,
        process_fn=process_query,
    )


@app.post("/upload")
async def upload_image(
    request: Request,
    response: Response,
    image: UploadFile = File(...), 
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    """Process medical image uploads with optional text input."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())
    response.headers["X-Session-ID"] = session_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Validate file type (extension + MIME)
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400, 
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )
    
    # MIME type validation (defense-in-depth)
    if not validate_mime_type(image.filename, image.content_type):
        logger.warning(f"[upload] MIME mismatch: {image.filename} / {image.content_type}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "File type mismatch detected. Please upload a valid image file."
            }
        )
    
    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413, 
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )
    
    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data['messages'][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "status": "success",
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] in ("SKIN_LESION_AGENT", "HUMAN_VALIDATION"):
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                logger.warning("Skin Lesion Output path does not exist.")
        
        # Remove temporary file after sending
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file: {e}")
        
        return result
    except Exception as e:
        logger.error(f"[upload-image] Internal error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...), 
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for medical AI outputs."""
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)
        
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query, session_id=session_id)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data['messages'][-1].content
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_data['messages'][-1].content
            }
    except Exception as e:
        logger.error(f"[validate] Internal error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your request.")

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Endpoint to transcribe speech using ElevenLabs API"""
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )
    
    try:
        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"
        
        # Read and save the file
        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)
        
        # Debug: Print file size to check if it's empty
        file_size = os.path.getsize(temp_audio)
        logger.info(f"Received audio file size: {file_size} bytes")
        
        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )
        
        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"
        
        try:
            # Use pydub with format detection
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")
            
            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            logger.info(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            logger.info("Converted audio file into byte array successfully")

            transcription = client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )
            
            # Clean up temp files
            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                logger.debug(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                logger.warning(f"Could not delete file: {e}")
            
            if transcription.text:
                return {"transcript": transcription.text}
            else:
                logger.error("[transcribe] API returned empty transcription")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Speech transcription failed. Please try again."}
                )

        except Exception as e:
            logger.error(f"[transcribe] Audio processing error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Error processing audio. Please try again."}
            )
                
    except Exception as e:
        logger.error(f"[transcribe] Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Transcription service unavailable. Please try again."}
        )

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using Edge TTS (free) with ElevenLabs fallback."""
    try:
        text = request.text
        if not text:
            return JSONResponse(status_code=400, content={"error": "Text is required"})

        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"

        # --- Strategy 1: Edge TTS (free, no API key) ---
        try:
            voice_id = get_voice_id(request.voice_id) if request.voice_id != "EXAMPLE_VOICE_ID" \
                else config.speech.edge_tts_voice
            audio_bytes = await edge_tts_generate(
                text=text,
                voice=voice_id,
                rate=config.speech.edge_tts_rate,
                pitch=config.speech.edge_tts_pitch,
            )
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"[generate-speech] Edge TTS OK ({len(audio_bytes)} bytes, voice={voice_id})")
            return FileResponse(path=temp_audio_path, media_type="audio/mpeg", filename="generated_speech.mp3")
        except Exception as edge_err:
            logger.warning(f"[generate-speech] Edge TTS failed: {edge_err}, trying ElevenLabs...")

        # --- Strategy 2: ElevenLabs fallback (requires API key) ---
        if config.speech.eleven_labs_api_key:
            try:
                selected_voice_id = request.voice_id if request.voice_id != "EXAMPLE_VOICE_ID" \
                    else config.speech.eleven_labs_voice_id
                elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": config.speech.eleven_labs_api_key,
                }
                payload = {
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
                }
                response = requests.post(elevenlabs_url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    with open(temp_audio_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"[generate-speech] ElevenLabs fallback OK ({len(response.content)} bytes)")
                    return FileResponse(path=temp_audio_path, media_type="audio/mpeg", filename="generated_speech.mp3")
                else:
                    logger.error(f"[generate-speech] ElevenLabs API error: {response.status_code}")
            except Exception as el_err:
                logger.error(f"[generate-speech] ElevenLabs fallback failed: {el_err}")

        return JSONResponse(status_code=503, content={"error": "All TTS services unavailable. Please try again later."})

    except Exception as e:
        logger.error(f"[generate-speech] Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Speech generation failed. Please try again."})


@app.get("/list-voices")
async def list_voices_endpoint():
    """List available Chinese TTS voices (Edge TTS, free)."""
    try:
        voices = await list_chinese_voices()
        return {"voices": voices, "default": config.speech.edge_tts_voice}
    except Exception as e:
        logger.error(f"[list-voices] Error: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to list voices."})

# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)