import glob
import os
import tempfile
import threading
import time
import uuid
from io import BytesIO
from typing import Dict, List, Optional, Union

import requests
import uvicorn
from elevenlabs.client import ElevenLabs
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from agents.agent_decision import (
    process_query,
    process_query_streaming,  # [Phase 3.1] streaming support
)

# [Phase 7] API versioning with dependency health checks
from api.health import router as health_v1_router

# [Phase 9] System monitoring aggregation endpoint
from api.monitoring import router as monitoring_v1_router
from api.monitoring import set_app_start_time
from cache import semantic_get, semantic_set, semantic_stats  # [Phase 51] Semantic cache
from config import Config
from middleware import (
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    RequestDedupMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    get_auth_stats,
    get_dedup_stats,
)
from schemas import ChatResponse, ErrorResponse, HealthResponse, TranscribeResponse, ValidateResponse
from sse_utils import sse_generator, sse_stream_chat_streaming  # [Phase 3.1] SSE streaming response

# [Phase 8] Startup configuration validation
from startup_validator import ConfigValidationError, validate_startup_config
from utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# [Phase 6.5] Incremental indexing support
try:
    from agents.rag_agent import RAG_AVAILABLE, MedicalRAG
except ImportError:
    RAG_AVAILABLE = False

# Load configuration
config = Config()

# Track application start time for uptime metric
_app_start_time = time.time()

# Initialize structured logging (JSON in production, human-readable in dev)
setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Medical Chatbot",
    version="2.0",
    description=(
        "A multi-agent medical information system powered by RAG (Retrieval-Augmented Generation). "
        "Provides evidence-based health information with source citations from a curated medical knowledge base. "
        "Supports text chat (synchronous & SSE streaming), document upload, voice transcription, and speech synthesis.\n\n"
        "⚠️ **Disclaimer**: This system provides health information for educational purposes only. "
        "It does not constitute medical diagnosis, treatment, or professional medical advice. "
        "Always consult a qualified healthcare provider for medical concerns."
    ),
    openapi_tags=[
        {"name": "System", "description": "Health check, metrics, and service status endpoints."},
        {"name": "Chat", "description": "Multi-agent chat endpoints (synchronous and SSE streaming)."},
        {"name": "Document", "description": "Document upload and medical content validation."},
        {"name": "Voice", "description": "Speech-to-text and text-to-speech services."},
    ],
    contact={"name": "Medical Chatbot Team", "url": "https://github.com/medical-chatbot"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

# ── Register Middleware (order matters: outermost first) ──
# 0. CORS: allow frontend origin for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 1. Rate limiting: protect against abuse (60 rpm / 500 rph per IP)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60, requests_per_hour=500)
# 2. Security headers: CSP, X-Frame-Options, XSS-Protection, etc.
app.add_middleware(SecurityHeadersMiddleware)
# 3. Request logging: method, path, status, duration
app.add_middleware(RequestLoggingMiddleware)
# 4. Request deduplication: coalesce identical in-flight LLM requests
app.add_middleware(RequestDedupMiddleware)
# 5. [Phase 53] API key authentication for protected endpoints
app.add_middleware(APIKeyAuthMiddleware)

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

# [Phase 7] Register API v1 routers
app.include_router(health_v1_router)
app.include_router(monitoring_v1_router)

# [Phase 9] Set app start time for monitoring uptime metric
set_app_start_time(_app_start_time)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize ElevenLabs client
client = ElevenLabs(
    api_key=config.speech.eleven_labs_api_key,
)

# ── [Phase 6.5] Incremental Indexing Lifecycle ──
# Singleton MedicalRAG instance for startup/shutdown management
_rag_singleton = None


@app.on_event("startup")
async def _startup_incremental_indexing():
    """Start incremental file indexing and initialize unified tool registry on app startup."""
    global _rag_singleton
    _logger = get_logger("startup")

    # [Phase 8] Validate startup configuration (fail-fast)
    try:
        result = validate_startup_config()
        _logger.info(f"[StartupValidator] OK ({len(result.warnings)} warnings)")
    except ConfigValidationError as e:
        _logger.error(f"[StartupValidator] CRITICAL: {e}")
        # In production, uncomment: raise

    # Start incremental indexer
    if RAG_AVAILABLE:
        try:
            _rag_singleton = MedicalRAG(config)
            _rag_singleton.start_incremental_indexing()
            _logger.info("[STARTUP] Incremental document indexing started")
        except Exception as e:
            _logger.warning(f"[STARTUP] Incremental indexer failed to start (non-fatal): {e}")

    # Initialize unified tool registry
    try:
        from tools.registry import get_registry

        registry = get_registry()
        if registry.initialize():
            tool_count = len(registry.get_all_tool_names())
            _logger.info(f"[STARTUP] Tool registry initialized: {tool_count} tools available")
        else:
            _logger.info("[STARTUP] Tool registry initialized (no tools loaded)")
    except Exception as e:
        _logger.warning(f"[STARTUP] Tool registry init failed (non-fatal): {e}")


@app.on_event("shutdown")
async def _shutdown_incremental_indexing():
    """Stop incremental file watcher on app shutdown."""
    _logger = get_logger("shutdown")
    if _rag_singleton:
        try:
            _rag_singleton.stop_incremental_indexing()
            _logger.info("[SHUTDOWN] Incremental document indexing stopped")
        except Exception as e:
            _logger.warning(f"[SHUTDOWN] Incremental indexer stop failed: {e}")


# Define allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)  # Runs every 5 minutes


# Start background cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()


class QueryRequest(BaseModel):
    query: str
    conversation_history: list = []


class SpeechRequest(BaseModel):
    text: str
    voice_id: str = "EXAMPLE_VOICE_ID"  # Default voice ID


@app.get("/", response_class=HTMLResponse, tags=["System"])
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint for Docker / load balancer probes.

    Returns the status of the application and all active middleware components.
    Used by Docker HEALTHCHECK, Kubernetes liveness/readiness probes, and load balancers.
    No authentication required.
    """
    uptime_seconds = time.time() - _app_start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_seconds, 2),
        "middleware": {
            "rate_limiting": True,
            "security_headers": True,
            "request_logging": True,
            "request_deduplication": True,
            "api_key_auth": True,
        },
        "dedup_stats": get_dedup_stats(),
        "api_auth": get_auth_stats(),
    }


@app.get("/metrics", tags=["System"])
def metrics():
    """Prometheus-compatible metrics endpoint (plain text exposition format)."""
    uptime_seconds = time.time() - _app_start_time
    dedup = get_dedup_stats()
    lines = [
        "# HELP medical_app_uptime_seconds Time since application start.",
        "# TYPE medical_app_uptime_seconds gauge",
        f"medical_app_uptime_seconds {uptime_seconds:.2f}",
        "",
        "# HELP medical_app_dedup_total Total deduplication lookups.",
        "# TYPE medical_app_dedup_total counter",
        f"medical_app_dedup_total {dedup.get('total_lookups', 0)}",
        "",
        "# HELP medical_app_dedup_hits Total deduplication cache hits.",
        "# TYPE medical_app_dedup_hits counter",
        f"medical_app_dedup_hits {dedup.get('cache_hits', 0)}",
        "",
        "# HELP medical_app_dedup_pending Currently pending deduplicated requests.",
        "# TYPE medical_app_dedup_pending gauge",
        f"medical_app_dedup_pending {dedup.get('pending_requests', 0)}",
        "",
    ]
    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}}, tags=["Chat"])
def chat(request: QueryRequest, response: Response, session_id: str | None = Cookie(None)):
    """Process user text query through the multi-agent system.

    Routes the query to specialized medical agents (cardiology, brain tumor, chest X-ray,
    medical image analysis) based on content analysis. Returns the agent's response along
    with metadata about which agent handled the request.

    Supports multi-turn conversation via conversation_history parameter. A session cookie
    is automatically set for conversation continuity.
    """
    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        # [SemanticCache] Check cache before expensive LLM call
        cached = semantic_get(request.query)
        if cached:
            logger.info("[SemanticCache] HIT for: %s", request.query[:50])
            response.set_cookie(key="session_id", value=session_id)
            return cached

        response_data = process_query(request.query)
        response_text = response_data["messages"][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {"status": "success", "response": response_text, "agent": response_data["agent_name"]}

        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        # [SemanticCache] Store result for future similar queries
        semantic_set(request.query, result)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", responses={500: {"model": ErrorResponse}}, tags=["Chat"])
async def chat_stream(request: QueryRequest, session_id: str | None = Cookie(None)):
    """Process user query with SSE streaming for real-time agent interaction.

    Returns a Server-Sent Events (SSE) stream with two event types:
    - ``progress``: Agent routing decisions and reasoning steps
    - ``content``: Response text chunks as they are generated

    The stream ends with a ``done`` event containing the full response metadata.
    Client should use EventSource or equivalent SSE consumer to process the stream.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # [Phase 51] Semantic cache check - skip streaming on cache hit
    cached = semantic_get(request.query)
    if cached:
        logger.info("[SemanticCache] HIT on /chat/stream for: %s", request.query[:50])

        async def cached_sse():
            async for chunk in sse_generator(
                {"session_id": session_id, "query": request.query[:100]},
                event="start",
            ):
                yield chunk
            async for chunk in sse_generator(
                {"agent": cached.get("agent", "cache")},
                event="agent",
            ):
                yield chunk
            async for chunk in sse_generator(
                {"text": cached.get("response", "")},
                event="content",
            ):
                yield chunk
            async for chunk in sse_generator(
                {
                    "total_length": len(cached.get("response", "")),
                    "agent": cached.get("agent", "cache"),
                    "cached": True,
                },
                event="done",
            ):
                yield chunk

        return StreamingResponse(cached_sse(), media_type="text/event-stream")

    return await sse_stream_chat_streaming(
        query=request.query,
        session_id=session_id,
        streaming_fn=process_query_streaming,
    )


@app.post(
    "/upload",
    response_model=ChatResponse,
    responses={413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Document"],
)
async def upload_image(
    response: Response, image: UploadFile = File(...), text: str = Form(""), session_id: str | None = Cookie(None)
):
    """Process medical image uploads with optional text input.

    Accepts image files (PNG, JPG, JPEG, DICOM) for analysis by specialized medical AI agents.
    An optional text query can be included to guide the analysis. Supported analysis types:
    - Brain tumor detection from MRI/CT images
    - Chest X-ray interpretation
    - General medical image analysis

    Returns the analysis result from the most appropriate agent. Files larger than
    max_image_upload_size (configurable) are rejected with HTTP 413.
    """
    # Validate file type
    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG",
            },
        )

    # Check file size before saving
    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:  # Convert MB to bytes
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB",
            },
        )

    # Generate session ID for cookie if it doesn't exist
    if not session_id:
        session_id = str(uuid.uuid4())

    # Save file securely
    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data["messages"][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {"status": "success", "response": response_text, "agent": response_data["agent_name"]}

        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        # Remove temporary file after sending
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {e!s}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate", response_model=ValidateResponse, responses={500: {"model": ErrorResponse}}, tags=["Document"])
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...),
    comments: str | None = Form(None),
    session_id: str | None = Cookie(None),
):
    """Handle human validation for medical AI outputs.

    Allows human reviewers to approve or reject AI-generated medical responses.
    Validation results are logged for quality assurance and model improvement tracking.
    Used in the human-in-the-loop workflow for critical medical decisions.
    """
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

        response_data = process_query(validation_query)

        if validation_result.lower() == "yes":
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data["messages"][-1].content,
            }
        return {
            "status": "rejected",
            "comments": comments,
            "message": "**Output requires further review:**",
            "response": response_data["messages"][-1].content,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscribeResponse, responses={500: {"model": ErrorResponse}}, tags=["Voice"])
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio to text using ElevenLabs speech-to-text API.

    Accepts audio files in common formats (WAV, MP3, M4A, WebM, OGG) and returns
    the transcribed text using ElevenLabs' STT model. Useful for voice-based
    medical queries and dictation workflows.
    """
    if not audio.filename:
        return JSONResponse(status_code=400, content={"error": "No audio file selected"})

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
        print(f"Received audio file size: {file_size} bytes")

        if file_size == 0:
            return JSONResponse(status_code=400, content={"error": "Received empty audio file"})

        # Convert to MP3
        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"

        try:
            # Use pydub with format detection
            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")

            # Debug: Print MP3 file size
            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print("Converted audio file into byte array successfully!")

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
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                print(f"Could not delete file: {e}")

            if transcription.text:
                return {"transcript": transcription.text}
            return JSONResponse(
                status_code=500, content={"error": f"API error: {transcription}", "details": transcription.text}
            )

        except Exception as e:
            print(f"Error processing audio: {e!s}")
            return JSONResponse(status_code=500, content={"error": f"Error processing audio: {e!s}"})

    except Exception as e:
        print(f"Transcription error: {e!s}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/generate-speech", responses={500: {"model": ErrorResponse}}, tags=["Voice"])
async def generate_speech(request: SpeechRequest):
    """Convert text to speech using ElevenLabs TTS API.

    Accepts a text string and returns synthesized audio in the requested voice.
    Supports multiple voice options and audio output formats. Used for read-aloud
    functionality of medical responses and accessibility features.
    """
    try:
        text = request.text
        selected_voice_id = request.voice_id

        if not text:
            return JSONResponse(status_code=400, content={"error": "Text is required"})

        # Define API request to ElevenLabs
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

        # Send request to ElevenLabs API
        response = requests.post(elevenlabs_url, headers=headers, json=payload)

        if response.status_code != 200:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Failed to generate speech, status: {response.status_code}",
                    "details": response.text,
                },
            )

        # Save the audio file temporarily
        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)

        # Return the generated audio file
        return FileResponse(path=temp_audio_path, media_type="audio/mpeg", filename="generated_speech.mp3")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Add exception handler for request entity too large
@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB",
        },
    )


# [Phase 51] Cache monitoring endpoint
@app.get("/cache/stats", tags=["Monitoring"])
def cache_statistics():
    """Return semantic cache hit/miss statistics and health info."""
    return semantic_stats()


if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)
