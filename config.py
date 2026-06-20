"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class.

Model Registry:
    Per-agent model routing via environment variables:
    - DECISION_MODEL: Agent routing decisions (default: model_name)
    - CONVERSATION_MODEL: General Q&A (default: model_name)
    - RAG_MODEL: RAG retrieval+answering (default: model_name)
    - WEB_SEARCH_MODEL: Web search processing (default: model_name)
    - VISION_MODEL: Medical image analysis (default: model_name)
    - SUMMARIZER_MODEL: Document summarization (default: model_name)
    All fall back to model_name if not set.
"""

import os
import logging
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

class LLMFallbackChain(BaseChatModel):
    """Chat model wrapper with automatic fallback to secondary models on failure.

    Usage: set FALLBACK_MODELS="gpt-4o-mini,deepseek-chat,qwen-plus" in .env
    The primary model comes from model_name (existing). On repeated failures,
    the chain automatically switches to the next model in the list.
    """
    models: list
    active_index: int = 0
    fail_threshold: int = 2  # consecutive failures before switching
    _fail_counts: dict = {}
    _last_switch: float = 0

    class Config:
        arbitrary_types_allowed = True

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_err = None
        for idx in range(self.active_index, len(self.models)):
            model = self.models[idx]
            try:
                result = model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                # Success: reset fail count
                self._fail_counts[idx] = 0
                if idx != self.active_index:
                    logger.info(f"[LLMFallback] Recovered to primary model, switching back")
                    self.active_index = 0
                return result
            except Exception as e:
                last_err = e
                self._fail_counts[idx] = self._fail_counts.get(idx, 0) + 1
                logger.warning(f"[LLMFallback] model={model.model} failed ({self._fail_counts[idx]}): {e}")
                if self._fail_counts[idx] >= self.fail_threshold and idx + 1 < len(self.models):
                    self.active_index = idx + 1
                    logger.warning(f"[LLMFallback] Switching to fallback model: {self.models[idx+1].model}")
        raise last_err

    @property
    def _llm_type(self) -> str:
        return "llm-fallback-chain"

    def bind_tools(self, tools, **kwargs):
        """Delegate bind_tools to the active model."""
        return self.models[self.active_index].bind_tools(tools, **kwargs)

    @property
    def model(self) -> str:
        return self.models[self.active_index].model

# Load environment variables from .env file
load_dotenv()

# ─── Model Registry (env-overridable per-agent routing) ──────────────────────
_DEFAULT_MODEL = os.getenv("model_name", "gpt-4o-mini")
_DEFAULT_API_KEY = os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY")
_DEFAULT_API_BASE = os.getenv("OPENAI_BASE_URL") or os.getenv("openai_base_url")

# Per-role model overrides (env vars); falls back to defaults above
_MODEL_ROLES = {
    "decision":     os.getenv("DECISION_MODEL",     _DEFAULT_MODEL),
    "conversation": os.getenv("CONVERSATION_MODEL", _DEFAULT_MODEL),
    "rag":          os.getenv("RAG_MODEL",           _DEFAULT_MODEL),
    "web_search":   os.getenv("WEB_SEARCH_MODEL",   _DEFAULT_MODEL),
    "vision":       os.getenv("VISION_MODEL",        _DEFAULT_MODEL),
    "summarizer":   os.getenv("SUMMARIZER_MODEL",    _DEFAULT_MODEL),
}


def _make_llm(temperature: float, role: str = "conversation") -> BaseChatModel:
    """Create a ChatOpenAI instance with optional fallback chain.

    Args:
        temperature: Sampling temperature.
        role: Model role key (decision/conversation/rag/web_search/vision/summarizer).
              Uses model_name env var as default; override per-role via DECISION_MODEL etc.
    Set FALLBACK_MODELS="model-a,model-b" in .env to enable automatic failover.
    """
    model = _MODEL_ROLES.get(role, _DEFAULT_MODEL)
    primary = ChatOpenAI(
        model=model,
        openai_api_key=_DEFAULT_API_KEY,
        openai_api_base=_DEFAULT_API_BASE,
        temperature=temperature,
        max_retries=3,
        request_timeout=60,
    )
    # Build fallback chain if FALLBACK_MODELS is set
    fallback_str = os.getenv("FALLBACK_MODELS", "")
    if fallback_str:
        fallback_names = [m.strip() for m in fallback_str.split(",") if m.strip()]
        fallback_models = [
            ChatOpenAI(
                model=fm,
                openai_api_key=_DEFAULT_API_KEY,
                openai_api_base=_DEFAULT_API_BASE,
                temperature=temperature,
                max_retries=3,
                request_timeout=60,
            )
            for fm in fallback_names
        ]
        all_models = [primary] + fallback_models
        if len(all_models) > 1 and role != "conversation":
            logger.info(f"[ModelRegistry] role={role}, primary={model}, fallbacks={fallback_names}")
        return LLMFallbackChain(models=all_models)
    if role != "conversation" and model != _DEFAULT_MODEL:
        logger.info(f"[ModelRegistry] role={role}, model={model}")
    return primary

def _make_embedding() -> OpenAIEmbeddings:
    """Create OpenAI embeddings using OpenAI-compatible API."""
    return OpenAIEmbeddings(
        model=os.getenv("embedding_model_name", "text-embedding-3-large"),
        openai_api_key=os.getenv("embedding_openai_api_key") or os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL") or os.getenv("openai_base_url"),
    )

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = _make_llm(0.1, role="decision")

class ConversationConfig:
    def __init__(self):
        self.llm = _make_llm(0.7, role="conversation")

class WebSearchConfig:
    def __init__(self):
        self.llm = _make_llm(0.3, role="web_search")
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536  # Add the embedding dimension here
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Modify based on documents and performance
        self.chunk_overlap = 50  # Modify based on documents and performance
        self.embedding_model = _make_embedding()
        self.llm = _make_llm(0.3, role="rag")
        self.summarizer_model = _make_llm(0.5, role="summarizer")
        self.chunker_model = _make_llm(0.0, role="rag")
        self.response_generator_model = _make_llm(0.3, role="rag")
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = _make_llm(0.1, role="vision")

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Fallback TTS
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)
        # Edge TTS (free, no API key needed)
        self.edge_tts_voice = os.getenv("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
        self.edge_tts_rate = os.getenv("EDGE_TTS_RATE", "+0%")
        self.edge_tts_pitch = os.getenv("EDGE_TTS_PITCH", "+0Hz")

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class MCPConfig:
    def __init__(self):
        self.biomcp = {
            "enabled": True,
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "biomcp", "stdio"],
            "cwd": "./biomcp",
            "timeout": 30,
        }
        self.autoicd = {
            "enabled": True,
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", "server.py"],
            "cwd": "./autoicd-mcp",
            "timeout": 30,
        }
        self.healthcare = {
            "enabled": True,
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@biantylabs/healthcare-mcp-public", "stdio"],
            "timeout": 30,
        }
        self.max_iterations = 5
        self.tool_timeout = 60

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.mcp = MCPConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history
        self.summarize_conversation_history = True  # Phase 51: Summarize old messages before truncating
        self.summary_keep_recent = 8  # Phase 51: Keep last 8 messages (4 Q&A pairs) when summarizing

# # Example usage
# config = Config()
