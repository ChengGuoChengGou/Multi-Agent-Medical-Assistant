"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from core.local_embeddings import HashingEmbeddings

# Load project-local environment variables from .env file.
# The desktop/global Python environment may already define OpenAI variables;
# this app should prefer the checked-out project's local configuration.
load_dotenv(override=True)


def _model_provider() -> str:
    return os.getenv("MODEL_PROVIDER", "azure").strip().lower()


def create_chat_model(temperature: float):
    return LazyChatModel(temperature=temperature)


def create_embedding_model():
    if os.getenv("USE_LOCAL_HASHING_EMBEDDINGS", "false").strip().lower() == "true":
        return HashingEmbeddings(dimension=int(os.getenv("EMBEDDING_DIM", "1536")))
    return LazyEmbeddingModel()


class LazyChatModel(Runnable):
    def __init__(self, temperature: float):
        self.temperature = temperature
        self._model = None

    def invoke(self, input, config=None, **kwargs):
        return self._get_model().invoke(input, config=config, **kwargs)

    def stream(self, input, config=None, **kwargs):
        return self._get_model().stream(input, config=config, **kwargs)

    def batch(self, inputs, config=None, **kwargs):
        return self._get_model().batch(inputs, config=config, **kwargs)

    def _get_model(self):
        if self._model is not None:
            return self._model

        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        if _model_provider() == "openai_compatible":
            self._model = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL_NAME", os.getenv("model_name", "gpt-4o")),
                api_key=os.getenv("OPENAI_API_KEY", os.getenv("openai_api_key")),
                base_url=os.getenv("OPENAI_BASE_URL"),
                temperature=self.temperature,
            )
        else:
            self._model = AzureChatOpenAI(
                deployment_name=os.getenv("deployment_name"),
                model_name=os.getenv("model_name"),
                azure_endpoint=os.getenv("azure_endpoint"),
                openai_api_key=os.getenv("openai_api_key"),
                openai_api_version=os.getenv("openai_api_version"),
                temperature=self.temperature,
            )
        return self._model


class LazyEmbeddingModel:
    def __init__(self):
        self._model = None

    def embed_documents(self, texts):
        return self._get_model().embed_documents(texts)

    def embed_query(self, text):
        return self._get_model().embed_query(text)

    def _get_model(self):
        if self._model is not None:
            return self._model

        from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

        if _model_provider() == "openai_compatible":
            self._model = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("embedding_model_name", "text-embedding-3-small")),
                api_key=os.getenv("OPENAI_API_KEY", os.getenv("openai_api_key")),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        else:
            self._model = AzureOpenAIEmbeddings(
                deployment=os.getenv("embedding_deployment_name"),
                model=os.getenv("embedding_model_name"),
                azure_endpoint=os.getenv("embedding_azure_endpoint"),
                openai_api_key=os.getenv("embedding_openai_api_key"),
                openai_api_version=os.getenv("embedding_openai_api_version"),
            )
        return self._model


def create_chat_model_eager(temperature: float):
    if _model_provider() == "openai_compatible":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_NAME", os.getenv("model_name", "gpt-4o")),
            api_key=os.getenv("OPENAI_API_KEY", os.getenv("openai_api_key")),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=temperature,
        )

    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        deployment_name=os.getenv("deployment_name"),
        model_name=os.getenv("model_name"),
        azure_endpoint=os.getenv("azure_endpoint"),
        openai_api_key=os.getenv("openai_api_key"),
        openai_api_version=os.getenv("openai_api_version"),
        temperature=temperature,
    )


def create_embedding_model_eager():
    if _model_provider() == "openai_compatible":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("embedding_model_name", "text-embedding-3-small")),
            api_key=os.getenv("OPENAI_API_KEY", os.getenv("openai_api_key")),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        deployment=os.getenv("embedding_deployment_name"),
        model=os.getenv("embedding_model_name"),
        azure_endpoint=os.getenv("embedding_azure_endpoint"),
        openai_api_key=os.getenv("embedding_openai_api_key"),
        openai_api_version=os.getenv("embedding_openai_api_version"),
    )

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = create_chat_model(temperature=0.1)

class ConversationConfig:
    def __init__(self):
        self.llm = create_chat_model(temperature=0.7)

class WebSearchConfig:
    def __init__(self):
        self.llm = create_chat_model(temperature=0.3)
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
        self.embedding_model = create_embedding_model()
        self.llm = create_chat_model(temperature=0.3)
        self.summarizer_model = create_chat_model(temperature=0.5)
        self.chunker_model = create_chat_model(temperature=0.0)
        self.response_generator_model = create_chat_model(temperature=0.3)
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value
        self.react_max_steps = 2  # bounded retrieval self-correction loop
        self.react_timeout_seconds = 8.0

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = create_chat_model(temperature=0.1)

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

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
        self.host = os.getenv("APP_HOST", "0.0.0.0")
        self.port = int(os.getenv("APP_PORT", "8000"))
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

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
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history
        self.memory_storage_dir = os.getenv("MEMORY_STORAGE_DIR", "./data/session_memory")
        self.memory_window_size = int(os.getenv("MEMORY_WINDOW_SIZE", "8"))
        self.memory_summary_trigger = int(os.getenv("MEMORY_SUMMARY_TRIGGER", "10"))

# # Example usage
# config = Config()
