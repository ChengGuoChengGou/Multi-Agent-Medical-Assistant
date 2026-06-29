## Interview-Ready Agentic RAG Refactor

This fork has been extended from a demo-style multi-agent medical assistant into an interview-ready Agentic RAG project. The refactor is inspired by the engineering structure of Ragent and focuses on explainability, evaluation, reliability, and operational traceability.

### What Was Added

- **Medical intent taxonomy**: 15 medical/user-safety intents with deterministic routing before LLM fallback.
- **Agentic RAG retrieval**: query planning, follow-up completion, sub-question split, query expansion, multi-channel retrieval, reranking, confidence scoring, and ReAct-style retrieval self-correction.
- **Active clarification**: ambiguous questions such as "is it serious?" without context are routed to a clarification response before generation.
- **Conversation memory**: file-backed sliding window plus extractive summary compression under `data/session_memory/`.
- **MCP-like tool registry**: `knowledge_base`, `web_search`, vision tools, and safety policy are registered by tool id for explainable routing traces.
- **Traceability**: `route_decision`, `agent_trace`, `retrieval_trace`, `react_trace`, and `ingestion_trace`.
- **Model reliability**: `ModelGateway` with retry, three-state circuit breaker, token estimation, prompt metadata, and FIFO concurrency limiting.
- **Prompt governance**: versioned prompt files under `prompts/` with `manifest.json` active-version control.
- **Evaluation**: seed intent/tool-call evaluation plus runnable FAQ retrieval metrics.

### Offline Demo

Run the walkthrough without API keys, Qdrant, Docling, or model calls:

```bash
python demo/offline_agentic_walkthrough.py
```

It demonstrates local medical knowledge routing, latest-research routing, vision-agent routing, ReAct retrieval self-correction, and unsafe prescription blocking.

For full JSON traces:

```bash
python demo/offline_agentic_walkthrough.py --json
```

### Evaluation Commands

```bash
python evals/run_eval.py --task intent
python evals/run_eval.py --task tool
python evals/run_faq_retrieval_eval.py --mode keyword
python evals/test_metrics.py
python evals/test_model_gateway.py
python evals/test_prompt_registry.py
python evals/test_rate_limiter.py
python evals/test_react_controller.py
python evals/test_query_planner.py
python evals/test_conversation_memory.py
python evals/test_tool_registry.py
```

Or run the full local non-network check suite:

```bash
python scripts/run_local_checks.py
```

Current seed baseline:

- Intent accuracy: `50/50 = 100%`
- Agent accuracy: `50/50 = 100%`
- Tool accuracy: `50/50 = 100%`
- Tool-call seed eval: `25/25 = 100%`
- FAQ retrieval seed eval: `50` samples, keyword Recall@1 `81.00%`, keyword Recall@5 `96.00%`, full RAG Recall@1 `80.00%`, full RAG Recall@5 `96.00%`, full RAG MRR `0.941`
- Knowledge inventory: `4` indexed PDFs, `12` candidate PDFs, `150` FAQ markdown entries, `131` currently indexed Qdrant/docstore chunks
- Ragent-inspired local checks: query planner, conversation memory, and tool registry all pass.

Knowledge inventory:

```bash
python scripts/knowledge_inventory.py --json-output data/knowledge_inventory.json
```

### Refactored Architecture

```mermaid
flowchart TD
    U["User Query / Image"] --> G["Input Guardrails"]
    G --> R["Medical Intent Router"]
    R --> C["Conversation Agent"]
    R --> K["RAG Agent"]
    R --> W["Web Search Agent"]
    R --> V["Vision Agents"]
    R --> S["Safety Block"]

    K --> QP["Query Planner / Clarification / Split"]
    QP --> QE["Query Expansion"]
    QE --> RC["ReAct Retrieval Controller"]
    RC --> MCR["Multi-Channel Retrieval"]
    MCR --> FQ["FAQ Keyword Channel"]
    MCR --> PP["Dedup + FAQ Priority Boost + Rerank + Confidence"]
    PP --> RG["Grounded Response Generation"]
    RG --> OG["Output Guardrails"]

    DOC["Uploaded Documents"] --> IP["Ingestion Pipeline Trace"]
    IP --> VS["Qdrant Vector Store + Docstore"]
    VS --> MCR

    MG["ModelGateway"] --> CB["Circuit Breaker"]
    MG --> RL["FIFO Rate Limiter"]
    MG --> PR["Prompt Registry"]
    MEM["Session Memory Summary + Window"] --> R
    TR["MCP-like Tool Registry"] --> R
    QE --> MG
    RG --> MG
```

### Engineering Notes

Detailed design notes are in:

- [`docs/business_scenario.md`](docs/business_scenario.md)
- [`docs/intent_taxonomy.md`](docs/intent_taxonomy.md)
- [`docs/agentic_rag_react.md`](docs/agentic_rag_react.md)
- [`docs/multi_channel_retrieval.md`](docs/multi_channel_retrieval.md)
- [`docs/model_reliability.md`](docs/model_reliability.md)
- [`docs/concurrency_control.md`](docs/concurrency_control.md)
- [`docs/ingestion_pipeline.md`](docs/ingestion_pipeline.md)
- [`docs/prompt_management.md`](docs/prompt_management.md)
- [`docs/evaluation_system.md`](docs/evaluation_system.md)
- [`docs/faq_dataset.md`](docs/faq_dataset.md)
- [`docs/engineering_cleanup.md`](docs/engineering_cleanup.md)
- [`docs/refactor_summary.md`](docs/refactor_summary.md)
- [`docs/ragent_comparison.md`](docs/ragent_comparison.md)
- [`docs/openai_compatible_setup.md`](docs/openai_compatible_setup.md)

----

<div align="center">
 
![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>⚕️ Multi-Agent-Medical-Assistant :<h6 align="center">AI-powered multi-agentic system for medical diagnosis and assistance</h6></strong></h1>

<!-- ![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0+-red?style=for-the-badge&logo=pytorch) -->
![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangGraph - Version](https://img.shields.io/badge/LangGraph-0.3+-teal?style=for-the-badge&logo=langgraph)
![LangChain - Version](https://img.shields.io/badge/LangChain-0.3+-teal?style=for-the-badge&logo=langchain)
![Qdrant Client - Version](https://img.shields.io/badge/Qdrant-1.13+-red?style=for-the-badge&logo=qdrant)
![Pydantic - Version](https://img.shields.io/badge/Pydantic-2.10+-red?style=for-the-badge&logo=pydantic)
![FastAPI - Version](https://img.shields.io/badge/FastAPI-0.115+-teal?style=for-the-badge&logo=fastapi)
![Docling - Version](https://img.shields.io/badge/Docling-3.1+-orange?style=for-the-badge&logo=docling)
[![Generic badge](https://img.shields.io/badge/License-Apache-<COLOR>.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/LICENSE) 
[![GitHub Issues](https://img.shields.io/github/issues/souvikmajumder26/Multi-Agent-Medical-Assistant.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)

</div>

----

> [!IMPORTANT]  
> 📋 Version Updates from v2.0 to v2.1 and further:
> 1. **Document Processing Upgrade**: Unstructured.io has been replaced with Docling for document parsing and extraction of text, tables, and images to be embedded.
> 2. **Enhanced RAG References**: Links to source documents and reference images present in reranked retrieved chunks stored in local storage are added to the bottom of the RAG responses.
>
> To use Unstructured.io based solution, refer release - [v2.0](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/tree/v2.0).
 
## 📚 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Technical Flow Chart](#technical-flowchart)
- [Key Features](#key-features)
- [Tech Stack](#technology-stack)
- [Installation and Setup](#installation-setup)
  - [Using Docker](#docker-setup)
  - [Manual Installation](#manual-setup)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)
- [Citing](#citing)
- [Contact](#contact)

----

## 📌 Overview <a name="overview"></a>

The **Multi-Agent Medical Assistant** is an **AI-powered chatbot** designed to assist with **medical diagnosis, research, and patient interactions**.  

🚀 **Powered by Multi-Agent Intelligence**, this system integrates:  
- **🤖 Large Language Models (LLMs)**  
- **🖼️ Computer Vision Models** for medical imaging analysis  
- **📚 Retrieval-Augmented Generation (RAG)** leveraging vector databases  
- **🌐 Real-time Web Search** for up-to-date medical insights  
- **👨‍⚕️ Human-in-the-Loop Validation** to verify AI-based medical image diagnoses  

### **What You’ll Learn from This Project** 📖  
🔹 **👨‍💻 Multi-Agent Orchestration** with structured graph workflows  
🔹 **🔍 Advanced RAG Techniques** – hybrid retrieval, semantic chunking, and vector search  
🔹 **⚡ Confidence-Based Routing** & **Agent-to-Agent Handoff**  
🔹 **🔒 Scalable, Production-Ready AI with Modularized Code & Robust Exception Handling**  

📂 **For learners**: Check out [`agents/README.md`](agents/README.md) for a **detailed breakdown** of the agentic workflow! 🎯  

<!-- The **Multi-Agent Medical Assistant** is an advanced AI-powered chatbot system designed to assist in medical diagnosis, research, and patient interactions.

Using a **Multi-Agentic framework**, this assistant integrates **Large Language Models, Computer Vision Models, Retrieval Augmented Generation leveraging Vector Database**, and **Web Search** to provide **accurate**, **reliable**, and **up-to-date** medical insights.

This project serves as a **comprehensive resource** for learning and implementing **GenAI with multi-agent orchestration**. It demonstrates **advanced Retrieval-Augmented Generation (RAG)**, integrating **vector search with hybrid retrieval techniques**, **efficient chunking strategies respecting document semantic boundaries**, **confidence-based routing**, **agent-to-agent handoff**, **real-time web search capabilities**, **human-in-the-loop validation** and more. The system incorporates **specialized AI agents** for medical reasoning, diagnosis, and research retrieval, all working seamlessly through a structured graph workflow. Developers will gain insights into **modular agentic AI solution design, and robust exception handling** — ensuring **scalability and production readiness**.

For learners: Refer `agents/README.md` for detailed explanation of the agentic workflow. -->

---

## 💫 Demo <a name="demo"></a>


https://github.com/user-attachments/assets/d27d4a2e-1c7d-45e2-bbc5-b3d95ccd5b35


If you like what you see and would want to support the project's developer, you can <a href="https://www.buymeacoffee.com/souvikmajumder" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a> ! :)

📂 **For an even more detailed demo video**: Check out [`Multi-Agent-Medical-Assistant-v1.9`](assets/Multi-Agent-Medical-Assistant-v1.9_Compressed.mp4). 📽️

---

## 🛡️ Technical Flow Chart  <a name="technical-flowchart"></a>

![Technical Flow Chart](assets/final_medical_assistant_flowchart_light_rounded.png)

---

<!-- ## 🌟 Key Features  <a name="key-features"></a>
✅ **Multi-Agent System** – Separate agents handle different tasks (diagnosis, retrieval, reasoning, etc.).  
✅ **RAG-based Retrieval** – Uses Qdrant for vector search & hybrid retrieval techniques.  
✅ **Medical Image Analysis** – Supports **brain tumor segmentation, chest X-ray disease detection, and skin lesion classification**.  
✅ **Web Search Agent** – Fetches the latest medical research when required.  
✅ **Confidence Score Check** – Ensures high accuracy with log probability-based verification.  
✅ **Speech-to-Text & Text-to-Speech** – Uses **Eleven Labs API** for voice interactions.  
✅ **Human-in-the-Loop Verification** – Medical professionals validate the AI’s results before final output.  
✅ **Intuitive UI** – Built for seamless user experience.  

---

## 🛠️ Tech Stack  <a name="tech-stack"></a>
🔹 **Backend**: FastAPI 🚀  
🔹 **Multi-Agent Orchestration**: LangGraph + LangChain 🤖  
🔹 **Vector Database**: Qdrant (for retrieval-augmented generation) 🔍  
🔹 **Medical Image Analysis**: Computer vision models (Brain Tumor - Semantic Segmentation, Chest X-ray - Object Detection, Skin Lesion - Classification) 🏥  
🔹 **Speech Processing**: Eleven Labs API 🎙️  
🔹 **UI**: HTML, CSS, JS 🌐  
🔹 **Deployment**: Docker 🛠️   -->

## ✨ Key Features  <a name="key-features"></a>

- 🤖 **Multi-Agent Architecture** : Specialized agents working in harmony to handle diagnosis, information retrieval, reasoning, and more

- 🔍 **Advanced Agentic RAG Retrieval System** :

  - Docling based parsing to extract text, tables, and images from PDFs.
  - Embedding markdown formatted text, tables and LLM based image summaries.
  - LLM based semantic chunking with structural boundary awareness.
  - LLM based query expansion with related medical domain terms.
  - Qdrant hybrid search combining BM25 sparse keyword search along with dense embedding vector search.
  - HuggingFace Cross-Encoder based reranking of retrieved document chunks for accurate LLM reponses.
  - Input-output guardrails to ensure safe and relevant responses.
  - Links to source documents and images present in reference document chunks provided with reponse.
  - Confidence-based agent-to-agent handoff between RAG and Web Search to prevent hallucinations.

- 🏥 **Medical Imaging Analysis**  
  - Brain Tumor Detection (TBD)
  - Chest X-ray Disease Classification
  - Skin Lesion Segmentation

- 🌐 **Real-time Research Integration** : Web search agent that retrieves the latest medical research papers and findings

- 📊 **Confidence-Based Verification** : Log probability analysis ensures high accuracy in medical recommendations

- 🎙️ **Voice Interaction Capabilities** : Seamless speech-to-text and text-to-speech powered by Eleven Labs API

- 👩‍⚕️ **Expert Oversight System** : Human-in-the-loop verification by medical professionals before finalizing outputs

- ⚔️ **Input & Output Guardrails** : Ensures safe, unbiased, and reliable medical responses while filtering out harmful or misleading content

- 💻 **Intuitive User Interface** : Designed for healthcare professionals with minimal technical expertise

> [!NOTE]  
> Upcoming features:
> 1. Brain Tumor Medical Computer Vision model integration.
> 2. Open to suggestions and contributions.

---

## 🛠️ Technology Stack  <a name="technology-stack"></a>

| Component | Technologies |
|-----------|-------------|
| 🔹 **Backend Framework** | FastAPI |
| 🔹 **Agent Orchestration** | LangGraph |
| 🔹 **Document Parsing** | Docling |
| 🔹 **Knowledge Storage** | Qdrant Vector Database |
| 🔹 **Medical Imaging** | Computer Vision Models |
| | • Brain Tumor: Object Detection (PyTorch) |
| | • Chest X-ray: Image Classification (PyTorch) |
| | • Skin Lesion: Semantic Segmentation (PyTorch) |
| 🔹 **Guardrails** | LangChain |
| 🔹 **Speech Processing** | Eleven Labs API |
| 🔹 **Frontend** | HTML, CSS, JavaScript |
| 🔹 **Deployment** | Docker, GitHub Actions CI/CD |

---

## 🚀 Installation & Setup  <a name="installation-setup"></a>

## 📌 Option 1: Using Docker  <a name="docker-setup"></a>

### Prerequisites:

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- API keys for the required services

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant.git
cd Multi-Agent-Medical-Assistant
```

### 2️⃣ Create Environment File
- Create a `.env` file in the root directory and add the following API keys:

> [!NOTE]  
> You may use any llm and embedding model of your choice...
> 1. If using Azure OpenAI, no modification required.
> 2. If using direct OpenAI, modify the llm and embedding model definitions in the 'config.py' and provide appropriate env variables.
> 3. If using local models, appropriate code changes might be required throughout the codebase especially in 'agents'.

> [!WARNING]  
> Ensure the API keys in the `.env` file are correct and have the necessary permissions.
> No trailing whitespaces after variable names.

```bash
# LLM Configuration (Azure Open AI - gpt-4o used in development)
# If using any other LLM API key or local LLM, appropriate code modification is required
deployment_name= 
model_name=gpt-4o
azure_endpoint=
openai_api_key=
openai_api_version=

# Embedding Model Configuration (Azure Open AI - text-embedding-ada-002 used in development)
# If using any other embedding model, appropriate code modification is required
embedding_deployment_name=
embedding_model_name=text-embedding-ada-002
embedding_azure_endpoint=
embedding_openai_api_key=
embedding_openai_api_version=

# Speech API Key (Free credits available with new Eleven Labs Account)
ELEVEN_LABS_API_KEY=

# Web Search API Key (Free credits available with new Tavily Account)
TAVILY_API_KEY=

# Hugging Face Token - using reranker model "ms-marco-TinyBERT-L-6"
HUGGINGFACE_TOKEN=

# (OPTIONAL) If using Qdrant server version, local does not require API key
QDRANT_URL=
QDRANT_API_KEY=
```

### 3️⃣ Build the Docker Image
```bash
docker build -t medical-assistant .
```

### 4️⃣ Run the Docker Container
```bash
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env medical-assistant
```
The application will be available at: [http://localhost:8000](http://localhost:8000)

### 5️⃣ Ingest Data into Vector DB from Docker Container

- To ingest a single document:
```bash
docker exec medical-assistant-app python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```

- To ingest multiple documents from a directory:
```bash
docker exec medical-assistant-app python ingest_rag_data.py --dir ./data/raw
```

### Managing the Container:

#### Stop the Container
```bash
docker stop medical-assistant-app
```

#### Start the Container
```bash
docker start medical-assistant-app
```

#### View Logs
```bash
docker logs medical-assistant-app
```

#### Remove the Container
```bash
docker rm medical-assistant-app
```

### Troubleshooting:

#### Container Health Check
The container includes a health check that monitors the application status. You can check the health status with:
```bash
docker inspect --format='{{.State.Health.Status}}' medical-assistant-app
```

#### Container Not Starting
If the container fails to start, check the logs for errors:
```bash
docker logs medical-assistant-app
```


## 📌 Option 2: Without Using Docker  <a name="manual-setup"></a>

### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant.git  
cd Multi-Agent-Medical-Assistant  
```

### 2️⃣ Create & Activate Virtual Environment  
- If using conda:
```bash
conda create --name <environment-name> python=3.11
conda activate <environment-name>
```
- If using python venv:
```bash
python -m venv <environment-name>
source <environment-name>/bin/activate  # For Mac/Linux
<environment-name>\Scripts\activate     # For Windows  
```

### 3️⃣ Install Dependencies  

> [!IMPORTANT]  
> ffmpeg is required for speech service to work.

- If using conda:
```bash
conda install -c conda-forge ffmpeg
```
```bash
pip install -r requirements.txt  
```
- If using python venv:
```bash
winget install ffmpeg
```
```bash
pip install -r requirements.txt  
```

### 4️⃣ Set Up API Keys  
- Create a `.env` file and add the required API keys as shown in `Option 1`.

### 5️⃣ Run the Application  
- Run the following command in the activate environment.

```bash
python app.py
```
The application will be available at: [http://localhost:8000](http://localhost:8000)

### 6️⃣ Ingest additional data into the Vector DB
Run any one of the following commands as required.
- To ingest one document at a time:
```bash
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```
- To ingest multiple documents from a directory:
```bash
python ingest_rag_data.py --dir ./data/raw
```

---

## 🧠 Usage  <a name="usage"></a>

> [!NOTE]
> 1. The first run can be jittery and may get errors - be patient and check the console for ongoing downloads and installations.
> 2. On the first run, many models will be downloaded - yolo for tesseract ocr, computer vision agent models, cross-encoder reranker model, etc.
> 3. Once they are completed, retry. Everything should work seamlessly since all of it is thoroughly tested.

- Upload medical images for **AI-based diagnosis**. Task specific Computer Vision model powered agents - upload images from 'sample_images' folder to try out.
- Ask medical queries to leverage **retrieval-augmented generation (RAG)** if information in memory or **web-search** to retrieve latest information.  
- Use **voice-based** interaction (speech-to-text and text-to-speech).  
- Review AI-generated insights with **human-in-the-loop verification**.  

---

## 🤝 Contributions  <a name="contributions"></a>
Contributions are welcome! Please check the [issues](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/issues) tab for feature requests and improvements.  

---

## ⚖️ License  <a name="license"></a>
This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.  

---

## 📝 Citing <a name="citing"></a>
```
@misc{Souvik2025,
  Author = {Souvik Majumder},
  Title = {Multi Agent Medical Assistant},
  Year = {2025},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant}}
}
```

---

## 📬 Contact  <a name="contact"></a>
For any questions or collaboration inquiries, reach out to **Souvik Majumder** on:  

🔗 **LinkedIn**: [https://www.linkedin.com/in/souvikmajumder26](https://www.linkedin.com/in/souvikmajumder26)

🔗 **GitHub**: [https://github.com/souvikmajumder26](https://github.com/souvikmajumder26)

<p align="right">
 <a href="#top"><b>🔝 Return </b></a>
</p>

---
