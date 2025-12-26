# 🤖 LabBot: Intelligent Robot QA Assistant

**LabBot** is a modern, voice-enabled AI assistant designed to answer questions about engineering and robotics projects. It uses **Retrieval Augmented Generation (RAG)** to provide accurate answers based on PDF manuals, powered by cloud-native APIs for maximum efficiency.

![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green) ![AI](https://img.shields.io/badge/AI-Groq%20%2B%20HuggingFace-orange) ![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-green) ![Supabase](https://img.shields.io/badge/Data%20%26%20Storage-Supabase-green)

---

## ✨ Key Features

-   **🗣️ Voice Interaction:** Ask questions via microphone and get spoken responses (Speech-to-Text & Text-to-Speech).
-   **📚 PDF Knowledge Base:** Automatically ingests and indexes PDF documents from **Supabase Storage**.
-   **🌩️ 100% Cloud Architecture:**
    -   **Reasoning:** Groq API (Llama 3.1 8b).
    -   **Embeddings:** Hugging Face Inference API (`all-MiniLM-L6-v2`).
    -   **Vector Search:** Pinecone (Serverless).
    -   **Storage:** Supabase (PostgreSQL & Object Storage).
-   **🎨 Modern UI:** Responsive React interface with real-time text streaming.

---

## 🏗️ Architecture

1.  **Frontend (`frontend/`)**: React + Vite (Static Build served by Backend).
2.  **Backend (`app/`)**: FastAPI server.
3.  **Data Layer**:
    -   **Pinecone**: Stores vector embeddings.
    -   **Supabase Storage**: Stores raw PDF files.
    -   **Supabase PostgreSQL**: Stores file metadata (via SQLAlchemy).

---

## 🚀 Installation & Setup

### Prerequisites
-   **Python 3.8+**
-   **Node.js 16+**
-   **Accounts:** [Groq](https://console.groq.com/), [Hugging Face](https://huggingface.co/), [Pinecone](https://pinecone.io/), [Supabase](https://supabase.com/).

### 1. Environment Variables
Create a `.env` file in the root directory:
```bash
# AI Models
GROQ_API_KEY=your_groq_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token

# Pinecone (Vector DB)
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=lab-bot
# Note: Index Dimension must be 384 (Metric: Cosine)

# Supabase (Data & Storage)
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
DATABASE_URL=postgresql://user:pass@host:port/postgres
# Note: URL-encode password if it has special chars like '@' -> '%40'
```

### 2. Local Setup
```bash
# Backend
python -m venv .venv
# Activate venv
pip install -r requirements.txt

# Frontend
cd frontend
npm install
npm run build 
# (The backend serves the 'dist' folder automatically)
```

---

## ▶️ How to Run

### Development
```bash
# Start Backend
python -m uvicorn app.main:app --reload

# Start Frontend (Separate Terminal)
cd frontend && npm run dev
```

### Cloud Deployment (Render)
This project is configured for **Render** (Web Service).
1.  **Build Command**: `./build.sh`
2.  **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
3.  **Environment Variables**: Add all keys from `.env` to Render Dashboard.

---

## 📝 License
MIT License.