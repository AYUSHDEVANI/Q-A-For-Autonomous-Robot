# 🤖 LabBot: Intelligent Robot QA Assistant

**LabBot** is a modern, voice-enabled AI assistant designed to answer questions about engineering and robotics projects. It uses **Retrieval Augmented Generation (RAG)** to provide accurate answers based on PDF manuals, powered by cloud-native APIs for maximum efficiency.

![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green) ![AI](https://img.shields.io/badge/AI-Groq%20%2B%20HuggingFace-orange)

---

## ✨ Key Features

-   **🗣️ Voice Interaction:** Ask questions via microphone and get spoken responses (Speech-to-Text & Text-to-Speech).
-   **📚 PDF Knowledge Base:** Automatically ingests and indexes PDF documents from the `pdfs/` folder.
-   **🌩️ 100% Cloud AI:**
    -   **Reasoning:** Groq API (Llama 3.1 8b) for instant answers.
    -   **Embeddings:** Hugging Face Inference API for semantic search.
    -   **Transcription:** Groq Whisper (Large-v3).
-   **🧠 Smart Memory:** Remembers the last 2 interactions for contextual follow-up questions.
-   **🎨 Modern UI:** Responsive, "Soft Light" themed React interface with real-time text streaming.

---

## 🏗️ Architecture

The project follows a **Headless Architecture**:

1.  **Frontend (`frontend/`)**: A **React** application (Vite) that handles the UI, audio recording, and state management.
2.  **Backend (`app/`)**: A **FastAPI** server that manages PDF processing (LangChain), vector search (FAISS), and LLM orchestration.

---

## 🚀 Installation

### Prerequisites
-   **Python 3.8+**
-   **Node.js 16+** (for Frontend)
-   **API Keys:**
    -   [Groq API Key](https://console.groq.com/)
    -   [Hugging Face Access Token](https://huggingface.co/settings/tokens)

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/robot-qa-bot.git
cd robot-qa-bot

# Create Virtual Environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Python Dependencies
pip install -r requirements.txt

# Create .env file
# Add your keys:
# GROQ_API_KEY=your_key_here
# HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 2. Frontend Setup
```bash
# Open a NEW terminal
cd frontend

# Install Node Dependencies
npm install
```

---

## ▶️ How to Run

You need to run the **Backend** and **Frontend** in separate terminals.

### Terminal 1: Start Backend (API)
```bash
python run.py
```
*Server will start at `http://localhost:8000` (API only).*

### Terminal 2: Start Frontend (UI)
```bash
cd frontend
npm run dev
```
*Click the link shown (usually `http://localhost:5173`) to open the app.*

---

## 📂 Project Structure

```
├── app/                  # FastAPI Backend
│   ├── services/         # Core Logic (PDF, Chat, Audio)
│   ├── routes.py         # API Endpoints
│   └── main.py           # App Entry Point
├── frontend/             # React Frontend
│   ├── src/              # Components & Styles
│   └── package.json      # Node Dependencies
├── pdfs/                 # Drop your PDF manuals here
├── requirements.txt      # Python Dependencies
└── run.py                # Backend Startup Script
```

## 🛠️ Configuration

-   **PDFs:** Place any `.pdf` file in the `pdfs/` folder. The app will automatically index it on the next restart.
-   **History Limit:** The bot remembers the last 2 turns. To change this, edit `request.session["history"]` in `app/routes.py`.
-   **Theme:** UI styles are defined in `frontend/src/index.css`.

---

## 📝 License
This project is open-source and free to use.