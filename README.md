# Robot QA Bot

## Project Description
This is a web-based QA bot designed for querying information from robot manuals and documents stored as PDFs in a `pdfs/` folder. The system processes PDFs into a single FAISS vector index for semantic search, using a SQLite database to track PDF hashes and index status. Questions can be submitted via text or audio (transcribed using Groq's Whisper model). The bot retrieves relevant content, generates structured answers (Definition, Key Points, Example/Relevance, Source) and summaries using Groq's LLM, and displays them in a simple Flask web interface. The index is reused if PDFs are unchanged, ensuring efficiency.

## Features
- **PDF Processing**: Automatically processes all PDFs in `pdfs/` into a combined FAISS index. Checks for changes (add/remove/modify) via SHA256 hashes stored in SQLite; reprocesses only when needed.
- **Question Answering**: Accepts text or audio questions, providing structured answers and summaries sourced from PDFs, with citations including PDF names and pages.
- **Audio Input**: Records audio in the browser (WebM format), transcribes it using Groq Whisper (`whisper-large-v3`), and processes it as a question.
- **Caching**: Caches questions and answers for 1 hour to improve performance.
- **Persistent Index**: Saves/loads index, chunks, and metadata to/from `index/` folder; database ensures index aligns with current PDFs.
- **Simple Interface**: Flask-based UI with text/audio input, displaying question, answer, and summary. No user authentication or PDF management.

## Dependencies and Their Roles
The following Python libraries are required (listed in `requirements.txt`):
- **flask**: Web framework for serving the app and handling routes (text/audio question submission).
- **pdfplumber**: Extracts text from PDFs for chunking.
- **langchain**: Provides `RecursiveCharacterTextSplitter` for splitting PDF text into chunks.
- **sentence-transformers**: Generates embeddings (`all-MiniLM-L6-v2`) and re-ranks results (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **faiss-cpu**: Builds and queries vector index for similarity search.
- **groq**: Interfaces with Groq API for LLM answer generation (`llama-3.1-8b-instant`) and audio transcription (`whisper-large-v3`).
- **cachetools**: Implements TTL cache for question-answer pairs.
- **python-dotenv**: Loads `GROQ_API_KEY` from `.env`.
- **hashlib**: Computes SHA256 hashes for PDF change detection.
- **sqlite3**: Manages SQLite database for PDF and index metadata (built-in to Python).
- **pickle**: Serializes/deserializes index data (chunks, metadata).
- **numpy**: Handles array operations for embeddings.
- **re**: Cleans up LLM responses with regex.
- **logging**: Logs errors for debugging.
- **collections**: Uses `defaultdict` for source aggregation.

No additional installations are needed beyond `requirements.txt`. The app uses `faiss-cpu` (CPU-based); for GPU, replace with `faiss-gpu` in `requirements.txt`.

## Installation
1. Clone or download the project files (`app.py`, `index.html`, `requirements.txt`).
2. Create a virtual environment (recommended to isolate dependencies):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/Mac
   venv\Scripts\activate     # Windows