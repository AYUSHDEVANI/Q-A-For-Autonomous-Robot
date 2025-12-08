from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import faiss
import numpy as np
from groq import Groq
import re
import logging
from cachetools import TTLCache
from dotenv import load_dotenv
from collections import defaultdict
import hashlib
import sqlite3
import pickle
from gtts import gTTS
import uuid
from ddgs import DDGS  # For web search fallback

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')  # Set in .env file
PDF_FOLDER = 'pdfs'
INDEX_FOLDER = 'index'
AUDIO_FOLDER = 'static/audio'
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
load_dotenv()

# Setup
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Groq client
def get_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

GROQ_MODEL = "llama-3.1-8b-instant"
WHISPER_MODEL = "whisper-large-v3"

# Cache for frequent questions (TTL: 1 hour)
question_cache = TTLCache(maxsize=100, ttl=3600)

# Global variables
index = None
chunks = []
metadata = []  # list of {'pdf': pdf_name, 'page': page_num} for each chunk

DB_PATH = 'instance/robot_qa.db'

def init_db():
    """Initialize the database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdfs (
            filename TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS index_status (
            id INTEGER PRIMARY KEY,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_pdf_hashes_from_db():
    """Return a dict mapping PDF filenames to their hashes from the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, hash FROM pdfs')
    pdf_hashes = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return pdf_hashes

def update_pdf_in_db(filename, file_hash):
    """Insert or update a PDF's hash in the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO pdfs (filename, hash) VALUES (?, ?)', (filename, file_hash))
    conn.commit()
    conn.close()

def clear_pdfs_in_db():
    """Remove all PDF records from the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pdfs')
    conn.commit()
    conn.close()

def has_index_in_db():
    """Check if index exists in the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM index_status')
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def update_index_status_in_db():
    """Update index status in DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO index_status (id, last_updated) VALUES (1, CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()

def get_file_hash(file_path):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_index():
    """Load FAISS index, chunks, metadata from files."""
    global index, chunks, metadata
    try:
        index = faiss.read_index(os.path.join(INDEX_FOLDER, 'faiss_index.index'))
        with open(os.path.join(INDEX_FOLDER, 'chunks.pkl'), 'rb') as f:
            chunks = pickle.load(f)
        with open(os.path.join(INDEX_FOLDER, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        return True
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return False

def save_index():
    """Save FAISS index, chunks, metadata to files."""
    faiss.write_index(index, os.path.join(INDEX_FOLDER, 'faiss_index.index'))
    with open(os.path.join(INDEX_FOLDER, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
    with open(os.path.join(INDEX_FOLDER, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

def check_and_process_pdfs():
    """Check if PDFs have changed and process if necessary."""
    global index, chunks, metadata

    # Get current PDFs and their hashes
    current_pdfs = {}
    for pdf_file in [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        file_hash = get_file_hash(pdf_path)
        current_pdfs[pdf_file] = file_hash

    # Get stored PDFs from DB
    stored_pdfs = get_pdf_hashes_from_db()

    # Check if index exists and PDFs match (same files, same hashes, all files exist)
    if has_index_in_db() and set(current_pdfs.keys()) == set(stored_pdfs.keys()) and all(current_pdfs[f] == stored_pdfs[f] for f in current_pdfs) and all(os.path.exists(os.path.join(PDF_FOLDER, f)) for f in stored_pdfs):
        if load_index():
            print("Loaded existing index.")
            return True, "Loaded existing index."
    
    # If not, process PDFs
    if not current_pdfs:
        logger.warning("No PDFs found in 'pdfs' folder.")
        return False, "No PDFs found in the 'pdfs' folder."

    page_texts = []
    for pdf_file, _ in current_pdfs.items():
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ''
                page_texts.append((pdf_file, page_num, page_text))
    
    text = ''.join([pt for _, _, pt in page_texts])
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        logger.error("No text chunks extracted from PDFs.")
        return False, "Failed to extract text from PDFs. Ensure PDFs contain readable text."
    
    # Build page bounds for mapping
    page_bounds = []
    cum = 0
    for pdf, pn, pt in page_texts:
        start = cum
        cum += len(pt)
        page_bounds.append((start, cum, pdf, pn))
    
    # Map chunks to metadata
    metadata = []
    char_pos = 0
    for chunk in chunks:
        while char_pos < len(text) and text[char_pos:char_pos + len(chunk)] != chunk:
            char_pos += 1
        chunk_start = char_pos
        for start_p, end_p, pdf, pn in page_bounds:
            if start_p <= chunk_start < end_p:
                metadata.append({'pdf': pdf, 'page': pn})
                break
        else:
            metadata.append({'pdf': 'unknown', 'page': 1})  # Fallback
    
    # Generate embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    num_embeddings = embeddings.shape[0]
    if num_embeddings < 10:
        index = faiss.IndexFlatL2(dimension)
    else:
        nlist = min(100, max(1, num_embeddings // 2))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        try:
            index.train(embeddings)
        except Exception as e:
            logger.error(f"Failed to train FAISS index: {str(e)}")
            return False, f"Error processing PDFs: {str(e)}"
    index.add(embeddings)
    
    # Save index and update DB
    save_index()
    clear_pdfs_in_db()
    for filename, file_hash in current_pdfs.items():
        update_pdf_in_db(filename, file_hash)
    update_index_status_in_db()
    
    print("Processed and saved new index.")
    return True, "PDFs processed successfully"

def generate_audio(answer_text):
    """Generate audio file from answer text using gTTS."""
    if not answer_text or answer_text.startswith("Error") or answer_text.startswith("No PDFs"):
        return None
    try:
        tts = gTTS(text=answer_text, lang='en')
        audio_filename = f"answer_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        tts.save(audio_path)
        return audio_filename
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

def answer_question(question):
    """Retrieve relevant chunks, re-rank, and generate answer using Groq API."""
    global index, chunks, metadata
    if index is None or not chunks:
        return "No PDFs processed. Add PDFs to the 'pdfs' folder and restart the app."
    
    # Check cache
    if question in question_cache:
        return question_cache[question]
    
    try:
        # Convert question to embedding
        question_embedding = embedder.encode([question], convert_to_numpy=True)
        
        # Search for top 10 chunks
        index.nprobe = 10
        distances, indices = index.search(question_embedding, k=10)
        
        # Re-rank top 10 chunks
        retrieved_chunks = [(chunks[idx], metadata[idx]) for idx in indices[0] if idx < len(chunks)]
        chunk_texts = [chunk for chunk, _ in retrieved_chunks]
        pairs = [[question, chunk] for chunk in chunk_texts]
        scores = cross_encoder.predict(pairs)
        
        # Select top 5 chunks
        ranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)[:5]
        context = ' '.join([chunk for (chunk, _), _ in ranked])
        
        # Fallback to web search if insufficient context (e.g., < 100 chars or no relevant chunks)
        if len(context.strip()) < 100:
            with DDGS() as ddgs:
                web_results = [r for r in ddgs.text(f"brief description of {question} project in robotics or engineering", max_results=1)]
                if web_results:
                    fallback_context = web_results[0]['body'][:1000]  # Truncate to first result body
                    context = f"Note: This project is not currently available in our lab. Here's a brief overview: {fallback_context}"
                else:
                    context = f"Note: This project is not currently available in our lab. Unfortunately, I couldn't find a quick overview right now."
        
        # Call Groq API for answer
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are LabBot, a friendly lab assistant robot designed to help visitors explore exciting engineering and robotics projects. "
                        "Your role is to answer visitor questions about the innovative project ideas, concepts, and technologies documented in the lab's PDF resources. "
                        "Be engaging, enthusiastic, and encouraging—like a knowledgeable guide sparking curiosity about STEM projects. "
                        "Provide clear, concise, and inspiring answers based solely on the provided content from the PDFs. "
                        "If the content indicates a project is not available in the lab, politely note 'This project is not currently available in our lab' and provide a brief general overview if possible. "
                        "Structure your response with these sections, each clearly labeled:\n"
                        "- Definition: Define the concept in one engaging sentence.\n"
                        "- Key Points: List 2-4 key points about the concept in bullet points (use '- ' for each point).\n"
                        "- Example/Relevance: Provide a practical example or explain its relevance to real-world projects or lab experiments in 1-2 sentences.\n"
                        "Only use the provided content; do not add external information unless explicitly noted as a general overview. "
                        "If the content lacks a complete answer, summarize what's available and suggest exploring related lab demos. "
                        "Write naturally with a welcoming, motivational tone to excite visitors about the projects."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContent: {context}"
                }
            ],
            max_tokens=500,
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.8,
            frequency_penalty=0.7
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up answer
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer)
        answer = answer.strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()
        
        # Ensure sections are clearly separated
        sections = []
        current_section = ""
        for line in answer.split('\n'):
            line = line.strip()
            if line.startswith(('Definition:', 'Key Points:', 'Example:', 'Relevance:')):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += '\n' + line
        if current_section:
            sections.append(current_section.strip())
        
        answer = '\n\n'.join(sections)
        
        # Cache the answer
        question_cache[question] = answer
        
        return answer
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio file."""
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        if not question:
            flash("Please enter a question.", "error")
            return redirect(url_for('index'))
        answer = answer_question(question)
        session['question'] = question
        session['answer'] = answer
        # Generate audio for answer
        audio_filename = generate_audio(answer)
        if audio_filename:
            session['audio_filename'] = audio_filename
        else:
            session.pop('audio_filename', None)
        flash("Question processed successfully!", "success")
        return redirect(url_for('index'))
    
    question = session.get('question')
    answer = session.get('answer')
    audio_filename = session.get('audio_filename')
    
    return render_template('index.html', question=question, answer=answer, audio_filename=audio_filename)

@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        flash("No audio file received.", "error")
        return redirect(url_for('index'))
    
    audio = request.files['audio']
    audio_data = audio.read()
    
    if not audio_data:
        flash("Empty audio file.", "error")
        return redirect(url_for('index'))
    
    try:
        client = get_groq_client()
        transcription = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.webm", audio_data, "audio/webm")
        )
        question = transcription.text.strip()
        
        if not question:
            flash("No text detected in audio.", "error")
            return redirect(url_for('index'))
        
        answer = answer_question(question)
        session['question'] = question
        session['answer'] = answer
        # Generate audio for answer
        audio_filename = generate_audio(answer)
        if audio_filename:
            session['audio_filename'] = audio_filename
        else:
            session.pop('audio_filename', None)
        flash("Audio question processed successfully!", "success")
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        flash(f"Error transcribing audio: {str(e)}", "error")
    
    return redirect(url_for('index'))

if __name__ == "__main__":
    init_db()
    success, message = check_and_process_pdfs()
    if not success:
        print(f"Warning: {message}")
    app.run(debug=True)