from flask import Flask, request, render_template, redirect, url_for, flash, session, send_from_directory
import os
import sqlite3
import pickle
import uuid
import hashlib
import logging
import re
import numpy as np
import pdfplumber
from gtts import gTTS
from dotenv import load_dotenv
from cachetools import TTLCache
from collections import defaultdict
from typing import TypedDict
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ddgs import DDGS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key')

PDF_FOLDER = 'pdfs'
INDEX_FOLDER = 'index'
AUDIO_FOLDER = 'static/audio'

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

load_dotenv()
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Models & Globals
# ---------------------------------------------------------
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=70)

GROQ_MODEL = "llama-3.1-8b-instant"
WHISPER_MODEL = "whisper-large-v3"
question_cache = TTLCache(maxsize=100, ttl=3600)

vectorstore = None
workflow = None
llm = None
retriever = None

DB_PATH = 'instance/robot_qa.db'

# ---------------------------------------------------------
# Database setup
# ---------------------------------------------------------
def init_db():
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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT filename, hash FROM pdfs')
    pdf_hashes = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return pdf_hashes


def update_pdf_in_db(filename, file_hash):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO pdfs (filename, hash) VALUES (?, ?)', (filename, file_hash))
    conn.commit()
    conn.close()


def clear_pdfs_in_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM pdfs')
    conn.commit()
    conn.close()


def has_index_in_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM index_status')
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def update_index_status_in_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO index_status (id, last_updated) VALUES (1, CURRENT_TIMESTAMP)')
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------
def get_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_and_process_pdfs():
    global vectorstore, retriever
    logger.debug("Starting PDF processing...")

    current_pdfs = {}
    for pdf_file in [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        file_hash = get_file_hash(pdf_path)
        current_pdfs[pdf_file] = file_hash
    logger.debug(f"Current PDFs found: {list(current_pdfs.keys())}")

    stored_pdfs = get_pdf_hashes_from_db()
    logger.debug(f"Stored PDFs in DB: {list(stored_pdfs.keys())}")

    # Load existing FAISS index if all PDFs unchanged
    if has_index_in_db() and set(current_pdfs.keys()) == set(stored_pdfs.keys()) and all(
        current_pdfs[f] == stored_pdfs[f] for f in current_pdfs
    ):
        try:
            vectorstore = FAISS.load_local(INDEX_FOLDER, embedder, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            logger.debug("Loaded existing FAISS index successfully.")
            return True, "Loaded existing index."
        except Exception as e:
            logger.exception("Failed to load existing index")

    if not current_pdfs:
        logger.warning("No PDFs found in folder")
        return False, "No PDFs found"

    # Extract text from PDFs
    page_texts = []
    for pdf_file in current_pdfs:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ''
                    page_texts.append((pdf_file, page_num, page_text))
        except Exception as e:
            logger.exception(f"Error reading PDF {pdf_file}")

    text = ''.join([pt for _, _, pt in page_texts])
    text = re.sub(r'\n\s*\n', '\n\n', text.strip())  # Normalize newlines
    text = re.sub(r' {2,}', ' ', text)  # Collapse multiple spaces
    chunks = text_splitter.split_text(text)
    logger.debug(f"Total text chunks extracted: {len(chunks)}")

    if not chunks:
        logger.error("No text chunks extracted from PDFs")
        return False, "Failed to extract text"

    # Build FAISS vectorstore
    try:
        metadata = []
        char_pos = 0
        page_bounds = []
        cum = 0
        for pdf, pn, pt in page_texts:
            start = cum
            cum += len(pt)
            page_bounds.append((start, cum, pdf, pn))

        for chunk in chunks:
            while char_pos < len(text) and text[char_pos:char_pos+len(chunk)] != chunk:
                char_pos += 1
            chunk_start = char_pos
            for start_p, end_p, pdf, pn in page_bounds:
                if start_p <= chunk_start < end_p:
                    metadata.append({'pdf': pdf, 'page': pn})
                    break
            else:
                metadata.append({'pdf': 'unknown', 'page': 1})

        documents = [Document(page_content=chunk, metadata=meta) for chunk, meta in zip(chunks, metadata)]
        vectorstore = FAISS.from_documents(documents, embedder)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        vectorstore.save_local(INDEX_FOLDER)

        with open(os.path.join(INDEX_FOLDER, 'chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)
        with open(os.path.join(INDEX_FOLDER, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        clear_pdfs_in_db()
        for filename, file_hash in current_pdfs.items():
            update_pdf_in_db(filename, file_hash)
        update_index_status_in_db()

        logger.debug("Processed and saved new FAISS index successfully.")
        return True, "PDFs processed successfully"

    except Exception as e:
        logger.exception("Error creating FAISS vectorstore")
        return False, f"Error processing PDFs: {str(e)}"


# ---------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------
class State(TypedDict):
    question: str
    context: str
    relevance_score: float
    answer: str


def pdf_retrieve_node(state: State) -> State:
    logger.debug(f"pdf_retrieve_node input state: {state}")
    print("PDF")
    try:
        docs = retriever.invoke(state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        state["context"] = context
        pairs = [[state["question"], doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)
        state["relevance_score"] = np.mean(scores)
        logger.debug(f"pdf_retrieve_node output state: {state}")
    except Exception as e:
        logger.exception("Error in pdf_retrieve_node")
        state["context"] = ""
        state["relevance_score"] = 0.0
    return state

def web_search_node(state: State) -> State:
    logger.debug(f"web_search_node input state: {state}")
    print("WEB _ SEARCH")
    try:
        with DDGS() as ddgs:
            web_results = [r for r in ddgs.text(f"brief description of {state['question']} project in robotics or engineering", max_results=1)]
            if web_results:
                fallback_context = web_results[0]['body'][:1000]
                state["context"] = f"Note: This project is not currently available in our lab. Here's a brief overview: {fallback_context}"
            else:
                state["context"] = "Note: This project is not currently available in our lab."
        logger.debug(f"web_search_node output state: {state}")
    except Exception as e:
        logger.exception("Error in web_search_node")
        state["context"] = ""
    return state

def generate_node(state: State) -> State:
    logger.debug(f"generate_node input state: {state}")
    try:
        prompt = ChatPromptTemplate.from_template("""
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

        Question: {question}
        Context: {context}
        """)
        formatted_prompt = prompt.format(question=state.get("question", ""), context=state.get("context", ""))
        response = llm.invoke(formatted_prompt)
        state["answer"] = getattr(response, "content", str(response))
        logger.debug(f"generate_node output state: {state}")
    except Exception as e:
        logger.exception("Error in generate_node")
        state["answer"] = f"Error generating answer: {str(e)}"
    return state



# Build the workflow graph
workflow = StateGraph(State)
workflow.set_entry_point("pdf_retrieve")
workflow.add_node("pdf_retrieve", pdf_retrieve_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)
workflow.add_conditional_edges(
    "pdf_retrieve",
    lambda state: "web_search" if state["relevance_score"] < 0.2 else "generate",
    {"web_search": "web_search", "generate": "generate"}
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)
workflow = workflow.compile()

# ---------------------------------------------------------
# Answer generation & Audio synthesis
# ---------------------------------------------------------
def generate_audio(answer_text):
    if not answer_text or answer_text.startswith(("Error", "No PDFs")):
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
    if vectorstore is None:
        return "No PDFs processed. Add PDFs to the 'pdfs' folder and restart the app."

    if question in question_cache:
        return question_cache[question]

    try:
        result = workflow.invoke({"question": question})
        answer = result["answer"]

        # Clean up output
        answer = re.sub(r'\*\*(.*?)\*\*', r'\1', answer).strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()

        question_cache[question] = answer
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}"


# ---------------------------------------------------------
# Flask routes
# ---------------------------------------------------------
@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form.get('question')
        logger.debug(f"Received question: {question}")
        if not question:
            flash("Please enter a question.", "error")
            return redirect(url_for('index'))

        answer = answer_question(question)
        logger.debug(f"Generated answer: {answer}")

        session['question'] = question
        session['answer'] = answer
        audio_filename = generate_audio(answer)
        session['audio_filename'] = audio_filename if audio_filename else None
        logger.debug(f"Generated audio: {audio_filename}")

        flash("Question processed successfully!", "success")
        return redirect(url_for('index'))

    logger.debug(f"Rendering index page: session data {dict(session)}")
    return render_template('index.html', question=session.get('question'), answer=session.get('answer'), audio_filename=session.get('audio_filename'))



@app.route('/ask_audio', methods=['POST'])
def ask_audio():
    from groq import Groq
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))

    if 'audio' not in request.files:
        flash("No audio file received.", "error")
        return redirect(url_for('index'))

    audio = request.files['audio']
    audio_data = audio.read()

    if not audio_data:
        flash("Empty audio file.", "error")
        return redirect(url_for('index'))

    try:
        transcription = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.webm", audio_data, "audio/webm"),
            language="en",  # Key addition: Forces English detection
            response_format="text"  # Optional: Returns plain text
        )
        question = transcription.text.strip()
        if not question:
            flash("No text detected in audio.", "error")
            return redirect(url_for('index'))

        answer = answer_question(question)
        session['question'] = question
        session['answer'] = answer

        audio_filename = generate_audio(answer)
        session['audio_filename'] = audio_filename if audio_filename else None

        flash("Audio question processed successfully!", "success")

    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        flash(f"Error transcribing audio: {str(e)}", "error")

    return redirect(url_for('index'))


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    init_db()
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=GROQ_MODEL, temperature=0.5)
    success, message = check_and_process_pdfs()
    if not success:
        print(f"Warning: {message}")
    app.run(debug=True)
