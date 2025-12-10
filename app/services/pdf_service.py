import os
import hashlib
import logging
import re
import pickle
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.db_service import (
    get_pdf_hashes_from_db,
    update_pdf_in_db,
    clear_pdfs_in_db,
    has_index_in_db,
    update_index_status_in_db
)

logger = logging.getLogger(__name__)

PDF_FOLDER = 'pdfs'
INDEX_FOLDER = 'index'

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Initialize Models
# Initialize Models
# Use API-based Embeddings (requires HUGGINGFACEHUB_API_TOKEN)
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    logger.warning("HUGGINGFACEHUB_API_TOKEN not found. Embeddings may fail if not set.")

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
# Note: To strictly use API (no local download), we should use HuggingFaceEndpointEmbeddings
# (renamed from HuggingFaceInferenceAPIEmbeddings in newer versions)
from langchain_huggingface import HuggingFaceEndpointEmbeddings

if hf_token:
    embedder = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )
else:
    # Fallback or Error? User said "API instead of locally".
    # We will assume they will add the token.
    logger.error("API Token missing for API-based embeddings!")
    embedder = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token="MISSING_TOKEN", 
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=70)

# Global Vectorstore/Retriever
vectorstore = None
retriever = None

def get_retriever():
    global retriever
    return retriever

def get_file_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_and_process_pdfs():
    global vectorstore, retriever
    logger.info("Starting checks for PDF changes...")

    current_pdfs = {}
    if not os.path.exists(PDF_FOLDER):
         logger.info(f"Creating PDF folder: {PDF_FOLDER}")
         os.makedirs(PDF_FOLDER)

    for pdf_file in [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        file_hash = get_file_hash(pdf_path)
        current_pdfs[pdf_file] = file_hash
    logger.info(f"Scan complete. Found {len(current_pdfs)} PDFs in directory.")
    logger.debug(f"Current PDF list: {list(current_pdfs.keys())}")

    stored_pdfs = get_pdf_hashes_from_db()
    
    # Load existing FAISS index if all PDFs unchanged
    if has_index_in_db() and set(current_pdfs.keys()) == set(stored_pdfs.keys()) and all(
        current_pdfs[f] == stored_pdfs[f] for f in current_pdfs
    ):
        index_file_path = os.path.join(INDEX_FOLDER, "index.faiss")
        if os.path.exists(index_file_path):
            try:
                vectorstore = FAISS.load_local(INDEX_FOLDER, embedder, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                logger.debug("Loaded existing FAISS index successfully.")
                return True, "Loaded existing index."
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                # Fall through to rebuild
        else:
            logger.warning("Index metadata suggests existing index, but 'index.faiss' file is missing. Rebuilding...")

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
    text = re.sub(r'\\n\\s*\\n', '\\n\\n', text.strip())  # Normalize newlines
    text = re.sub(r' {2,}', ' ', text)  # Collapse multiple spaces
    chunks = text_splitter.split_text(text)
    
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

        return True, "PDFs processed successfully"

    except Exception as e:
        logger.exception("Error creating FAISS vectorstore")
        return False, f"Error processing PDFs: {str(e)}"
