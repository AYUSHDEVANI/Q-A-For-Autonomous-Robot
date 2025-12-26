import os
import hashlib
import logging
import re
import pickle
import io
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.db_service import (
    get_pdf_hashes_from_db,
    update_pdf_in_db,
    clear_pdfs_in_db,
    has_index_in_db,
    update_index_status_in_db
)
from app.services.supabase_client import supabase

logger = logging.getLogger(__name__)

PDF_FOLDER = 'pdfs' # Local Folder fallback
INDEX_FOLDER = 'index'

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Initialize Models
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    logger.warning("HUGGINGFACEHUB_API_TOKEN not found.")

from langchain_huggingface import HuggingFaceEndpointEmbeddings

if hf_token:
    embedder = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )
else:
    logger.error("API Token missing for API-based embeddings!")
    embedder = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token="MISSING_TOKEN", 
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction"
    )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=70)

vectorstore = None
retriever = None

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lab-bot")
SUPABASE_BUCKET = "pdfs"

def get_retriever():
    global retriever
    return retriever

def get_bytes_hash(file_bytes):
    sha256 = hashlib.sha256()
    sha256.update(file_bytes)
    return sha256.hexdigest()

def check_and_process_pdfs():
    global vectorstore, retriever
    logger.info("Starting checks for PDF changes...")

    current_pdfs = {}
    pdf_contents = {} # In-memory storage of PDF bytes: filename -> bytes

    if supabase:
        logger.info("Checking Supabase Storage...")
        try:
            files = supabase.storage.from_(SUPABASE_BUCKET).list()

            for f in files:
                if f['name'].endswith('.pdf'):
                    # We download meta first? List gives basic info.
                    # We need content to hash? Or rely on metadata?
                    # Ideally hash is consistent. Let's download content to be safe and consistent with previous logic.
                    # Note: downloading all PDFs to memory on every startup is heavy.
                    # Optimization: Store hash in DB, check metadata 'updated_at' or ETag?
                    # For now, to keep logic identical: download (or stream).
                    
                    data = supabase.storage.from_(SUPABASE_BUCKET).download(f['name'])
                    file_hash = get_bytes_hash(data)
                    current_pdfs[f['name']] = file_hash
                    pdf_contents[f['name']] = data
        except Exception as e:
            logger.error(f"Supabase Storage Error: {e}")
            return False, f"Supabase Storage Error: {e}"
    else:
        # Fallback to local
        logger.info("Supabase client not active. Checking local folder...")
        for pdf_file in [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            with open(pdf_path, 'rb') as f:
                data = f.read()
            file_hash = get_bytes_hash(data)
            current_pdfs[pdf_file] = file_hash
            pdf_contents[pdf_file] = data

    logger.info(f"Scan complete. Found {len(current_pdfs)} PDFs.")

    stored_pdfs = get_pdf_hashes_from_db()
    
    use_pinecone = bool(os.getenv('PINECONE_API_KEY'))
    
    if use_pinecone:
        logger.info("PINECONE_API_KEY found. Using Pinecone.")
        try:
            vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embedder)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            
            if has_index_in_db() and set(current_pdfs.keys()) == set(stored_pdfs.keys()) and all(
                current_pdfs[f] == stored_pdfs[f] for f in current_pdfs
            ):
                 logger.info("Pinecone index up-to-date.")
                 return True, "Loaded existing Pinecone index."
            
            logger.info("PDF changes detected. Updating Pinecone index...")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            return False, f"Pinecone Error: {e}"

    elif has_index_in_db() and set(current_pdfs.keys()) == set(stored_pdfs.keys()) and all(
        current_pdfs[f] == stored_pdfs[f] for f in current_pdfs
    ):
        # FAISS existing loading logic
        index_file_path = os.path.join(INDEX_FOLDER, "index.faiss")
        if os.path.exists(index_file_path):
             try:
                vectorstore = FAISS.load_local(INDEX_FOLDER, embedder, allow_dangerous_deserialization=True)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
                return True, "Loaded existing FAISS index."
             except Exception as e:
                logger.error(f"Failed to load FAISS: {e}")
        else:
            logger.warning("FAISS index missing.")

    if not current_pdfs:
        logger.warning("No PDFs found.")
        return False, "No PDFs found"

    # Extract text from PDF bytes
    page_texts = []
    for pdf_file, content in pdf_contents.items():
        try:
            # pdfplumber can open file-like objects
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ''
                    page_texts.append((pdf_file, page_num, page_text))
        except Exception as e:
            logger.exception(f"Error reading PDF {pdf_file}")

    text = ''.join([pt for _, _, pt in page_texts])
    text = re.sub(r'\\n\\s*\\n', '\\n\\n', text.strip())
    text = re.sub(r' {2,}', ' ', text)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        logger.error("No text chunks extracted")
        return False, "Failed to extract text"

    # Metadata mapping (Reuse existing logic)
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
        
        if use_pinecone:
            vectorstore = PineconeVectorStore.from_documents(
                documents, 
                embedder, 
                index_name=PINECONE_INDEX_NAME
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            logger.info("Uploaded documents to Pinecone.")
        else:
            vectorstore = FAISS.from_documents(documents, embedder)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            vectorstore.save_local(INDEX_FOLDER)

        # Clear/Update DB
        clear_pdfs_in_db()
        for filename, file_hash in current_pdfs.items():
            update_pdf_in_db(filename, file_hash)
        update_index_status_in_db()

        return True, "PDFs processed successfully"

    except Exception as e:
        logger.exception("Error processing PDFs")
        return False, f"Error processing PDFs: {str(e)}"
