import os
import logging
from sqlalchemy import create_engine, Column, String, Integer, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Use DATABASE_URL if available (Supabase/Render Postgres), else local sqlite
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    # Ensure instance dir exists for local sqlite
    os.makedirs('instance', exist_ok=True)
    DATABASE_URL = 'sqlite:///instance/robot_qa.db'
    logger.info("Using Local SQLite Database")
else:
    logger.info("Using Remote Database")

Base = declarative_base()

class PDFFile(Base):
    __tablename__ = 'pdfs'
    filename = Column(String, primary_key=True)
    hash = Column(String, nullable=False)
    last_processed = Column(DateTime, default=func.now())

class IndexStatus(Base):
    __tablename__ = 'index_status'
    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now())

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Service Functions
def get_pdf_hashes_from_db():
    db = SessionLocal()
    try:
        pdfs = db.query(PDFFile).all()
        return {pdf.filename: pdf.hash for pdf in pdfs}
    finally:
        db.close()

def update_pdf_in_db(filename, file_hash):
    db = SessionLocal()
    try:
        pdf = db.query(PDFFile).filter(PDFFile.filename == filename).first()
        if pdf:
            pdf.hash = file_hash
            pdf.last_processed = func.now()
        else:
            pdf = PDFFile(filename=filename, hash=file_hash)
            db.add(pdf)
        db.commit()
    finally:
        db.close()

def clear_pdfs_in_db():
    # Only clear if absolutely necessary. With Supabase, we might want to sync instead of wipe?
    # Keeping original logic: wipe all.
    db = SessionLocal()
    try:
        db.query(PDFFile).delete()
        db.commit()
    finally:
        db.close()

def has_index_in_db():
    db = SessionLocal()
    try:
        count = db.query(IndexStatus).count()
        return count > 0
    finally:
        db.close()

def update_index_status_in_db():
    db = SessionLocal()
    try:
        status = db.query(IndexStatus).filter(IndexStatus.id == 1).first()
        if status:
            status.last_updated = func.now()
        else:
            status = IndexStatus(id=1)
            db.add(status)
        db.commit()
    finally:
        db.close()
