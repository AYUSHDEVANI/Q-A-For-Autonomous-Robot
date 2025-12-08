import os
import sqlite3

DB_PATH = 'instance/robot_qa.db'

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
