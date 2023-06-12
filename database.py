import sqlite3
import os
from config import DATABASE_FILE


def create_database():
    if not os.path.exists(DATABASE_FILE):
        conn = sqlite3.connect(DATABASE_FILE)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                audio_file TEXT PRIMARY KEY,
                transcript_text TEXT
            )
        ''')
        conn.close()


def fetch_transcription(audio_file):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT transcript_text FROM transcriptions WHERE audio_file = ?", (audio_file,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def cache_transcription(audio_file, transcript_text):
    conn = sqlite3.connect(DATABASE_FILE)
    conn.execute("INSERT INTO transcriptions (audio_file, transcript_text) VALUES (?, ?)", (audio_file, transcript_text))
    conn.commit()
    conn.close()
