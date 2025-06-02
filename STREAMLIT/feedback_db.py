import sqlite3
from datetime import datetime

DB_PATH = 'feedback.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            feedback_type TEXT,
            details TEXT,
            text_analyzed TEXT,
            prediction_result TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_feedback(feedback_type, details, text_analyzed, prediction_result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (timestamp, feedback_type, details, text_analyzed, prediction_result)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), feedback_type, details, text_analyzed, prediction_result))
    conn.commit()
    conn.close() 