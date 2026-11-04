import sqlite3
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events (ts TEXT, event TEXT, name TEXT, score REAL, note TEXT)"
        )
        self.db.commit()

    def log(self, event, name, score, note=""):
        self.db.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), event, name, float(score), note),
        )
        self.db.commit()

    def close(self):
        self.db.close()
