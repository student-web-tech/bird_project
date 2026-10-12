
import sqlite3
from pathlib import Path

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "bird_observations.db"


def create_database():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            observation_date TEXT,
            habitat TEXT,
            image_path TEXT,
            confidence REAL,
            data_source TEXT
        )
    """)

    connection.commit()
    connection.close()

    print("Bird observation database created successfully!")


if __name__ == "__main__":
    create_database()
