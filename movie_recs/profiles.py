"""Persistent user IDs and rating upserts, using only the standard library."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class UserStore:
    def __init__(self, path: Path, reserved_max_id: int):
        self.path = path
        self.reserved_max_id = reserved_max_id
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (user_id INTEGER PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS ratings (
                    user_id INTEGER NOT NULL REFERENCES users(user_id),
                    movie_id INTEGER NOT NULL,
                    rating REAL NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
                    PRIMARY KEY (user_id, movie_id)
                );
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, ratings: dict[int, float]) -> int:
        with self.connect() as connection:
            # Serialize ID allocation and the initial ratings in one transaction.
            connection.execute("BEGIN IMMEDIATE")
            last_id = connection.execute("SELECT MAX(user_id) FROM users").fetchone()[0] or 0
            user_id = max(self.reserved_max_id, last_id) + 1
            connection.execute("INSERT INTO users VALUES (?)", (user_id,))
            connection.executemany(
                "INSERT INTO ratings VALUES (?, ?, ?)",
                [(user_id, movie_id, rating) for movie_id, rating in ratings.items()],
            )
            return user_id

    def exists(self, user_id: int) -> bool:
        with self.connect() as connection:
            return (
                connection.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone()
                is not None
            )

    def ratings(self, user_id: int) -> dict[int, float]:
        with self.connect() as connection:
            return dict(
                connection.execute(
                    "SELECT movie_id, rating FROM ratings WHERE user_id=? ORDER BY movie_id",
                    (user_id,),
                )
            )

    def rate(self, user_id: int, movie_id: int, rating: float) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO users VALUES (?)", (user_id,))
            connection.execute(
                "INSERT INTO ratings VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, movie_id) DO UPDATE SET rating=excluded.rating",
                (user_id, movie_id, rating),
            )
