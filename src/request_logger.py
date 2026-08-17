"""
request_logger.py — SQLite database logger and analytics aggregator for pipeline requests.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.config import LOG_DB_PATH
from src.logger import get_logger
from src.models import RequestLog

log = get_logger(__name__)


class RequestLogger:
    """
    SQLite-backed persistence and metrics engine for recording every pipeline request.
    """

    def __init__(self, db_path: Path = LOG_DB_PATH):
        """
        Initialize the RequestLogger with a target SQLite database path.

        Args:
            db_path: Path to SQLite .db file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a SQLite database connection with row dictionary support."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the request_logs table if it does not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    is_hit INTEGER NOT NULL,
                    similarity_score REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    matched_query TEXT
                )
            """)
            conn.commit()
        log.debug(f"RequestLogger initialized with DB at '{self.db_path}'.")

    def log(self, entry: RequestLog) -> int:
        """
        Write a single RequestLog entry to the SQLite database.

        Args:
            entry: RequestLog instance containing request details.

        Returns:
            The inserted database row ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO request_logs (query, response, is_hit, similarity_score, latency_ms, timestamp, matched_query)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.query,
                entry.response,
                1 if entry.is_hit else 0,
                float(entry.similarity_score),
                float(entry.latency_ms),
                entry.timestamp,
                entry.matched_query
            ))
            conn.commit()
            inserted_id = cursor.lastrowid
            log.debug(f"Logged request entry ID {inserted_id} (hit={entry.is_hit}, latency={entry.latency_ms:.2f}ms).")
            return inserted_id

    def get_all(self, limit: Optional[int] = None) -> List[RequestLog]:
        """
        Retrieve log records ordered by timestamp descending.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of RequestLog instances.
        """
        query = "SELECT query, response, is_hit, similarity_score, latency_ms, timestamp, matched_query FROM request_logs ORDER BY id DESC"
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        logs = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(query).fetchall()
            for row in rows:
                logs.append(RequestLog(
                    query=row["query"],
                    response=row["response"],
                    is_hit=bool(row["is_hit"]),
                    similarity_score=float(row["similarity_score"]),
                    latency_ms=float(row["latency_ms"]),
                    timestamp=row["timestamp"],
                    matched_query=row["matched_query"]
                ))
        return logs

    def get_stats(self, cost_per_1k_tokens: float = 0.00015) -> Dict[str, Any]:
        """
        Calculate aggregate KPI metrics for dashboard analytics.

        Args:
            cost_per_1k_tokens: Estimated API cost per 1,000 tokens saved on hit (defaults to ~$0.00015 for gpt-4o-mini).

        Returns:
            Dictionary containing analytics KPIs.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            total_requests = cursor.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
            total_hits = cursor.execute("SELECT COUNT(*) FROM request_logs WHERE is_hit = 1").fetchone()[0]
            total_misses = cursor.execute("SELECT COUNT(*) FROM request_logs WHERE is_hit = 0").fetchone()[0]
            
            hit_rate_pct = (total_hits / total_requests * 100.0) if total_requests > 0 else 0.0

            avg_hit_latency = cursor.execute("SELECT AVG(latency_ms) FROM request_logs WHERE is_hit = 1").fetchone()[0] or 0.0
            avg_miss_latency = cursor.execute("SELECT AVG(latency_ms) FROM request_logs WHERE is_hit = 0").fetchone()[0] or 0.0
            avg_total_latency = cursor.execute("SELECT AVG(latency_ms) FROM request_logs").fetchone()[0] or 0.0

            # Calculate estimated tokens saved (estimating avg 150 tokens per query/response pair)
            estimated_tokens_saved = total_hits * 150
            estimated_cost_saved = (estimated_tokens_saved / 1000.0) * cost_per_1k_tokens

            return {
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate_pct": round(hit_rate_pct, 2),
                "avg_hit_latency_ms": round(float(avg_hit_latency), 2),
                "avg_miss_latency_ms": round(float(avg_miss_latency), 2),
                "avg_total_latency_ms": round(float(avg_total_latency), 2),
                "estimated_tokens_saved": estimated_tokens_saved,
                "estimated_cost_saved_usd": round(estimated_cost_saved, 5),
            }

    def clear(self) -> None:
        """Truncate the request_logs table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM request_logs")
            conn.commit()
        log.info("Request logs table cleared.")
