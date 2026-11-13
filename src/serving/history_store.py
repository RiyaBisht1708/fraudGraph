"""SQLite persistence for non-sensitive FraudGraph inference history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


class HistoryStoreError(Exception):
    """Raised when history storage cannot complete an operation."""


class HistoryStore:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=5)
        except (OSError, sqlite3.Error) as exc:
            raise HistoryStoreError("History storage is unavailable.") from exc

        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise HistoryStoreError("History storage is unavailable.") from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL,
                    fraud_score REAL NOT NULL,
                    decision TEXT NOT NULL CHECK (decision IN ('APPROVE', 'FLAG', 'DECLINE')),
                    latency_ms REAL,
                    transaction_amount REAL NOT NULL,
                    product_category TEXT NOT NULL,
                    transaction_dt REAL NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    top_reasons_json TEXT NOT NULL,
                    graph_evidence_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transaction_history_created
                ON transaction_history (created_at_utc DESC, id DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_transaction_history_decision
                ON transaction_history (decision)
                """
            )

    @staticmethod
    def _json_default(value: Any) -> str:
        return str(value)

    @staticmethod
    def _reason_data(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source = result.get("top_reasons")
        reasons = source if isinstance(source, list) else []
        top_reasons: list[dict[str, Any]] = []
        graph_evidence: list[dict[str, Any]] = []
        for reason in reasons:
            if not isinstance(reason, dict):
                continue
            evidence = reason.get("evidence")
            safe_evidence = evidence if isinstance(evidence, dict) else {}
            top_reasons.append({
                "factor": reason.get("factor"),
                "weight": reason.get("weight"),
                "edge_type": reason.get("edge_type"),
                "evidence": safe_evidence,
            })
            edge_type = reason.get("edge_type")
            if edge_type in {"on_device", "uses_card", "at_merchant", "velocity_cluster"}:
                graph_evidence.append({
                    "edge_type": edge_type,
                    "evidence": safe_evidence,
                })
        return top_reasons, graph_evidence

    def save_analysis(self, payload: dict[str, Any], result: dict[str, Any]) -> int:
        top_reasons, graph_evidence = self._reason_data(result)
        created_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO transaction_history (
                        transaction_id, fraud_score, decision, latency_ms,
                        transaction_amount, product_category, transaction_dt,
                        created_at_utc, top_reasons_json, graph_evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(result["transaction_id"]),
                        float(result["fraud_score"]),
                        str(result["decision"]),
                        float(result["latency_ms"]) if result.get("latency_ms") is not None else None,
                        float(payload["TransactionAmt"]),
                        str(payload["ProductCD"]),
                        float(payload["TransactionDT"]),
                        created_at,
                        json.dumps(top_reasons, separators=(",", ":"), default=self._json_default),
                        json.dumps(graph_evidence, separators=(",", ":"), default=self._json_default),
                    ),
                )
                record_id = cursor.lastrowid
                if record_id is None:
                    raise HistoryStoreError("History record was not created.")
                return int(record_id)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise HistoryStoreError("History record could not be stored.") from exc

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "transaction_id": row["transaction_id"],
            "fraud_score": row["fraud_score"],
            "decision": row["decision"],
            "latency_ms": row["latency_ms"],
            "transaction_amount": row["transaction_amount"],
            "product_category": row["product_category"],
            "transaction_dt": row["transaction_dt"],
            "created_at_utc": row["created_at_utc"],
        }

    @classmethod
    def _detail(cls, row: sqlite3.Row) -> dict[str, Any]:
        record = cls._summary(row)
        try:
            record["top_reasons"] = json.loads(row["top_reasons_json"])
            record["graph_evidence"] = json.loads(row["graph_evidence_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise HistoryStoreError("Stored history record is invalid.") from exc
        return record

    def list_records(
        self,
        limit: int = 20,
        offset: int = 0,
        search: str | None = None,
        decision: str | None = None,
    ) -> dict[str, Any]:
        conditions: list[str] = []
        parameters: list[Any] = []
        if search:
            conditions.append("instr(transaction_id, ?) > 0")
            parameters.append(search)
        if decision:
            conditions.append("decision = ?")
            parameters.append(decision)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM transaction_history {where_clause}",
                parameters,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT id, transaction_id, fraud_score, decision, latency_ms,
                       transaction_amount, product_category, transaction_dt, created_at_utc
                FROM transaction_history
                {where_clause}
                ORDER BY created_at_utc DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()
        return {"items": [self._summary(row) for row in rows], "total": total}

    def get_record(self, record_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM transaction_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._detail(row) if row is not None else None

    def is_healthy(self) -> bool:
        try:
            with self._connection() as connection:
                connection.execute("SELECT 1 FROM transaction_history LIMIT 1").fetchone()
            return True
        except HistoryStoreError:
            return False
