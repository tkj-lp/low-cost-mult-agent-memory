from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .builder import build_hypergraph_view
from .distill import lifecycle_stage
from .layers import memory_layer_for_record, normalize_memory_layer
from .memory_model import normalize_memory_record
from .retrieval import RetrievalMode, search_memory_by_mode
from .text import tokenize


class SQLiteHypergraphMemoryStore:
    """支持事务、重启恢复和增量索引的轻量持久化共享记忆库。"""

    def __init__(self, path: str | Path = "hypergraph_memory.sqlite3") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    global_memory_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_index (
                    global_memory_id TEXT NOT NULL,
                    index_type TEXT NOT NULL,
                    index_key TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (global_memory_id, index_type, index_key),
                    FOREIGN KEY (global_memory_id) REFERENCES memories(global_memory_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_index_lookup ON memory_index(index_type, index_key)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_legacy_id ON memories(memory_id)"
            )

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        with self._lock:
            depth = self._transaction_depth
            savepoint = f"memory_tx_{depth}"
            self._connection.execute(
                ("BEGIN IMMEDIATE" if write else "BEGIN")
                if depth == 0
                else f"SAVEPOINT {savepoint}"
            )
            self._transaction_depth += 1
            try:
                yield self._connection
                if depth == 0:
                    self._connection.commit()
                else:
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                if depth == 0:
                    self._connection.rollback()
                else:
                    self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
            finally:
                self._transaction_depth -= 1

    def read_rows(self, sql: str, parameters: Iterable[Any] = ()) -> list[sqlite3.Row]:
        """Read under the same lock used by memory-service transactions."""

        with self._lock:
            return self._connection.execute(sql, tuple(parameters)).fetchall()

    @property
    def revision(self) -> tuple[int, int]:
        with self._lock:
            version = self._connection.execute("PRAGMA data_version").fetchone()[0]
            return self._connection.total_changes, int(version)

    @staticmethod
    def _storage_record(record: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_memory_record(record)
        normalized.pop("hypergraph", None)
        return normalized

    @staticmethod
    def _index_entries(
        record: dict[str, Any],
        *,
        hypergraph: dict[str, Any] | None = None,
    ) -> set[tuple[str, str, float]]:
        hypergraph = hypergraph or build_hypergraph_view(record, persist=False)
        entries: set[tuple[str, str, float]] = set()
        entries.add(("memory_layer", memory_layer_for_record(record), 5.0))
        for token in tokenize(str(record.get("search_document") or "")):
            entries.add(("keyword", token, 1.0))
        for tag in record.get("tags", []):
            entries.add(("tag", str(tag).lower().replace("_", "-"), 3.0))
        for edge in hypergraph.get("hyperedges", []):
            if not isinstance(edge, dict):
                continue
            for node in edge.get("nodes", []):
                entries.add(("hypergraph_node", str(node), 1.0))
        chain = (
            record.get("evidence_chain")
            if isinstance(record.get("evidence_chain"), dict)
            else {}
        )
        if chain.get("chain_id"):
            entries.add(
                (
                    "evidence_chain",
                    str(chain["chain_id"]),
                    float(chain.get("completeness", 0.0)),
                )
            )
        return entries

    def upsert(
        self,
        record: dict[str, Any],
        *,
        hypergraph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self._storage_record(record)
        memory_id = str(normalized["memory_id"])
        global_memory_id = str(normalized["global_memory_id"])
        payload = json.dumps(
            normalized, ensure_ascii=False, sort_keys=True, default=str
        )
        entries = self._index_entries(normalized, hypergraph=hypergraph)
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO memories(global_memory_id, memory_id, memory_type, source_agent, created_at, updated_at, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(global_memory_id) DO UPDATE SET
                    memory_id=excluded.memory_id,
                    memory_type=excluded.memory_type,
                    source_agent=excluded.source_agent,
                    updated_at=excluded.updated_at,
                    record_json=excluded.record_json
                """,
                (
                    global_memory_id,
                    memory_id,
                    normalized["memory_type"],
                    normalized["source_agent"],
                    normalized["created_at"],
                    normalized["updated_at"],
                    payload,
                ),
            )
            connection.execute(
                "DELETE FROM memory_index WHERE global_memory_id = ?",
                (global_memory_id,),
            )
            connection.executemany(
                "INSERT INTO memory_index(global_memory_id, index_type, index_key, weight) VALUES (?, ?, ?, ?)",
                [
                    (global_memory_id, index_type, index_key, weight)
                    for index_type, index_key, weight in entries
                ],
            )
        return normalized

    def upsert_many(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        with self.transaction():
            return [self.upsert(record) for record in records]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT record_json FROM memories
                WHERE global_memory_id = ? OR memory_id = ?
                ORDER BY updated_at DESC, global_memory_id
                LIMIT 1
                """,
                (str(memory_id), str(memory_id)),
            ).fetchone()
        return json.loads(row["record_json"]) if row else None

    def list(
        self,
        *,
        memory_types: Iterable[str] | None = None,
        memory_layers: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        sql = "SELECT record_json FROM memories"
        allowed_types = [str(value) for value in memory_types or []]
        if allowed_types:
            placeholders = ",".join("?" for _ in allowed_types)
            sql += f" WHERE memory_type IN ({placeholders})"
            parameters.extend(allowed_types)
        sql += " ORDER BY created_at, memory_id"
        allowed_layers = (
            {normalize_memory_layer(layer) for layer in memory_layers}
            if memory_layers
            else set()
        )
        if limit is not None and not allowed_layers:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            sql += " LIMIT ?"
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        records = [json.loads(row["record_json"]) for row in rows]
        if allowed_layers:
            records = [
                record
                for record in records
                if memory_layer_for_record(record) in allowed_layers
            ]
            if limit is not None:
                if limit < 0:
                    raise ValueError("limit must be non-negative")
                records = records[:limit]
        return records

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM memories"
            ).fetchone()
        return int(row["count"])

    def delete(self, memory_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE global_memory_id = ? OR memory_id = ?",
                (str(memory_id), str(memory_id)),
            )
        return cursor.rowcount > 0

    def record_reuse(
        self,
        memory_id: str,
        *,
        adopted: bool,
        successful: bool = False,
    ) -> dict[str, Any] | None:
        """原子记录显式采用结果，并更新复用时间、版本和生命周期。"""

        with self._lock:
            record = self.get(memory_id)
            if record is None or not adopted:
                return record
            record["reuse_count"] = int(record.get("reuse_count", 0) or 0) + 1
            if successful:
                record["reuse_success_count"] = (
                    int(record.get("reuse_success_count", 0) or 0) + 1
                )
            timestamp = datetime.now(timezone.utc).isoformat()
            record["last_reused_at"] = timestamp
            record["updated_at"] = timestamp
            record["record_version"] = int(record.get("record_version", 1) or 1) + 1
            record["lifecycle_stage"] = lifecycle_stage(record)
            return self.upsert(record)

    def _indexed_candidate_ids(self, query: str, mode: RetrievalMode) -> list[str]:
        if mode not in {"keyword", "tag"}:
            return []
        values = tokenize(query)
        if not values:
            return []
        placeholders = ",".join("?" for _ in values)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT global_memory_id, SUM(weight) AS score
                FROM memory_index
                WHERE index_type = ? AND index_key IN ({placeholders})
                GROUP BY global_memory_id
                ORDER BY score DESC, global_memory_id
                """,
                [mode, *values],
            ).fetchall()
        return [str(row["global_memory_id"]) for row in rows]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        mode: RetrievalMode = "hypergraph",
        memory_layers: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidate_ids = self._indexed_candidate_ids(query, mode)
        if candidate_ids:
            records = [
                record
                for memory_id in candidate_ids
                if (record := self.get(memory_id)) is not None
            ]
        else:
            records = self.list(memory_layers=memory_layers)
        if memory_layers and candidate_ids:
            allowed_layers = {normalize_memory_layer(layer) for layer in memory_layers}
            records = [
                record
                for record in records
                if memory_layer_for_record(record) in allowed_layers
            ]
        return search_memory_by_mode(records, query, top_k=top_k, mode=mode)

    def rebuild_indexes(self) -> int:
        records = self.list()
        with self.transaction() as connection:
            connection.execute("DELETE FROM memory_index")
            for record in records:
                entries = self._index_entries(record)
                connection.executemany(
                    "INSERT INTO memory_index(global_memory_id, index_type, index_key, weight) VALUES (?, ?, ?, ?)",
                    [
                        (record["global_memory_id"], index_type, index_key, weight)
                        for index_type, index_key, weight in entries
                    ],
                )
        return len(records)

    def export_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self.list():
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        return target

    def import_jsonl(self, path: str | Path) -> int:
        source = Path(path)
        records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.upsert_many(records)
        return len(records)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteHypergraphMemoryStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
