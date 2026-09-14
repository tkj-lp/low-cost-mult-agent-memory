from __future__ import annotations

import json
from typing import Any

from .contracts import RetrievalRequest
from .store import SQLiteHypergraphMemoryStore
from .text import indexed_tokens


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class ServiceStore(SQLiteHypergraphMemoryStore):
    """Service-owned metadata keeps legacy records and interfaces isolated."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        try:
            with self.transaction() as connection:
                statements = (
                    """CREATE TABLE IF NOT EXISTS memory_batches (
                        namespace TEXT NOT NULL, batch_id TEXT NOT NULL,
                        task_family TEXT NOT NULL, dataset_id TEXT NOT NULL,
                        group_id TEXT NOT NULL, status TEXT NOT NULL,
                        policy_json TEXT NOT NULL, created_at TEXT NOT NULL,
                        report_json TEXT, PRIMARY KEY(namespace, batch_id))""",
                    """CREATE TABLE IF NOT EXISTS memory_scopes (
                        global_memory_id TEXT PRIMARY KEY REFERENCES memories(global_memory_id) ON DELETE CASCADE,
                        namespace TEXT NOT NULL, batch_id TEXT NOT NULL,
                        layer TEXT NOT NULL, status TEXT NOT NULL,
                        scope_kind TEXT NOT NULL, scope_key TEXT NOT NULL)""",
                    """CREATE INDEX IF NOT EXISTS idx_memory_scopes
                        ON memory_scopes(namespace, batch_id, layer, status)""",
                    """CREATE TABLE IF NOT EXISTS memory_operations (
                        namespace TEXT NOT NULL, batch_id TEXT NOT NULL,
                        operation_id TEXT NOT NULL, request_json TEXT NOT NULL,
                        receipt_json TEXT NOT NULL,
                        PRIMARY KEY(namespace, batch_id, operation_id))""",
                    """CREATE TABLE IF NOT EXISTS memory_feedback (
                        namespace TEXT NOT NULL, batch_id TEXT NOT NULL,
                        task_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                        PRIMARY KEY(namespace, batch_id, task_id))""",
                    """CREATE TABLE IF NOT EXISTS memory_retrievals (
                        namespace TEXT NOT NULL, batch_id TEXT NOT NULL,
                        retrieval_id TEXT NOT NULL, request_json TEXT NOT NULL,
                        result_json TEXT NOT NULL, created_at TEXT NOT NULL,
                        PRIMARY KEY(namespace, retrieval_id))""",
                    """CREATE INDEX IF NOT EXISTS idx_memory_retrievals_batch
                        ON memory_retrievals(namespace, batch_id, created_at)""",
                    """CREATE TABLE IF NOT EXISTS memory_aliases (
                        namespace TEXT NOT NULL, alias_id TEXT NOT NULL,
                        canonical_id TEXT NOT NULL,
                        PRIMARY KEY(namespace, alias_id))""",
                    """CREATE TABLE IF NOT EXISTS memory_provenance (
                        namespace TEXT NOT NULL, global_memory_id TEXT NOT NULL,
                        snapshot_json TEXT NOT NULL,
                        PRIMARY KEY(namespace, global_memory_id))""",
                    """CREATE TABLE IF NOT EXISTS memory_vectors (
                        global_memory_id TEXT PRIMARY KEY REFERENCES memories(global_memory_id) ON DELETE CASCADE,
                        model_id TEXT NOT NULL, dimensions INTEGER NOT NULL,
                        vector_json TEXT NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS memory_content_revision (
                        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                        revision INTEGER NOT NULL)""",
                    """INSERT OR IGNORE INTO memory_content_revision VALUES (1, 0)""",
                    """CREATE TRIGGER IF NOT EXISTS memory_content_revision_insert
                        AFTER INSERT ON memories BEGIN
                        UPDATE memory_content_revision SET revision=revision+1 WHERE singleton=1;
                        END""",
                    """CREATE TRIGGER IF NOT EXISTS memory_content_revision_update
                        AFTER UPDATE ON memories BEGIN
                        UPDATE memory_content_revision SET revision=revision+1 WHERE singleton=1;
                        END""",
                    """CREATE TRIGGER IF NOT EXISTS memory_content_revision_delete
                        AFTER DELETE ON memories BEGIN
                        UPDATE memory_content_revision SET revision=revision+1 WHERE singleton=1;
                        END""",
                    """CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
                        global_memory_id UNINDEXED, document, tokenize='unicode61')""",
                    """CREATE TRIGGER IF NOT EXISTS memory_search_delete
                        AFTER DELETE ON memories BEGIN
                        DELETE FROM memory_search WHERE rowid=old.rowid;
                        END""",
                )
                for statement in statements:
                    connection.execute(statement)
                self._ensure_column(
                    connection,
                    "memory_batches",
                    "task_family",
                    "TEXT NOT NULL DEFAULT ''",
                )
                self._ensure_column(
                    connection,
                    "memory_batches",
                    "dataset_id",
                    "TEXT NOT NULL DEFAULT ''",
                )
                self._ensure_column(
                    connection,
                    "memory_scopes",
                    "scope_kind",
                    "TEXT NOT NULL DEFAULT 'batch'",
                )
                self._ensure_column(
                    connection, "memory_scopes", "scope_key", "TEXT NOT NULL DEFAULT ''"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_scope_key "
                    "ON memory_scopes(namespace, scope_kind, scope_key, layer, status)"
                )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _ensure_column(
        connection: Any, table: str, column: str, declaration: str
    ) -> None:
        columns = {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @property
    def revision(self) -> tuple[int, int]:
        """Revision of searchable memory content, excluding audit-only writes."""

        row = self.read_rows(
            "SELECT revision FROM memory_content_revision WHERE singleton=1"
        )
        return int(row[0][0]), 0

    def put(
        self,
        namespace: str,
        record: dict[str, Any],
        *,
        vector: list[float] | None = None,
        model_id: str = "",
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            record = dict(record)
            for field in ("source_memory_ids", "knowledge_refs"):
                refs = []
                for ref in record.get(field, []):
                    row = connection.execute(
                        "SELECT canonical_id FROM memory_aliases WHERE namespace=? AND alias_id=?",
                        (namespace, ref),
                    ).fetchone()
                    refs.append(str(row[0]) if row else ref)
                record[field] = sorted(set(refs))
            stored = self.upsert(record)
            memory_id = stored["global_memory_id"]
            if stored["memory_layer"] == "experience" and not stored["batch_id"]:
                scope_kind = str(stored.get("experience_scope") or "task_family")
                scope_key = str(
                    stored.get("experience_scope_key")
                    or stored.get(
                        "dataset_id" if scope_kind == "dataset" else "task_family"
                    )
                    or ""
                )
                if scope_kind not in {"task_family", "dataset"} or not scope_key:
                    raise ValueError(
                        "long-term experience requires an explicit reuse scope"
                    )
            else:
                scope_kind = "batch"
                scope_key = stored["batch_id"]
            rowid = connection.execute(
                "SELECT rowid FROM memories WHERE global_memory_id=?", (memory_id,)
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO memory_scopes(
                       global_memory_id, namespace, batch_id, layer, status, scope_kind, scope_key
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(global_memory_id) DO UPDATE SET
                   batch_id=excluded.batch_id, layer=excluded.layer, status=excluded.status,
                   scope_kind=excluded.scope_kind, scope_key=excluded.scope_key""",
                (
                    memory_id,
                    namespace,
                    stored["batch_id"],
                    stored["memory_layer"],
                    stored["review_status"],
                    scope_kind,
                    scope_key,
                ),
            )
            connection.execute("DELETE FROM memory_search WHERE rowid=?", (rowid,))
            connection.execute(
                "INSERT INTO memory_search(rowid, global_memory_id, document) VALUES (?, ?, ?)",
                (
                    rowid,
                    memory_id,
                    " ".join(
                        indexed_tokens(
                            " ".join(
                                [
                                    stored["search_document"],
                                    *stored.get("entities", []),
                                    *stored.get("entity_aliases", {}).keys(),
                                ]
                            )
                        )
                    ),
                ),
            )
            for field, index_type in (
                ("source_memory_ids", "source_reference"),
                ("knowledge_refs", "knowledge_reference"),
            ):
                connection.executemany(
                    "INSERT OR IGNORE INTO memory_index VALUES (?, ?, ?, 3.0)",
                    [
                        (memory_id, index_type, str(ref))
                        for ref in stored.get(field, [])
                    ],
                )
            if vector is not None:
                dimensions = connection.execute(
                    "SELECT DISTINCT dimensions FROM memory_vectors WHERE model_id=?",
                    (model_id,),
                ).fetchall()
                if any(int(row[0]) != len(vector) for row in dimensions):
                    raise ValueError(
                        "one model_id cannot represent multiple embedding dimensions"
                    )
                connection.execute(
                    """INSERT INTO memory_vectors VALUES (?, ?, ?, ?)
                       ON CONFLICT(global_memory_id) DO UPDATE SET
                       model_id=excluded.model_id, dimensions=excluded.dimensions,
                       vector_json=excluded.vector_json""",
                    (memory_id, model_id, len(vector), json_text(vector)),
                )
        return stored

    def batch(self, namespace: str, batch_id: str) -> dict[str, Any]:
        rows = self.read_rows(
            "SELECT * FROM memory_batches WHERE namespace=? AND batch_id=?",
            (namespace, batch_id),
        )
        if not rows:
            raise KeyError(f"Unknown batch: {batch_id}")
        return dict(rows[0])

    def retrieval_receipt(
        self, namespace: str, batch_id: str, retrieval_id: str
    ) -> dict[str, Any]:
        rows = self.read_rows(
            "SELECT * FROM memory_retrievals "
            "WHERE namespace=? AND batch_id=? AND retrieval_id=?",
            (namespace, batch_id, retrieval_id),
        )
        if not rows:
            raise KeyError(f"Unknown retrieval receipt: {retrieval_id}")
        return dict(rows[0])

    def eligible_sql(
        self,
        namespace: str,
        request: RetrievalRequest,
        *,
        all_layers: bool = False,
    ) -> tuple[str, list[Any]]:
        batch = self.batch(namespace, request.batch_id)
        statuses = ["active"]
        if request.include_candidates:
            statuses.append("candidate")
        where = (
            "s.namespace=? AND (s.batch_id=? OR (s.batch_id='' AND s.layer='experience' AND "
            "((s.scope_kind='task_family' AND s.scope_key=?) OR "
            "(s.scope_kind='dataset' AND s.scope_key=?))))"
        )
        params: list[Any] = [
            namespace,
            request.batch_id,
            batch["task_family"],
            batch["dataset_id"],
        ]
        where += " AND s.status IN (" + ",".join("?" for _ in statuses) + ")"
        params.extend(statuses)
        layers = (
            tuple(dict.fromkeys((*request.layers, *request.bridge_layers)))
            if all_layers
            else request.layers
        )
        where += " AND s.layer IN (" + ",".join("?" for _ in layers) + ")"
        params.extend(layers)
        if request.exclude_ids:
            where += (
                " AND s.global_memory_id NOT IN ("
                + ",".join("?" for _ in request.exclude_ids)
                + ")"
            )
            params.extend(request.exclude_ids)
        return where, params

    def resolve(self, namespace: str, batch_id: str, memory_id: str) -> dict[str, Any]:
        batch = self.batch(namespace, batch_id)
        alias = self.read_rows(
            "SELECT canonical_id FROM memory_aliases WHERE namespace=? AND alias_id=?",
            (namespace, memory_id),
        )
        resolved = str(alias[0][0]) if alias else memory_id
        rows = self.read_rows(
            """SELECT m.record_json FROM memories m JOIN memory_scopes s
               ON m.global_memory_id=s.global_memory_id
               WHERE s.namespace=? AND (
                   s.batch_id=? OR (s.batch_id='' AND s.layer='experience' AND
                   ((s.scope_kind='task_family' AND s.scope_key=?) OR
                    (s.scope_kind='dataset' AND s.scope_key=?)))
               )
               AND (m.global_memory_id=? OR m.memory_id=?)""",
            (
                namespace,
                batch_id,
                batch["task_family"],
                batch["dataset_id"],
                resolved,
                resolved,
            ),
        )
        if len(rows) > 1:
            raise ValueError(f"Ambiguous memory ID; use global_memory_id: {memory_id}")
        if not rows:
            raise KeyError(f"Memory is missing or outside batch scope: {memory_id}")
        return json.loads(rows[0][0])

    def fetch_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        rows = self.read_rows(
            "SELECT global_memory_id, record_json FROM memories WHERE global_memory_id IN ("
            + ",".join("?" for _ in ids)
            + ")",
            ids,
        )
        return {str(row[0]): json.loads(row[1]) for row in rows}

    def scoped_records(self, namespace: str, batch_id: str) -> list[dict[str, Any]]:
        batch = self.batch(namespace, batch_id)
        rows = self.read_rows(
            """SELECT m.record_json FROM memories m JOIN memory_scopes s
               ON m.global_memory_id=s.global_memory_id
               WHERE s.namespace=? AND (
                   s.batch_id=? OR (s.batch_id='' AND s.layer='experience' AND
                   ((s.scope_kind='task_family' AND s.scope_key=?) OR
                    (s.scope_kind='dataset' AND s.scope_key=?)))
               ) ORDER BY m.global_memory_id""",
            (namespace, batch_id, batch["task_family"], batch["dataset_id"]),
        )
        return [json.loads(row[0]) for row in rows]
