from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from .contracts import (
    BatchPolicy,
    EpisodeMemoryInput,
    ExperienceMemoryInput,
    ExperienceReview,
    KnowledgeMemoryInput,
    MaintenanceReport,
    MemoryEncoder,
    MemoryRelationUpdate,
    RetrievalRequest,
    RetrievalResult,
    TaskFeedback,
    TaskMemoryCommit,
    WriteReceipt,
)
from .entities import canonical_entity, extract_canonical_entities
from .indexed_retrieval import IndexedRetriever, checked_vector
from .maintenance import plan_maintenance, review_status
from .memory_model import normalize_memory_record
from .service_store import ServiceStore, json_text
from .text import stable_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryService:
    """Public memory boundary; the caller drives task and batch scheduling."""

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str = "default",
        encoder: MemoryEncoder | None = None,
        vector_backend: str = "exact",
        reranker: Callable[[str, list[dict[str, Any]]], list[float]] | None = None,
        merge_verifier: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace is required")
        if encoder is not None and not str(encoder.model_id).strip():
            raise ValueError(
                "encoder.model_id must identify the complete embedding space"
            )
        self.namespace, self.encoder, self.merge_verifier = (
            namespace,
            encoder,
            merge_verifier,
        )
        self.store = ServiceStore(str(path))
        try:
            self.retriever = IndexedRetriever(
                self.store,
                namespace,
                encoder,
                vector_backend=vector_backend,
                reranker=reranker,
            )
        except Exception:
            self.store.close()
            raise

    def begin_batch(
        self,
        batch_id: str,
        *,
        task_family: str,
        dataset_id: str,
        group_id: str = "",
        policy: BatchPolicy | None = None,
    ) -> dict[str, Any]:
        if not batch_id.strip() or not task_family.strip() or not dataset_id.strip():
            raise ValueError("batch_id, task_family and dataset_id are required")
        policy_json = json_text(asdict(policy or BatchPolicy()))
        with self.store.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_batches WHERE namespace=? AND batch_id=?",
                (self.namespace, batch_id),
            ).fetchall()
            if rows:
                batch = dict(rows[0])
                if (
                    batch["task_family"] != task_family
                    or batch["dataset_id"] != dataset_id
                    or batch["group_id"] != group_id
                    or batch["policy_json"] != policy_json
                ):
                    raise ValueError(
                        "batch ID already exists with different configuration"
                    )
            else:
                connection.execute(
                    """INSERT INTO memory_batches(
                           namespace, batch_id, task_family, dataset_id, group_id,
                           status, policy_json, created_at, report_json
                       ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, NULL)""",
                    (
                        self.namespace,
                        batch_id,
                        task_family,
                        dataset_id,
                        group_id,
                        policy_json,
                        _now(),
                    ),
                )
                batch = self.store.batch(self.namespace, batch_id)
        return {
            key: batch[key]
            for key in (
                "batch_id",
                "task_family",
                "dataset_id",
                "group_id",
                "status",
                "created_at",
            )
        }

    def _open_batch(self, batch_id: str) -> dict[str, Any]:
        batch = self.store.batch(self.namespace, batch_id)
        if batch["status"] != "open":
            raise ValueError(f"Batch is {batch['status']}: {batch_id}")
        return batch

    def _operation(
        self,
        batch_id: str,
        operation_id: str,
        request: dict[str, Any],
        action: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if not operation_id.strip():
            raise ValueError("operation_id is required for retry-safe writes")
        request_json = json_text(request)
        with self.store.transaction() as connection:
            previous = connection.execute(
                "SELECT request_json,receipt_json FROM memory_operations "
                "WHERE namespace=? AND batch_id=? AND operation_id=?",
                (self.namespace, batch_id, operation_id),
            ).fetchone()
            if previous:
                if previous[0] != request_json:
                    raise ValueError("operation_id was reused with different inputs")
                return json.loads(previous[1])
            self._open_batch(batch_id)
            receipt = action()
            connection.execute(
                "INSERT INTO memory_operations VALUES (?, ?, ?, ?, ?)",
                (
                    self.namespace,
                    batch_id,
                    operation_id,
                    request_json,
                    json_text(receipt),
                ),
            )
            return receipt

    def _prepare(
        self, batch_id: str, layer: str, task_id: str, raw: dict[str, Any]
    ) -> dict[str, Any]:
        for field in ("source_agent", "task_topic", "summary"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                raise ValueError(f"memory requires a non-empty {field}")
        record = deepcopy(raw)
        memory_id = str(
            record.get("memory_id") or "mem-" + stable_id(json_text(raw), length=20)
        )
        global_id = "gmem-" + stable_id(
            json_text([self.namespace, batch_id, layer, task_id, memory_id]), length=32
        )
        if self.store.get(global_id) is not None:
            raise ValueError(
                "memory ID already exists; retry the original operation instead of overwriting"
            )
        batch = self.store.batch(self.namespace, batch_id)
        policy = BatchPolicy(**json.loads(batch["policy_json"]))
        record.update(
            {
                "memory_id": memory_id,
                "global_memory_id": global_id,
                "batch_id": batch_id,
                "origin_batch_id": batch_id,
                "task_id": task_id,
                "group_id": batch["group_id"],
                "task_family": batch["task_family"],
                "dataset_id": batch["dataset_id"],
                "experience_scope": policy.experience_scope,
                "experience_scope_key": (
                    batch["dataset_id"]
                    if policy.experience_scope == "dataset"
                    else batch["task_family"]
                ),
                "memory_layer": layer,
                "memory_type": "experience" if layer == "experience" else "evidence",
                "review_status": "candidate" if layer == "experience" else "active",
                "lifecycle_stage": (
                    "needs-validation" if layer == "experience" else "episodic"
                ),
                "verified_success_count": 0,
                "verified_failure_count": 0,
                "verification_history": [],
                "reuse_count": 0,
                "retrieval_count": 0,
                "reuse_success_count": 0,
                "reuse_failure_count": 0,
                "record_version": 1,
                "created_at": str(raw.get("created_at") or _now()),
            }
        )
        record.pop("embedding", None)
        record.pop("hypergraph", None)
        aliases = record.get("entity_aliases", {})
        if not isinstance(aliases, dict):
            raise ValueError(
                "entity_aliases must map explicit aliases to canonical labels"
            )
        aliases = {
            canonical_entity(key): canonical_entity(value)
            for key, value in aliases.items()
        }
        if any(not key or not value for key, value in aliases.items()):
            raise ValueError("entity aliases must be non-empty labels")
        record["entity_aliases"] = aliases
        explicit = record.get("entities", [])
        if not isinstance(explicit, list) or any(
            not isinstance(value, str) for value in explicit
        ):
            raise ValueError("entities must be a list of labels supplied by the caller")
        entities = extract_canonical_entities(
            record["task_topic"],
            record["summary"],
            explicit=[*explicit, *aliases.values()],
        )
        record["entities"] = sorted(
            {aliases.get(entity, entity) for entity in entities}
        )
        record["entity_origin"] = (
            "caller-and-heuristic" if explicit or aliases else "heuristic"
        )
        for field in ("knowledge_refs", "source_memory_ids"):
            values = record.get(field, [])
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list")
            record[field] = sorted(
                {
                    self.store.resolve(self.namespace, batch_id, str(value))[
                        "global_memory_id"
                    ]
                    for value in values
                }
            )
        if layer == "experience":
            if not record["source_memory_ids"]:
                raise ValueError("experience must cite source memory IDs")
            for field in ("problem_pattern", "solution_pattern"):
                if not str(record.get(field) or "").strip():
                    raise ValueError(f"experience requires {field}")
        return normalize_memory_record(record)

    def _write(
        self,
        batch_id: str,
        layer: str,
        task_id: str,
        records: Iterable[dict[str, Any]],
        operation_id: str,
    ) -> WriteReceipt:
        inputs = [dict(record) for record in records]
        if not inputs:
            raise ValueError("records must not be empty")
        if layer != "knowledge" and not task_id.strip():
            raise ValueError("task_id is required")

        def action() -> dict[str, Any]:
            stored: list[dict[str, Any]] = []
            for raw in inputs:
                record = self._prepare(batch_id, layer, task_id, raw)
                self._store_record(record)
                stored.append(
                    {
                        "memory_id": record["global_memory_id"],
                        "local_memory_id": record["memory_id"],
                        "memory_layer": layer,
                    }
                )
            return {
                "batch_id": batch_id,
                "operation_id": operation_id,
                "records": stored,
            }

        return self._operation(
            batch_id,
            operation_id,
            {"layer": layer, "task_id": task_id, "records": inputs},
            action,
        )

    def write_knowledge(
        self,
        batch_id: str,
        records: Iterable[KnowledgeMemoryInput | dict[str, Any]],
        *,
        operation_id: str,
    ) -> WriteReceipt:
        return self._write(batch_id, "knowledge", "", records, operation_id)

    def write_episode(
        self,
        batch_id: str,
        task_id: str,
        record: EpisodeMemoryInput | dict[str, Any],
        *,
        operation_id: str,
    ) -> WriteReceipt:
        return self._write(batch_id, "episodic", task_id, [record], operation_id)

    def propose_experience(
        self,
        batch_id: str,
        task_id: str,
        record: ExperienceMemoryInput | dict[str, Any],
        *,
        operation_id: str,
    ) -> WriteReceipt:
        return self._write(batch_id, "experience", task_id, [record], operation_id)

    def _store_record(
        self,
        record: dict[str, Any],
        *,
        previous_search_document: str | None = None,
    ) -> dict[str, Any]:
        vector = None
        if self.encoder is not None and (
            previous_search_document is None
            or record["search_document"] != previous_search_document
        ):
            values = self.encoder.encode([record["search_document"]])
            if len(values) != 1:
                raise ValueError("encoder returned an unexpected batch size")
            vector = checked_vector(values[0])
        return self.store.put(
            self.namespace,
            record,
            vector=vector,
            model_id=self.encoder.model_id if self.encoder else "",
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        with self.store.transaction() as connection:
            self._open_batch(request.batch_id)
            seeds = tuple(
                self.store.resolve(self.namespace, request.batch_id, value)[
                    "global_memory_id"
                ]
                for value in request.seed_memory_ids
            )
            excluded = tuple(
                self.store.resolve(self.namespace, request.batch_id, value)[
                    "global_memory_id"
                ]
                for value in request.exclude_ids
            )
            result = self.retriever.search(
                replace(request, seed_memory_ids=seeds, exclude_ids=excluded)
            )
            retrieval_id = "ret-" + uuid4().hex
            receipt: RetrievalResult = {
                "retrieval_id": retrieval_id,
                "batch_id": request.batch_id,
                "hits": result["hits"],
                "diagnostics": result["diagnostics"],
            }
            connection.execute(
                "INSERT INTO memory_retrievals VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.namespace,
                    request.batch_id,
                    retrieval_id,
                    json_text(asdict(request)),
                    json_text(receipt),
                    _now(),
                ),
            )
            return receipt

    def record_feedback(self, feedback: TaskFeedback) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            batch = self._open_batch(feedback.batch_id)
            policy = BatchPolicy(**json.loads(batch["policy_json"]))
            receipt_memory_ids: set[str] = set()
            for retrieval_id in feedback.retrieval_ids:
                receipt = self.store.retrieval_receipt(
                    self.namespace, feedback.batch_id, retrieval_id
                )
                result = json.loads(receipt["result_json"])
                receipt_memory_ids.update(
                    str(hit["memory_id"])
                    for hit in result.get("hits", [])
                    if hit.get("memory_id")
                )
            ids = (
                receipt_memory_ids
                | set(feedback.retrieved_ids)
                | set(feedback.adopted_ids)
                | set(feedback.memory_verdicts)
            )
            records = {
                key: self.store.resolve(self.namespace, feedback.batch_id, key)
                for key in ids
            }
            canonical = {
                key: record["global_memory_id"] for key, record in records.items()
            }
            reported_retrieved = {canonical[key] for key in feedback.retrieved_ids}
            receipt_retrieved = {canonical[key] for key in receipt_memory_ids}
            if feedback.retrieval_ids and reported_retrieved != receipt_retrieved:
                if reported_retrieved:
                    raise ValueError(
                        "retrieved_ids must exactly match the referenced retrieval receipts"
                    )
                reported_retrieved = receipt_retrieved
            retrieved = reported_retrieved
            adopted = {canonical[key] for key in feedback.adopted_ids}
            if not adopted <= retrieved:
                raise ValueError("adopted_ids must be included in retrieved_ids")
            verdicts: dict[str, bool | None] = {}
            for key, value in feedback.memory_verdicts.items():
                resolved = canonical[key]
                if resolved in verdicts and verdicts[resolved] is not value:
                    raise ValueError(
                        "conflicting verdicts for the same canonical memory"
                    )
                verdicts[resolved] = value
            touched = {
                record["global_memory_id"]: record for record in records.values()
            }
            if feedback.answer_correct is False:
                produced = self.store.read_rows(
                    "SELECT m.record_json FROM memories m JOIN memory_scopes s "
                    "ON m.global_memory_id=s.global_memory_id "
                    "WHERE s.namespace=? AND s.batch_id=? AND s.layer='episodic' "
                    "AND json_extract(m.record_json, '$.task_id')=?",
                    (self.namespace, feedback.batch_id, feedback.task_id),
                )
                for row in produced:
                    output = json.loads(row[0])
                    output["unresolved_failure"] = True
                    touched[output["global_memory_id"]] = output
            for key, record in touched.items():
                if key in retrieved:
                    record["retrieval_count"] += 1
                verdict = verdicts.get(key)
                if key in adopted:
                    record["reuse_count"] += 1
                    record["last_reused_at"] = _now()
                    if verdict is True and feedback.answer_correct is True:
                        record["reuse_success_count"] += 1
                    elif verdict is False:
                        record["reuse_failure_count"] += 1
                    else:
                        record["reuse_unknown_count"] = (
                            int(record.get("reuse_unknown_count", 0)) + 1
                        )
                    if feedback.answer_correct is False and verdict is not True:
                        record["unresolved_failure"] = True
                if verdict is not None:
                    success = verdict is True and feedback.answer_correct is True
                    record["verified_success_count"] = int(
                        record.get("verified_success_count", 0)
                    ) + int(success)
                    record["verified_failure_count"] = int(
                        record.get("verified_failure_count", 0)
                    ) + int(verdict is False)
                    record.setdefault("verification_history", []).append(
                        {
                            "id": feedback.batch_id + ":" + feedback.task_id,
                            "success": success,
                            "failure": verdict is False,
                            "validation_source": feedback.validation_source,
                        }
                    )
                    if success:
                        record["unresolved_failure"] = False
                if record["memory_layer"] == "experience":
                    status = review_status(record, policy)
                    # A candidate is promoted only by finalize_batch.
                    record["review_status"] = (
                        "candidate"
                        if record["review_status"] != "active" and status == "active"
                        else status
                    )
                elif verdict is False or record.get("unresolved_failure"):
                    record["review_status"] = "pending"
                elif verdict is True:
                    record["review_status"] = "active"
                record["verified_confidence"] = (
                    int(record.get("verified_success_count", 0)) + 1.0
                ) / (
                    int(record.get("verified_success_count", 0))
                    + int(record.get("verified_failure_count", 0))
                    + 2.0
                )
                record["updated_at"] = _now()
                record["record_version"] += 1
                self.store.put(self.namespace, record)
            payload = asdict(feedback)
            payload.update(
                {
                    "retrieval_ids": list(feedback.retrieval_ids),
                    "retrieved_ids": sorted(retrieved),
                    "adopted_ids": sorted(adopted),
                    "memory_verdicts": verdicts,
                }
            )
            with self.store.transaction() as connection:
                connection.execute(
                    "INSERT INTO memory_feedback VALUES (?, ?, ?, ?)",
                    (
                        self.namespace,
                        feedback.batch_id,
                        feedback.task_id,
                        json_text(payload),
                    ),
                )
            return {
                "batch_id": feedback.batch_id,
                "task_id": feedback.task_id,
                "retrieval_ids": list(feedback.retrieval_ids),
                "retrieved_count": len(retrieved),
                "adopted_count": len(adopted),
            }

        return self._operation(
            feedback.batch_id, "feedback:" + feedback.task_id, asdict(feedback), action
        )

    def commit_task_memory(self, commit: TaskMemoryCommit) -> dict[str, Any]:
        """Atomically persist one task's memory outputs and optional feedback."""

        request = asdict(commit)

        def action() -> dict[str, Any]:
            episode_receipt: WriteReceipt | None = None
            experience_receipt: WriteReceipt | None = None
            feedback_receipt: dict[str, Any] | None = None
            if commit.episode is not None:
                episode_receipt = self._write(
                    commit.batch_id,
                    "episodic",
                    commit.task_id,
                    [commit.episode],
                    commit.operation_id + ":episode",
                )
            if commit.experiences:
                experience_receipt = self._write(
                    commit.batch_id,
                    "experience",
                    commit.task_id,
                    commit.experiences,
                    commit.operation_id + ":experience",
                )
            if commit.feedback is not None:
                feedback_receipt = self.record_feedback(commit.feedback)
            return {
                "batch_id": commit.batch_id,
                "task_id": commit.task_id,
                "operation_id": commit.operation_id,
                "episode": episode_receipt,
                "experiences": experience_receipt,
                "feedback": feedback_receipt,
            }

        return self._operation(
            commit.batch_id,
            commit.operation_id,
            {"kind": "task_memory_commit", **request},
            action,
        )

    def review_experience(self, review: ExperienceReview) -> dict[str, Any]:
        """Record an explicit tool, human, or offline review of experience cards."""

        for memory_id in review.memory_verdicts:
            record = self.get_memory(review.batch_id, memory_id)
            if record["memory_layer"] != "experience":
                raise ValueError("experience review only accepts experience memories")
        receipt = self.record_feedback(
            TaskFeedback(
                batch_id=review.batch_id,
                task_id="review:" + review.review_id,
                answer_correct=True,
                memory_verdicts=review.memory_verdicts,
                validation_source=review.validation_source,
                details=review.details,
            )
        )
        return {**receipt, "review_id": review.review_id}

    def _maintenance_plan(
        self, batch_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        batch = self._open_batch(batch_id)
        return plan_maintenance(
            self.store.scoped_records(self.namespace, batch_id),
            batch_id=batch_id,
            policy=BatchPolicy(**json.loads(batch["policy_json"])),
            encoder=self.encoder,
            merge_verifier=self.merge_verifier,
        )

    def preview_maintenance(self, batch_id: str) -> dict[str, Any]:
        with self.store.transaction(write=False):
            _, report = self._maintenance_plan(batch_id)
            return report

    def finalize_batch(self, batch_id: str) -> dict[str, Any]:
        with self.store.transaction() as connection:
            batch = self.store.batch(self.namespace, batch_id)
            if batch["status"] == "closed":
                return json.loads(batch["report_json"])
            retained, report = self._maintenance_plan(batch_id)
            originals = self.store.scoped_records(self.namespace, batch_id)
            by_id = {item["global_memory_id"]: item for item in originals}
            # Keep the complete source chain in a non-searchable provenance
            # archive before temporary layers and duplicate cards are removed.
            refs = {
                ref
                for item in retained
                if item["memory_layer"] == "experience"
                for ref in (
                    *item.get("source_memory_ids", []),
                    *item.get("knowledge_refs", []),
                )
            }
            pending = list(refs)
            archived: set[str] = set()
            while pending:
                ref = pending.pop()
                if ref in archived:
                    continue
                source = by_id.get(ref)
                if source is None:
                    continue
                archived.add(ref)
                snapshot = {
                    field: source.get(field)
                    for field in (
                        "global_memory_id",
                        "memory_id",
                        "task_id",
                        "origin_batch_id",
                        "source_agent",
                        "created_at",
                        "memory_layer",
                        "summary",
                        "evidence_chain",
                        "knowledge_refs",
                        "source_memory_ids",
                    )
                }
                connection.execute(
                    "INSERT OR REPLACE INTO memory_provenance VALUES (?, ?, ?)",
                    (self.namespace, ref, json_text(snapshot)),
                )
                pending.extend(
                    (
                        *source.get("knowledge_refs", []),
                        *source.get("source_memory_ids", []),
                    )
                )
            kept_ids = {item["global_memory_id"] for item in retained}
            for item in retained:
                normalized = normalize_memory_record(item)
                old = by_id.get(item["global_memory_id"], {})
                vector = None
                if self.encoder is not None and normalized[
                    "search_document"
                ] != old.get("search_document"):
                    encoded = self.encoder.encode([normalized["search_document"]])
                    if len(encoded) != 1:
                        raise ValueError("encoder returned an unexpected batch size")
                    vector = checked_vector(encoded[0])
                self.store.put(
                    self.namespace,
                    normalized,
                    vector=vector,
                    model_id=self.encoder.model_id if self.encoder else "",
                )
            for memory_id in by_id.keys() - kept_ids:
                self.store.delete(memory_id)
            for merge in report["merged"]:
                target = merge["canonical_id"]
                for alias in merge["merged_ids"]:
                    connection.execute(
                        "UPDATE memory_aliases SET canonical_id=? WHERE namespace=? AND canonical_id=?",
                        (target, self.namespace, alias),
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO memory_aliases VALUES (?, ?, ?)",
                        (self.namespace, alias, target),
                    )
                    connection.execute(
                        """INSERT OR IGNORE INTO memory_index
                           SELECT i.global_memory_id,i.index_type,?,i.weight FROM memory_index i
                           JOIN memory_scopes s ON s.global_memory_id=i.global_memory_id
                           WHERE s.namespace=? AND i.index_key=?
                           AND i.index_type IN ('source_reference','knowledge_reference')""",
                        (target, self.namespace, alias),
                    )
                    connection.execute(
                        """DELETE FROM memory_index WHERE index_key=?
                           AND index_type IN ('source_reference','knowledge_reference')
                           AND global_memory_id IN (SELECT global_memory_id FROM memory_scopes WHERE namespace=?)""",
                        (alias, self.namespace),
                    )
            report["archived_source_ids"] = sorted(archived)
            report["completed_at"] = _now()
            connection.execute(
                "UPDATE memory_batches SET status='closed',report_json=? WHERE namespace=? AND batch_id=?",
                (json_text(report), self.namespace, batch_id),
            )
            return report

    def abort_batch(self, batch_id: str, *, reason: str) -> dict[str, Any]:
        with self.store.transaction() as connection:
            batch = self.store.batch(self.namespace, batch_id)
            if batch["status"] == "aborted":
                return json.loads(batch["report_json"])
            self._open_batch(batch_id)
            ids = [
                item["global_memory_id"]
                for item in self.store.scoped_records(self.namespace, batch_id)
                if item["batch_id"] == batch_id
            ]
            for memory_id in ids:
                self.store.delete(memory_id)
            report = {
                "batch_id": batch_id,
                "status": "aborted",
                "reason": reason,
                "cleared_ids": ids,
            }
            connection.execute(
                "UPDATE memory_batches SET status='aborted',report_json=? WHERE namespace=? AND batch_id=?",
                (json_text(report), self.namespace, batch_id),
            )
            return report

    def get_memory(self, batch_id: str, memory_id: str) -> dict[str, Any]:
        with self.store.transaction(write=False):
            self.store.batch(self.namespace, batch_id)
            return self.store.resolve(self.namespace, batch_id, memory_id)

    def supersede_memory(
        self,
        batch_id: str,
        memory_id: str,
        replacement: dict[str, Any],
        *,
        operation_id: str,
    ) -> dict[str, Any]:
        """Create a reviewed replacement without destroying the original record."""

        replacement_input = dict(replacement)

        def action() -> dict[str, Any]:
            old = self.store.resolve(self.namespace, batch_id, memory_id)
            layer = old["memory_layer"]
            version = int(old.get("record_version", 1)) + 1
            raw = {
                "memory_id": f"{old['memory_id']}-v{version}",
                "source_agent": replacement_input.get(
                    "source_agent", old["source_agent"]
                ),
                "task_topic": replacement_input.get("task_topic", old["task_topic"]),
                "summary": replacement_input.get("summary", old["summary"]),
                "tags": replacement_input.get("tags", old.get("tags", [])),
                "evidence": replacement_input.get("evidence", old.get("evidence", [])),
                "entities": replacement_input.get("entities", old.get("entities", [])),
                "entity_aliases": replacement_input.get(
                    "entity_aliases", old.get("entity_aliases", {})
                ),
                "knowledge_refs": replacement_input.get(
                    "knowledge_refs", old.get("knowledge_refs", [])
                ),
                **replacement_input,
            }
            if layer == "experience":
                raw.setdefault("problem_pattern", old.get("problem_pattern", ""))
                raw.setdefault("solution_pattern", old.get("solution_pattern", ""))
                raw.setdefault("trigger_conditions", old.get("trigger_conditions", []))
                raw.setdefault("failure_pattern", old.get("failure_pattern", ""))
                raw.setdefault(
                    "source_memory_ids",
                    [old["global_memory_id"], *old.get("source_memory_ids", [])],
                )
            task_id = "revision:" + old["global_memory_id"][-12:]
            new = self._prepare(batch_id, layer, task_id, raw)
            new["record_version"] = version
            new["supersedes_memory_id"] = old["global_memory_id"]
            new["updated_at"] = _now()
            self._store_record(new)

            old["review_status"] = "superseded"
            old["lifecycle_stage"] = "deprecated"
            old["archive_scope"] = "review-only"
            old["superseded_by"] = new["global_memory_id"]
            old["updated_at"] = _now()
            old["record_version"] = version
            self._store_record(
                old, previous_search_document=old.get("search_document", "")
            )
            return {
                "batch_id": batch_id,
                "operation_id": operation_id,
                "superseded_id": old["global_memory_id"],
                "replacement_id": new["global_memory_id"],
                "memory_layer": layer,
                "record_version": version,
            }

        return self._operation(
            batch_id,
            operation_id,
            {
                "kind": "supersede_memory",
                "memory_id": memory_id,
                "replacement": replacement_input,
            },
            action,
        )

    def update_memory_relations(self, update: MemoryRelationUpdate) -> dict[str, Any]:
        """Add or remove provenance links while retaining a versioned audit trail."""

        request = asdict(update)

        def action() -> dict[str, Any]:
            record = self.store.resolve(
                self.namespace, update.batch_id, update.memory_id
            )
            old_document = str(record.get("search_document") or "")

            def canonical(values: tuple[str, ...]) -> set[str]:
                return {
                    self.store.resolve(self.namespace, update.batch_id, value)[
                        "global_memory_id"
                    ]
                    for value in values
                }

            source_ids = set(record.get("source_memory_ids", []))
            knowledge_refs = set(record.get("knowledge_refs", []))
            source_ids |= canonical(update.add_source_memory_ids)
            source_ids -= canonical(update.remove_source_memory_ids)
            knowledge_refs |= canonical(update.add_knowledge_refs)
            knowledge_refs -= canonical(update.remove_knowledge_refs)
            source_ids.discard(record["global_memory_id"])
            knowledge_refs.discard(record["global_memory_id"])
            if record["memory_layer"] == "experience" and not source_ids:
                raise ValueError("experience must retain at least one source memory")
            record["source_memory_ids"] = sorted(source_ids)
            record["knowledge_refs"] = sorted(knowledge_refs)
            record.setdefault("relation_history", []).append(
                {**request, "updated_at": _now()}
            )
            record["updated_at"] = _now()
            record["record_version"] = int(record.get("record_version", 1)) + 1
            record = normalize_memory_record(record)
            self._store_record(record, previous_search_document=old_document)
            return {
                "batch_id": update.batch_id,
                "operation_id": update.operation_id,
                "memory_id": record["global_memory_id"],
                "record_version": record["record_version"],
                "source_memory_ids": record["source_memory_ids"],
                "knowledge_refs": record["knowledge_refs"],
            }

        return self._operation(
            update.batch_id,
            update.operation_id,
            {"kind": "update_memory_relations", **request},
            action,
        )

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Return durable lifecycle state without reopening or mutating a batch."""

        with self.store.transaction(write=False):
            batch = self.store.batch(self.namespace, batch_id)
            counts = {}
            for table in (
                "memory_operations",
                "memory_retrievals",
                "memory_feedback",
            ):
                counts[table] = int(
                    self.store.read_rows(
                        f"SELECT COUNT(*) FROM {table} WHERE namespace=? AND batch_id=?",
                        (self.namespace, batch_id),
                    )[0][0]
                )
            return {
                key: batch[key]
                for key in (
                    "batch_id",
                    "task_family",
                    "dataset_id",
                    "group_id",
                    "status",
                    "created_at",
                )
            } | {
                "operation_count": counts["memory_operations"],
                "retrieval_count": counts["memory_retrievals"],
                "feedback_count": counts["memory_feedback"],
                "maintenance_report": (
                    json.loads(batch["report_json"])
                    if batch.get("report_json")
                    else None
                ),
            }

    def get_batch_memory_stats(self, batch_id: str) -> dict[str, Any]:
        """Aggregate memory-only retrieval and adoption metrics for one batch."""

        with self.store.transaction(write=False):
            batch = self.store.batch(self.namespace, batch_id)
            retrieval_rows = self.store.read_rows(
                "SELECT result_json FROM memory_retrievals "
                "WHERE namespace=? AND batch_id=? ORDER BY created_at",
                (self.namespace, batch_id),
            )
            feedback_rows = self.store.read_rows(
                "SELECT payload_json FROM memory_feedback "
                "WHERE namespace=? AND batch_id=? ORDER BY task_id",
                (self.namespace, batch_id),
            )
            retrievals = [json.loads(row[0]) for row in retrieval_rows]
            feedbacks = [json.loads(row[0]) for row in feedback_rows]
            returned_hits = sum(len(item.get("hits", [])) for item in retrievals)
            payload_bytes = sum(
                int(item.get("diagnostics", {}).get("payload_bytes", 0))
                for item in retrievals
            )
            retrieval_ms = sum(
                float(item.get("diagnostics", {}).get("total_ms", 0.0))
                for item in retrievals
            )
            retrieved = sum(len(item.get("retrieved_ids", [])) for item in feedbacks)
            adopted = sum(len(item.get("adopted_ids", [])) for item in feedbacks)
            correct_adoptions = 0
            for item in feedbacks:
                if item.get("answer_correct") is not True:
                    continue
                verdicts = item.get("memory_verdicts", {})
                correct_adoptions += sum(
                    verdicts.get(memory_id) is True
                    for memory_id in item.get("adopted_ids", [])
                )
            visible = self.store.scoped_records(self.namespace, batch_id)
            visible_by_layer = {
                layer: sum(item["memory_layer"] == layer for item in visible)
                for layer in ("knowledge", "episodic", "experience")
            }
            return {
                "batch_id": batch_id,
                "status": batch["status"],
                "retrieval_count": len(retrievals),
                "returned_hit_count": returned_hits,
                "feedback_count": len(feedbacks),
                "retrieved_count": retrieved,
                "adopted_count": adopted,
                "correct_adoption_count": correct_adoptions,
                "reuse_rate": adopted / retrieved if retrieved else 0.0,
                "correct_reuse_rate": (correct_adoptions / adopted if adopted else 0.0),
                "retrieval_payload_bytes": payload_bytes,
                "average_retrieval_payload_bytes": (
                    payload_bytes / len(retrievals) if retrievals else 0.0
                ),
                "retrieval_elapsed_ms": round(retrieval_ms, 3),
                "average_retrieval_ms": round(
                    retrieval_ms / len(retrievals) if retrievals else 0.0, 3
                ),
                "visible_by_layer": visible_by_layer,
            }

    def list_active_experiences(
        self, batch_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self.store.transaction(write=False):
            records = self.store.scoped_records(self.namespace, batch_id)
            return [
                item
                for item in records
                if item["memory_layer"] == "experience"
                and item["review_status"] == "active"
            ][:limit]

    def list_pending_reviews(
        self, batch_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self.store.transaction(write=False):
            records = self.store.scoped_records(self.namespace, batch_id)
            return [
                item
                for item in records
                if item["review_status"] in {"candidate", "pending"}
            ][:limit]

    def reindex_embeddings(self, batch_id: str) -> dict[str, Any]:
        if self.encoder is None:
            raise RuntimeError("reindex_embeddings requires an encoder")
        with self.store.transaction():
            self._open_batch(batch_id)
            records = self.store.scoped_records(self.namespace, batch_id)
            encoded = (
                self.encoder.encode([record["search_document"] for record in records])
                if records
                else []
            )
            if len(encoded) != len(records):
                raise ValueError("encoder returned an unexpected batch size")
            vectors = [checked_vector(vector) for vector in encoded]
            if len({len(vector) for vector in vectors}) > 1:
                raise ValueError("encoder returned inconsistent dimensions")
            for record, vector in zip(records, vectors):
                self.store.put(
                    self.namespace,
                    record,
                    vector=vector,
                    model_id=self.encoder.model_id,
                )
            return {"record_count": len(records), "model_id": self.encoder.model_id}

    def export_snapshot(self) -> dict[str, Any]:
        with self.store.transaction(write=False):
            records = [
                json.loads(row[0])
                for row in self.store.read_rows(
                    "SELECT m.record_json FROM memories m JOIN memory_scopes s "
                    "ON m.global_memory_id=s.global_memory_id WHERE s.namespace=? ORDER BY m.global_memory_id",
                    (self.namespace,),
                )
            ]
            tables = {}
            for table in (
                "memory_batches",
                "memory_operations",
                "memory_feedback",
                "memory_retrievals",
                "memory_aliases",
                "memory_provenance",
            ):
                tables[table] = [
                    dict(row)
                    for row in self.store.read_rows(
                        f"SELECT * FROM {table} WHERE namespace=? ORDER BY rowid",
                        (self.namespace,),
                    )
                ]
            vectors = [
                dict(row)
                for row in self.store.read_rows(
                    "SELECT v.* FROM memory_vectors v JOIN memory_scopes s "
                    "ON v.global_memory_id=s.global_memory_id WHERE s.namespace=? ORDER BY v.global_memory_id",
                    (self.namespace,),
                )
            ]
            return {
                "schema_version": 2,
                "namespace": self.namespace,
                "records": records,
                "tables": tables,
                "vectors": vectors,
            }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if (
            snapshot.get("schema_version") not in {1, 2}
            or snapshot.get("namespace") != self.namespace
        ):
            raise ValueError("snapshot schema or namespace does not match")
        columns = {
            "memory_batches": (
                "namespace",
                "batch_id",
                "task_family",
                "dataset_id",
                "group_id",
                "status",
                "policy_json",
                "created_at",
                "report_json",
            ),
            "memory_operations": (
                "namespace",
                "batch_id",
                "operation_id",
                "request_json",
                "receipt_json",
            ),
            "memory_feedback": ("namespace", "batch_id", "task_id", "payload_json"),
            "memory_retrievals": (
                "namespace",
                "batch_id",
                "retrieval_id",
                "request_json",
                "result_json",
                "created_at",
            ),
            "memory_aliases": ("namespace", "alias_id", "canonical_id"),
            "memory_provenance": ("namespace", "global_memory_id", "snapshot_json"),
        }
        with self.store.transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM memory_batches WHERE namespace=? LIMIT 1",
                (self.namespace,),
            ).fetchone():
                raise ValueError("restore requires an empty namespace")
            for table, fields in columns.items():
                for row in snapshot["tables"].get(table, []):
                    if row["namespace"] != self.namespace:
                        raise ValueError("snapshot contains another namespace")
                    connection.execute(
                        f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                        [row[field] for field in fields],
                    )
            vectors = {row["global_memory_id"]: row for row in snapshot["vectors"]}
            for record in snapshot["records"]:
                if self.store.get(record["global_memory_id"]) is not None:
                    raise ValueError(
                        "snapshot memory ID collides with an existing record"
                    )
                if record["memory_layer"] not in {
                    "knowledge",
                    "episodic",
                    "experience",
                }:
                    raise ValueError("snapshot contains an unknown memory layer")
                vector = vectors.get(record["global_memory_id"])
                self.store.put(
                    self.namespace,
                    record,
                    vector=(
                        checked_vector(json.loads(vector["vector_json"]))
                        if vector
                        else None
                    ),
                    model_id=str(vector["model_id"]) if vector else "",
                )
            return {
                "record_count": len(snapshot["records"]),
                "batch_count": len(snapshot["tables"]["memory_batches"]),
            }

    def get_evidence_path(self, batch_id: str, memory_id: str) -> dict[str, Any]:
        with self.store.transaction(write=False):
            root = self.get_memory(batch_id, memory_id)
            records = {root["global_memory_id"]: root}
            pending = [root]
            unresolved: set[str] = set()
            while pending:
                current = pending.pop()
                for ref in (
                    *current.get("source_memory_ids", []),
                    *current.get("knowledge_refs", []),
                ):
                    if ref in records or ref in unresolved:
                        continue
                    try:
                        source = self.store.resolve(self.namespace, batch_id, ref)
                    except KeyError:
                        rows = self.store.read_rows(
                            "SELECT snapshot_json FROM memory_provenance WHERE namespace=? AND global_memory_id=?",
                            (self.namespace, ref),
                        )
                        source = json.loads(rows[0][0]) if rows else None
                    if source is None:
                        unresolved.add(ref)
                    else:
                        records[ref] = source
                        pending.append(source)
            return {
                "root_memory_id": root["global_memory_id"],
                "memories": [
                    {
                        key: value
                        for key, value in record.items()
                        if key
                        in {
                            "global_memory_id",
                            "memory_id",
                            "memory_layer",
                            "summary",
                            "task_id",
                            "source_agent",
                            "created_at",
                            "evidence_chain",
                            "knowledge_refs",
                            "source_memory_ids",
                        }
                    }
                    for record in records.values()
                ],
                "unresolved_ids": sorted(unresolved),
                "structurally_complete": not unresolved
                and all(
                    record.get("evidence_chain", {}).get("valid", False)
                    for record in records.values()
                ),
                "factual_correctness": "requires_external_validation",
            }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> MemoryService:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
