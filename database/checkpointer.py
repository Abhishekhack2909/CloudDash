"""Persistent LangGraph checkpointer backed by SQLAlchemy (SQLite/PostgreSQL)."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from database.connection import get_db_session, init_db
from database.models import CheckpointRecord, CheckpointWriteRecord


class PersistentCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpointer that persists graph state and writes across restarts."""

    def __init__(self):
        super().__init__()
        init_db()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch the latest or specified checkpoint tuple for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with get_db_session() as session:
            query = session.query(CheckpointRecord).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            )
            if checkpoint_id:
                row = query.filter_by(checkpoint_id=checkpoint_id).first()
            else:
                row = query.order_by(CheckpointRecord.created_at.desc(), CheckpointRecord.checkpoint_id.desc()).first()

            if not row:
                return None

            checkpoint = self.serde.loads_typed((row.type_name, row.checkpoint_blob))
            metadata = self.serde.loads_typed((row.metadata_type, row.metadata_blob))

            # Fetch writes
            writes_rows = (
                session.query(CheckpointWriteRecord)
                .filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=row.checkpoint_id,
                )
                .order_by(CheckpointWriteRecord.idx.asc())
                .all()
            )

            pending_writes = [
                (
                    w.task_id,
                    w.channel,
                    self.serde.loads_typed((w.type_name, w.value_blob)),
                )
                for w in writes_rows
            ]

            parent_config = None
            if row.parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.parent_checkpoint_id,
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row.checkpoint_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread ordered from newest to oldest."""
        if not config:
            return iter([])

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        with get_db_session() as session:
            query = session.query(CheckpointRecord).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            ).order_by(CheckpointRecord.created_at.desc(), CheckpointRecord.checkpoint_id.desc())

            if limit:
                query = query.limit(limit)

            rows = query.all()
            tuples = []
            for row in rows:
                cp = self.serde.loads_typed((row.type_name, row.checkpoint_blob))
                meta = self.serde.loads_typed((row.metadata_type, row.metadata_blob))
                parent_cfg = (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row.parent_checkpoint_id,
                        }
                    }
                    if row.parent_checkpoint_id
                    else None
                )
                tuples.append(
                    CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": row.checkpoint_id,
                            }
                        },
                        checkpoint=cp,
                        metadata=meta,
                        parent_config=parent_cfg,
                    )
                )
            return iter(tuples)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint into the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        cp_type, cp_blob = self.serde.dumps_typed(checkpoint)
        meta_type, meta_blob = self.serde.dumps_typed(metadata)

        with get_db_session() as session:
            existing = session.query(CheckpointRecord).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
            ).first()

            if existing:
                existing.parent_checkpoint_id = parent_id
                existing.type_name = cp_type
                existing.checkpoint_blob = cp_blob
                existing.metadata_type = meta_type
                existing.metadata_blob = meta_blob
            else:
                record = CheckpointRecord(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_id,
                    type_name=cp_type,
                    checkpoint_blob=cp_blob,
                    metadata_type=meta_type,
                    metadata_blob=meta_blob,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(record)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes for a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with get_db_session() as session:
            for idx, (channel, val) in enumerate(writes):
                val_type, val_blob = self.serde.dumps_typed(val)
                write_rec = session.query(CheckpointWriteRecord).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=idx,
                ).first()

                if write_rec:
                    write_rec.channel = channel
                    write_rec.type_name = val_type
                    write_rec.value_blob = val_blob
                else:
                    session.add(
                        CheckpointWriteRecord(
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            task_id=task_id,
                            idx=idx,
                            channel=channel,
                            type_name=val_type,
                            value_blob=val_blob,
                        )
                    )
