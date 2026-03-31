import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from core.config.milvus_config import MILVUS_DB, MILVUS_DOCS_COLLECTION
from core.utils.file_utils import get_file_content_md5_hex
from core.utils.logger_utils import logger
from infra.milvus.client import get_client


@dataclass(frozen=True)
class MilvusDocumentSyncPlan:
    action: str
    reason: str
    doc_id: str
    doc_hash: str
    source: str
    existing_chunk_count: int = 0
    duplicate_doc_id: str | None = None

    @property
    def should_skip(self) -> bool:
        return self.action == "skip"

    def to_result(
            self,
            *,
            delete_result: dict | None = None,
            insert_result: dict | None = None,
    ) -> dict:
        result = {
            "action": self.action,
            "reason": self.reason,
            "doc_id": self.doc_id,
            "doc_hash": self.doc_hash,
            "source": self.source,
            "existing_chunk_count": self.existing_chunk_count,
            "duplicate_doc_id": self.duplicate_doc_id,
        }

        if delete_result is not None:
            result["delete_result"] = delete_result

        if insert_result is not None:
            result.update(insert_result)
            return result

        result.update({
            "insert_count": 0,
            "ids": [],
        })
        return result


class MilvusDocumentComparator:
    """
    Compare a local document against existing Milvus chunks and decide whether
    to skip, insert, or replace the current document.

    Rules:
        1. Same normalized source path + same content -> skip.
        2. Same normalized source path + different content -> replace.
        3. Different source path + same content -> skip.
        4. No matching path and no matching content -> insert.
    """

    output_fields = ["pk", "doc_id", "doc_hash", "source", "chunk_id"]

    @staticmethod
    def normalize_source_path(file_path: str | Path) -> str:
        return str(Path(file_path).resolve())

    @classmethod
    def build_doc_id(cls, file_path: str | Path) -> str:
        normalized_path = cls.normalize_source_path(file_path)
        return f"doc_{hashlib.md5(normalized_path.encode('utf-8')).hexdigest()}"

    @classmethod
    def build_doc_hash(cls, file_path: str | Path) -> str:
        normalized_path = cls.normalize_source_path(file_path)
        doc_hash = get_file_content_md5_hex(normalized_path)
        if not doc_hash:
            raise ValueError(f"Failed to build doc hash for file: {normalized_path}")
        return doc_hash

    def inspect_document(self, file_path: str | Path) -> MilvusDocumentSyncPlan:
        source = self.normalize_source_path(file_path)
        doc_id = self.build_doc_id(source)
        doc_hash = self.build_doc_hash(source)

        same_path_records = self._query_by_field("doc_id", doc_id)
        if same_path_records:
            existing_hashes = {
                str(record.get("doc_hash", "")).strip()
                for record in same_path_records
                if str(record.get("doc_hash", "")).strip()
            }
            if doc_hash in existing_hashes:
                return MilvusDocumentSyncPlan(
                    action="skip",
                    reason="same_path_same_content",
                    doc_id=doc_id,
                    doc_hash=doc_hash,
                    source=source,
                    existing_chunk_count=len(same_path_records),
                )
            return MilvusDocumentSyncPlan(
                action="replace",
                reason="same_path_content_changed",
                doc_id=doc_id,
                doc_hash=doc_hash,
                source=source,
                existing_chunk_count=len(same_path_records),
            )

        same_hash_records = self._query_by_field("doc_hash", doc_hash)
        duplicate_doc_ids = {
            str(record.get("doc_id", "")).strip()
            for record in same_hash_records
            if str(record.get("doc_id", "")).strip()
        }
        duplicate_doc_ids.discard(doc_id)

        if duplicate_doc_ids:
            duplicate_doc_id = sorted(duplicate_doc_ids)[0]
            return MilvusDocumentSyncPlan(
                action="skip",
                reason="different_path_same_content",
                doc_id=doc_id,
                doc_hash=doc_hash,
                source=source,
                existing_chunk_count=len(same_hash_records),
                duplicate_doc_id=duplicate_doc_id,
            )

        return MilvusDocumentSyncPlan(
            action="insert",
            reason="new_document",
            doc_id=doc_id,
            doc_hash=doc_hash,
            source=source,
        )

    def delete_document_chunks(self, doc_id: str) -> dict:
        client = self.get_client()
        result = client.delete(
            collection_name=MILVUS_DOCS_COLLECTION,
            filter=self._build_equals_filter("doc_id", doc_id),
        )
        logger.info(f"Delete Milvus document chunks: doc_id={doc_id}, result={result}")
        return result

    def _query_by_field(self, field_name: str, field_value: str) -> list[dict]:
        client = self.get_client()
        if not client.has_collection(MILVUS_DOCS_COLLECTION):
            return []
        return client.query(
            collection_name=MILVUS_DOCS_COLLECTION,
            filter=self._build_equals_filter(field_name, field_value),
            output_fields=self.output_fields,
            consistency_level="Strong",
        )

    @staticmethod
    def _build_equals_filter(field_name: str, field_value: str) -> str:
        return f'{field_name} == {json.dumps(field_value, ensure_ascii=False)}'

    @staticmethod
    def get_client():
        client = get_client()
        client.using_database(MILVUS_DB)
        return client
