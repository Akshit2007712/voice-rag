import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.rag.indexing.embedder import E5Embedder
from app.rag.indexing.vector_store import (
    QdrantSettings,
    VectorStore,
    build_strategy_aware_point_id,
    deterministic_point_id,
    payload_from_chunk,
    strategy_aware_point_id,
)
from app.rag.ingestion.chunker import Chunk


def make_chunk() -> Chunk:
    return Chunk(
        text="हिंदी परीक्षण पाठ",
        metadata={
            "query_id": 7,
            "query_type": "DESCRIPTION",
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
            "passage_index": 2,
            "is_selected": 1,
            "chunk_index": 0,
            "chunk_strategy": "whole_passage",
            "token_count": 4,
        },
    )


class FakeModel:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.eval_called = False

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        self.inputs.extend(texts)
        return np.tile(np.array([3.0, 4.0, 0.0], dtype=np.float32), (len(texts), 1))


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created = None
        self.points = {}

    def collection_exists(self, name):
        return self.created is not None

    def create_collection(self, collection_name, vectors_config):
        self.created = (collection_name, vectors_config)

    def delete_collection(self, name):
        self.created = None
        self.points.clear()

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=self.created[1].size))))

    def upsert(self, collection_name, points, wait):
        for point in points:
            self.points[str(point.id)] = point

    def query_points(self, **kwargs):
        result = SimpleNamespace(id="point", score=0.9, payload={"text": "result"})
        return SimpleNamespace(points=[result])

    def count(self, **kwargs):
        return SimpleNamespace(count=len(self.points))

    def close(self):
        pass


class IndexingTests(unittest.TestCase):
    def test_e5_prefixes_dimension_shape_and_normalization(self) -> None:
        model = FakeModel()
        embedder = E5Embedder(model=model, device="cpu")

        passage_vectors = embedder.embed_passages(["पाठ"])
        query_vector = embedder.embed_query("प्रश्न")

        self.assertEqual(model.inputs, ["passage: पाठ", "query: प्रश्न"])
        self.assertTrue(model.eval_called)
        self.assertEqual(embedder.dimension, 3)
        self.assertEqual(passage_vectors.shape, (1, 3))
        self.assertEqual(query_vector.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(passage_vectors[0])), 1.0)

    def test_payload_and_point_id_are_deterministic(self) -> None:
        chunk = make_chunk()
        self.assertEqual(deterministic_point_id(chunk), deterministic_point_id(chunk))
        self.assertEqual(payload_from_chunk(chunk)["text"], chunk.text)
        self.assertEqual(payload_from_chunk(chunk)["is_selected"], 1)

    def test_strategy_aware_point_id_is_deterministic_and_distinguishes_all_identity_fields(self) -> None:
        chunk = make_chunk()
        same_chunk = replace(chunk, metadata=dict(chunk.metadata))
        other_strategy = replace(chunk, metadata={**chunk.metadata, "chunk_strategy": "token_window_fallback"})
        other_chunk = replace(chunk, metadata={**chunk.metadata, "chunk_index": 1})
        other_passage = replace(chunk, metadata={**chunk.metadata, "passage_index": 3})
        other_query = replace(chunk, metadata={**chunk.metadata, "query_id": 8})

        self.assertEqual(strategy_aware_point_id(chunk), strategy_aware_point_id(same_chunk))
        self.assertNotEqual(strategy_aware_point_id(chunk), strategy_aware_point_id(other_strategy))
        self.assertNotEqual(strategy_aware_point_id(chunk), strategy_aware_point_id(other_chunk))
        self.assertNotEqual(strategy_aware_point_id(chunk), strategy_aware_point_id(other_passage))
        self.assertNotEqual(strategy_aware_point_id(chunk), strategy_aware_point_id(other_query))
        self.assertEqual(
            strategy_aware_point_id(chunk),
            build_strategy_aware_point_id(7, 2, 0, "whole_passage"),
        )

    def test_full_store_uses_strategy_aware_ids_while_legacy_store_keeps_legacy_ids(self) -> None:
        first = make_chunk()
        second = replace(first, metadata={**first.metadata, "chunk_strategy": "token_window_fallback"})
        vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        legacy_client = FakeQdrantClient()
        legacy_store = VectorStore(QdrantSettings(collection_name="msmarco_xi_hindi"), client=legacy_client)
        legacy_store.ensure_collection(3)
        legacy_store.upsert_chunks([first, second], vectors)
        self.assertEqual(len(legacy_client.points), 1)

        full_client = FakeQdrantClient()
        full_store = VectorStore(
            QdrantSettings(collection_name="msmarco_xi_hindi_full"),
            client=full_client,
            point_id_builder=strategy_aware_point_id,
        )
        full_store.ensure_collection(3)
        full_store.upsert_chunks([first, second], vectors)
        self.assertEqual(len(full_client.points), 2)

    def test_full_index_script_explicitly_uses_strategy_aware_builder(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "index_full_hindi.py"
        source = script.read_text(encoding="utf-8")
        self.assertIn("point_id_builder=strategy_aware_point_id", source)
        self.assertIn("POINT_ID_SCHEME=query_id + passage_index + chunk_index + chunk_strategy", source)

    def test_cosine_collection_small_batch_upsert_and_search(self) -> None:
        client = FakeQdrantClient()
        store = VectorStore(QdrantSettings(collection_name="test_hindi"), client=client)
        store.ensure_collection(3)
        self.assertEqual(client.created[1].size, 3)
        self.assertEqual(str(client.created[1].distance), "Cosine")

        chunk = make_chunk()
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        store.upsert_chunks([chunk], vectors)
        store.upsert_chunks([chunk], vectors)
        self.assertEqual(store.point_count(), 1)
        self.assertEqual(store.search(np.array([1.0, 0.0, 0.0])), [SimpleNamespace(id="point", score=0.9, payload={"text": "result"})])


if __name__ == "__main__":
    unittest.main()
