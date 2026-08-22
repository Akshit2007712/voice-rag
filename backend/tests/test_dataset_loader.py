import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.rag.ingestion.dataset_loader import (
    get_msmarco_xi_local_file_path,
    iter_msmarco_xi_records,
)
from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record


class _FakeBatch:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def to_pylist(self) -> list[dict[str, object]]:
        return self.records


class _FakeParquetFile:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.second_batch_requested = False
        self.closed = False

    def iter_batches(self, batch_size: int):
        self.batch_sizes.append(batch_size)
        yield _FakeBatch([{"record": 1}, {"record": 2}])
        self.second_batch_requested = True
        yield _FakeBatch([{"record": 3}])

    def close(self) -> None:
        self.closed = True


class LocalMSMarcoXILoaderTests(unittest.TestCase):
    def test_local_file_exists_and_opens_for_iteration(self) -> None:
        self.assertTrue(get_msmarco_xi_local_file_path().is_file())
        record = next(iter_msmarco_xi_records(batch_size=1))
        self.assertIsInstance(record, dict)

    def test_batch_size_is_passed_to_pyarrow(self) -> None:
        fake_file = _FakeParquetFile()
        with patch("app.rag.ingestion.dataset_loader.pq.ParquetFile", return_value=fake_file):
            self.assertEqual(list(iter_msmarco_xi_records(batch_size=7)), [{"record": 1}, {"record": 2}, {"record": 3}])

        self.assertEqual(fake_file.batch_sizes, [7])
        self.assertTrue(fake_file.closed)

    def test_iterator_does_not_advance_past_first_batch_for_first_record(self) -> None:
        fake_file = _FakeParquetFile()
        with patch("app.rag.ingestion.dataset_loader.pq.ParquetFile", return_value=fake_file):
            iterator = iter_msmarco_xi_records(batch_size=2)
            self.assertEqual(next(iterator), {"record": 1})

        self.assertFalse(fake_file.second_batch_requested)

    def test_record_reaches_existing_preprocessor(self) -> None:
        record = next(iter_msmarco_xi_records(batch_size=1))
        documents = preprocess_msmarco_xi_record(record)

        self.assertIsInstance(documents, list)

    def test_missing_local_file_raises_clear_error(self) -> None:
        with TemporaryDirectory() as temp_directory:
            with patch("app.rag.ingestion.dataset_loader.LOCAL_DATA_ROOT", Path(temp_directory)):
                with self.assertRaisesRegex(FileNotFoundError, "Local MSMARCO-XI Parquet file not found"):
                    next(iter_msmarco_xi_records())

    def test_invalid_language_and_split_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            next(iter_msmarco_xi_records(language="ta"))
        with self.assertRaisesRegex(ValueError, "Unsupported split"):
            next(iter_msmarco_xi_records(split="test"))

    def test_invalid_batch_size_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be at least 1"):
            next(iter_msmarco_xi_records(batch_size=0))


if __name__ == "__main__":
    unittest.main()
