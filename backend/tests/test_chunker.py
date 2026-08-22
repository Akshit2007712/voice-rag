import unittest

from app.rag.ingestion.chunker import chunk_retrieval_document
from app.rag.ingestion.preprocessor import RetrievalDocument


class FakeTokenizer:
    """Whitespace tokenizer used to test chunking logic without model downloads."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_ids = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            token_ids.append(self._token_to_id[token])
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


def make_document(text: str, is_selected: int = 1) -> RetrievalDocument:
    return RetrievalDocument(
        text=text,
        metadata={
            "query_id": 1,
            "query_type": "DESCRIPTION",
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
            "passage_index": 2,
            "is_selected": is_selected,
        },
    )


class ChunkRetrievalDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FakeTokenizer()

    def test_short_document_is_one_whole_passage_chunk(self) -> None:
        chunks = chunk_retrieval_document(make_document("छोटा passage."), max_tokens=256, tokenizer=self.tokenizer)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["chunk_strategy"], "whole_passage")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_parent_metadata_is_preserved(self) -> None:
        chunk = chunk_retrieval_document(make_document("one two."), tokenizer=self.tokenizer)[0]

        self.assertEqual(chunk.metadata["query_id"], 1)
        self.assertEqual(chunk.metadata["passage_index"], 2)
        self.assertEqual(chunk.metadata["is_selected"], 1)

    def test_long_multi_sentence_document_uses_sentence_overlap(self) -> None:
        document = make_document("one two. three four. five six.")
        chunks = chunk_retrieval_document(document, max_tokens=4, tokenizer=self.tokenizer)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["chunk_strategy"] == "sentence_overlap" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["token_count"] <= 4 for chunk in chunks))
        self.assertIn("three four.", chunks[0].text)
        self.assertIn("three four.", chunks[1].text)

    def test_very_long_single_sentence_uses_token_window_fallback(self) -> None:
        chunks = chunk_retrieval_document(make_document("one two three four five six seven"), max_tokens=3, tokenizer=self.tokenizer)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["chunk_strategy"] == "token_window_fallback" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["token_count"] <= 3 for chunk in chunks))

    def test_overlap_is_dropped_when_it_would_exceed_the_budget(self) -> None:
        document = make_document("one two three. four five six. seven eight nine.")
        chunks = chunk_retrieval_document(document, max_tokens=5, tokenizer=self.tokenizer)

        self.assertEqual([chunk.text for chunk in chunks], ["one two three.", "four five six.", "seven eight nine."])
        self.assertTrue(all(chunk.metadata["token_count"] <= 5 for chunk in chunks))

    def test_extremely_long_valid_document_never_raises_for_needing_more_chunks(self) -> None:
        document = make_document(" ".join(f"word{index}" for index in range(600)))
        chunks = chunk_retrieval_document(document, max_tokens=256, tokenizer=self.tokenizer)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["token_count"] <= 256 for chunk in chunks))

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_retrieval_document(make_document(" \t\n "), tokenizer=self.tokenizer), [])

    def test_invalid_max_tokens_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_tokens must be at least 1"):
            chunk_retrieval_document(make_document("text"), max_tokens=0, tokenizer=self.tokenizer)

    def test_is_selected_does_not_affect_chunk_text_or_strategy(self) -> None:
        selected_chunks = chunk_retrieval_document(make_document("one two three four five", 1), max_tokens=3, tokenizer=self.tokenizer)
        unselected_chunks = chunk_retrieval_document(make_document("one two three four five", 0), max_tokens=3, tokenizer=self.tokenizer)

        self.assertEqual([chunk.text for chunk in selected_chunks], [chunk.text for chunk in unselected_chunks])
        self.assertEqual(
            [chunk.metadata["chunk_strategy"] for chunk in selected_chunks],
            [chunk.metadata["chunk_strategy"] for chunk in unselected_chunks],
        )
        self.assertTrue(all(chunk.metadata["is_selected"] == 1 for chunk in selected_chunks))
        self.assertTrue(all(chunk.metadata["is_selected"] == 0 for chunk in unselected_chunks))


if __name__ == "__main__":
    unittest.main()
