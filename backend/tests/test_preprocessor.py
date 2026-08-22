import unittest

from app.rag.ingestion.preprocessor import preprocess_msmarco_xi_record


def make_record(
    translated_passages: object = None,
    is_selected: object = None,
) -> dict[str, object]:
    return {
        "query_id": 42,
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "Translated_passages": (
                translated_passages
                if translated_passages is not None
                else ["first passage", "second passage", "third passage"]
            ),
            "is_selected": is_selected if is_selected is not None else [1, 0, 1],
        },
    }


class PreprocessMSMarcoXIRecordTests(unittest.TestCase):
    def test_multiple_passages_produce_one_document_each(self) -> None:
        documents = preprocess_msmarco_xi_record(make_record())

        self.assertEqual(len(documents), 3)

    def test_passage_index_and_selection_remain_aligned(self) -> None:
        documents = preprocess_msmarco_xi_record(
            make_record(["one", "two", "three"], [0, 1, 0])
        )

        self.assertEqual([document.metadata["passage_index"] for document in documents], [0, 1, 2])
        self.assertEqual([document.metadata["is_selected"] for document in documents], [0, 1, 0])

    def test_record_metadata_is_preserved(self) -> None:
        document = preprocess_msmarco_xi_record(make_record())[0]

        self.assertEqual(
            document.metadata,
            {
                "query_id": 42,
                "query_type": "DESCRIPTION",
                "source_lang": "eng_Latn",
                "target_lang": "hin_Deva",
                "passage_index": 0,
                "is_selected": 1,
            },
        )

    def test_whitespace_is_normalized(self) -> None:
        document = preprocess_msmarco_xi_record(make_record(["  नमस्ते\n\t दुनिया  "], [1]))[0]

        self.assertEqual(document.text, "नमस्ते दुनिया")

    def test_empty_passages_are_skipped_without_changing_other_indexes(self) -> None:
        documents = preprocess_msmarco_xi_record(make_record(["one", " \t\n ", "three"], [1, 0, 1]))

        self.assertEqual([document.text for document in documents], ["one", "three"])
        self.assertEqual([document.metadata["passage_index"] for document in documents], [0, 2])
        self.assertEqual([document.metadata["is_selected"] for document in documents], [1, 1])

    def test_mismatched_passage_and_selection_lengths_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have the same length"):
            preprocess_msmarco_xi_record(make_record(["one", "two"], [1]))

    def test_missing_passages_returns_no_documents(self) -> None:
        record = make_record()
        record.pop("passages")

        self.assertEqual(preprocess_msmarco_xi_record(record), [])

    def test_malformed_passages_raises_clear_error(self) -> None:
        record = make_record()
        record["passages"] = "not a passage mapping"

        with self.assertRaisesRegex(ValueError, "passages must be a mapping"):
            preprocess_msmarco_xi_record(record)

    def test_missing_selection_list_raises_clear_error(self) -> None:
        record = make_record()
        record["passages"] = {"Translated_passages": ["one"]}

        with self.assertRaisesRegex(ValueError, "is_selected must be a list"):
            preprocess_msmarco_xi_record(record)


if __name__ == "__main__":
    unittest.main()
