"""Focused tests for deterministic extractive answer composition."""

import unittest

from app.rag.generation.answer_composer import AnswerComposer, NO_ANSWER_TEXT, split_sentences
from app.rag.retrieval.retriever import RetrievedChunk


def chunk(text: str, score: float = 0.8, **metadata: object) -> RetrievedChunk:
    return RetrievedChunk(score=score, text=text, metadata=dict(metadata))


class AnswerComposerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.composer = AnswerComposer()

    def test_empty_query_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.composer.compose("  ", [])

    def test_empty_results_return_no_answer(self) -> None:
        answer = self.composer.compose("सवाल", [])
        self.assertTrue(answer.is_no_answer)
        self.assertEqual(answer.text, NO_ANSWER_TEXT)

    def test_duplicate_sentences_removed(self) -> None:
        answer = self.composer.compose("दूरी", [chunk("दूरी बीस किलोमीटर है। दूरी बीस किलोमीटर है।", query_id=1)])
        self.assertEqual(answer.text, "दूरी बीस किलोमीटर है।")

    def test_higher_ranked_chunk_is_preferred(self) -> None:
        answer = self.composer.compose("जानकारी", [chunk("पहले स्रोत की जानकारी।", 0.9), chunk("दूसरे स्रोत की जानकारी।", 0.4)], max_sentences=1)
        self.assertEqual(answer.text, "पहले स्रोत की जानकारी।")

    def test_query_overlap_is_preferred(self) -> None:
        answer = self.composer.compose("फिलाडेल्फिया दूरी", [chunk("यह सामान्य वाक्य है। फिलाडेल्फिया की दूरी 110 किलोमीटर है।", 0.7)], max_sentences=1)
        self.assertEqual(answer.text, "फिलाडेल्फिया की दूरी 110 किलोमीटर है।")

    def test_max_sentences_respected_without_partial_sentence(self) -> None:
        answer = self.composer.compose("वाक्य", [chunk("पहला पूरा वाक्य है। दूसरा पूरा वाक्य है। तीसरा पूरा वाक्य है।")], max_sentences=2)
        self.assertEqual(len(answer.evidence), 2)
        self.assertNotIn("तीसरा", answer.text)
        self.assertTrue(answer.text.endswith("।"))

    def test_output_and_provenance_come_from_retrieved_text(self) -> None:
        source = "मूल प्रमाण वाक्य है।"
        answer = self.composer.compose("प्रमाण", [chunk(source, query_id=9, passage_index=2, chunk_index=1)])
        self.assertEqual(answer.text, source)
        evidence = answer.evidence[0]
        self.assertEqual((evidence.query_id, evidence.passage_index, evidence.chunk_index), (9, 2, 1))
        self.assertEqual(evidence.source_sentence, source)

    def test_low_score_and_empty_chunks_return_no_answer(self) -> None:
        answer = AnswerComposer(min_retrieval_score=0.5).compose("सवाल", [chunk("कम स्कोर।", 0.1), chunk("   ", 0.9)])
        self.assertTrue(answer.is_no_answer)

    def test_hindi_and_english_sentence_boundaries_work(self) -> None:
        self.assertEqual(split_sentences("पहला वाक्य है। दूसरा है!"), ["पहला वाक्य है।", "दूसरा है!"])
        self.assertEqual(split_sentences("First sentence. Second one?"), ["First sentence.", "Second one?"])

    def test_philadelphia_distance_sentence_outranks_generic_navigation(self) -> None:
        distance = "लैंकेस्टर और फिलाडेल्फिया के बीच सीधी रेखा में दूरी 57 मील या 91.71 किलोमीटर है।"
        answer = self.composer.compose("फिलाडेल्फिया लैंकेस्टर से कितनी दूर है", [
            chunk("ड्राइविंग दूरी, मानचित्र और यात्रा समय की जानकारी उपलब्ध है।", 0.93),
            chunk(distance, 0.925),
        ], max_sentences=1)
        self.assertFalse(answer.is_no_answer)
        self.assertEqual(answer.text, distance)
        self.assertIn("91.71", answer.text)

    def test_mars_query_with_nearest_but_unrelated_chunks_returns_no_answer(self) -> None:
        answer = self.composer.compose("मंगल ग्रह पर सबसे पहले कौन गया था?", [
            chunk("बैटमैन गोथम शहर का काल्पनिक नायक है।", 0.77),
            chunk("मारिजुआना पर स्थानीय कानून अलग-अलग हैं।", 0.76),
            chunk("लैंडफिल कचरे के निपटान की जगह है।", 0.75),
        ])
        self.assertTrue(answer.is_no_answer)
        self.assertEqual(answer.text, NO_ANSWER_TEXT)

    def test_multiple_decimal_values_remain_in_one_sentence(self) -> None:
        text = "Pi लगभग 3.14 है। दूरी 1.5 और गति 79.9 km है।"
        self.assertEqual(split_sentences(text), ["Pi लगभग 3.14 है।", "दूरी 1.5 और गति 79.9 km है।"])

    def test_hindi_abbreviations_remain_inside_the_same_sentence(self) -> None:
        text = "फिलाडेल्फिया, पी.ए. से लैंकेस्टर, पी.ए. तक की दूरी 80 मील है। अगला वाक्य है।"
        self.assertEqual(split_sentences(text), [
            "फिलाडेल्फिया, पी.ए. से लैंकेस्टर, पी.ए. तक की दूरी 80 मील है।",
            "अगला वाक्य है।",
        ])

    def test_english_abbreviations_and_initials_remain_inside_the_same_sentence(self) -> None:
        text = "Dr. J. R. Smith lives in the U.S. He travels to the U.K. often."
        self.assertEqual(split_sentences(text), [
            "Dr. J. R. Smith lives in the U.S.",
            "He travels to the U.K. often.",
        ])

    def test_direct_numeric_distance_answer_stops_irrelevant_padding_in_hindi(self) -> None:
        distance = "फिलाडेल्फिया, पी.ए. से लैंकेस्टर, पी.ए. तक की दूरी 80 मील या 129 किमी है।"
        answer = self.composer.compose("फिलाडेल्फिया लैंकेस्टर से कितनी दूर है?", [
            chunk(distance, 0.84, query_id=232017, passage_index=8, chunk_index=0),
            chunk("लैंकेस्टर की जनसंख्या और पारिवारिक जानकारी उपलब्ध है।", 0.95, query_id=232017, passage_index=3, chunk_index=0),
        ])
        self.assertEqual(answer.text, distance)
        self.assertEqual(len(answer.evidence), 1)
        self.assertEqual((answer.evidence[0].query_id, answer.evidence[0].passage_index, answer.evidence[0].chunk_index), (232017, 8, 0))

    def test_direct_numeric_distance_answer_stops_irrelevant_padding_in_english(self) -> None:
        distance = "Distance from Philadelphia, PA to Lancaster, PA is 80 Miles or 129 Km."
        answer = self.composer.compose("How far is Philadelphia from Lancaster?", [
            chunk(distance, 0.84, query_id=232017, passage_index=8, chunk_index=0),
            chunk("Lancaster population and family history are widely discussed.", 0.95, query_id=232017, passage_index=3, chunk_index=0),
        ])
        self.assertEqual(answer.text, distance)
        self.assertEqual(len(answer.evidence), 1)
        self.assertIn("80 Miles", answer.text)

    def test_direct_answer_outranks_weak_single_entity_overlap(self) -> None:
        answer = self.composer.compose("How far is Philadelphia from Lancaster?", [
            chunk("Philadelphia has a large population.", 0.99),
            chunk("Distance from Philadelphia to Lancaster is 80 miles.", 0.7),
        ], max_sentences=1)
        self.assertEqual(answer.text, "Distance from Philadelphia to Lancaster is 80 miles.")

    def test_direct_answer_selection_is_deterministic(self) -> None:
        chunks = [
            chunk("Distance from Philadelphia to Lancaster is 80 miles.", 0.8, query_id=9, passage_index=2, chunk_index=1),
            chunk("Philadelphia has a large population.", 0.99, query_id=9, passage_index=3, chunk_index=1),
        ]
        first = self.composer.compose("How far is Philadelphia from Lancaster?", chunks)
        second = self.composer.compose("How far is Philadelphia from Lancaster?", chunks)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.evidence, second.evidence)

    def test_strong_relevant_evidence_still_answers(self) -> None:
        source = "भारत की राजधानी नई दिल्ली है।"
        answer = self.composer.compose("भारत की राजधानी क्या है?", [chunk(source, 0.9)])
        self.assertFalse(answer.is_no_answer)
        self.assertEqual(answer.text, source)


if __name__ == "__main__":
    unittest.main()
