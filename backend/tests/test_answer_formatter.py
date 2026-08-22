"""Focused regression tests for fail-closed deterministic answer formatting."""

import unittest

from app.rag.generation.answer_composer import AnswerEvidence, ComposedAnswer, NO_ANSWER_TEXT
from app.rag.generation.answer_formatter import format_answer, format_composed_answer


HINDI_DISTANCE_SOURCE = "फिलाडेल्फिया, पी.ए. से लैंकेस्टर, पी.ए. तक की दूरी 80 मील या 129 किमी है।"
ENGLISH_DISTANCE_SOURCE = "Distance from Philadelphia, PA to Lancaster, PA is 80 Miles or 129 Km."


class AnswerFormatterTests(unittest.TestCase):
    def test_hindi_distance_uses_query_direction_and_preserves_values(self) -> None:
        result = format_answer(
            "फिलाडेल्फिया लैंकेस्टर से कितनी दूर है?",
            HINDI_DISTANCE_SOURCE,
            HINDI_DISTANCE_SOURCE,
            "hi",
        )
        self.assertTrue(result.formatted)
        self.assertEqual(result.format_type, "distance")
        self.assertEqual(result.answer, "फिलाडेल्फिया लैंकेस्टर से लगभग 80 मील (129 किमी) दूर है।")

    def test_english_distance_uses_query_direction(self) -> None:
        result = format_answer(
            "How far is Lancaster from Philadelphia?",
            ENGLISH_DISTANCE_SOURCE,
            ENGLISH_DISTANCE_SOURCE,
            "en",
        )
        self.assertTrue(result.formatted)
        self.assertEqual(result.answer, "Lancaster is about 80 miles (129 km) from Philadelphia.")

    def test_reversed_distance_order_does_not_reverse_the_question(self) -> None:
        result = format_answer(
            "How far is Philadelphia from Lancaster?",
            ENGLISH_DISTANCE_SOURCE,
            ENGLISH_DISTANCE_SOURCE,
            "en",
        )
        self.assertEqual(result.answer, "Philadelphia is about 80 miles (129 km) from Lancaster.")

    def test_quantity_formats_only_one_unambiguous_value(self) -> None:
        source = "Mars has 2 moons."
        result = format_answer("How many moons does Mars have?", source, source, "en")
        self.assertTrue(result.formatted)
        self.assertEqual(result.format_type, "quantity")
        self.assertEqual(result.answer, "Mars has 2 moons.")

    def test_date_formats_only_one_supported_year(self) -> None:
        source = "Apollo 11 landing happened in 1969."
        result = format_answer("When did Apollo 11 landing happen?", source, source, "en")
        self.assertTrue(result.formatted)
        self.assertEqual(result.format_type, "date")
        self.assertEqual(result.answer, "Apollo 11 landing happened in 1969.")

    def test_location_formats_explicit_source_relation(self) -> None:
        source = "Paris is located in France."
        result = format_answer("Where is Paris?", source, source, "en")
        self.assertTrue(result.formatted)
        self.assertEqual(result.format_type, "location")
        self.assertEqual(result.answer, "Paris is in France.")

    def test_why_and_explanation_questions_remain_unchanged(self) -> None:
        source = "The journey takes longer because the route has heavy traffic."
        self.assertEqual(
            format_answer("Why is the journey slow?", source, source, "en").answer,
            source,
        )

    def test_ambiguous_distance_values_remain_unchanged(self) -> None:
        source = "Distance is 80 miles or 129 km, while another route is 100 miles."
        self.assertEqual(
            format_answer("How far is Philadelphia from Lancaster?", source, source, "en").answer,
            source,
        )

    def test_unsupported_pattern_and_malformed_input_remain_unchanged(self) -> None:
        source = "Philadelphia is a city in Pennsylvania."
        self.assertEqual(format_answer("Compare Philadelphia and Lancaster", source, source, "en").answer, source)
        self.assertEqual(format_answer(None, source, source, "en").answer, source)  # type: ignore[arg-type]

    def test_no_answer_is_not_formatted(self) -> None:
        original = ComposedAnswer(NO_ANSWER_TEXT, [], None, 0.1, True)
        self.assertIs(format_composed_answer("Where is Paris?", original, "en"), original)

    def test_formatting_preserves_evidence_and_provenance(self) -> None:
        evidence = AnswerEvidence(232017, 8, 0, 0.84, ENGLISH_DISTANCE_SOURCE)
        original = ComposedAnswer(ENGLISH_DISTANCE_SOURCE, [evidence], 0.84, 0.1, False)
        formatted = format_composed_answer("How far is Lancaster from Philadelphia?", original, "en")
        self.assertEqual(formatted.text, "Lancaster is about 80 miles (129 km) from Philadelphia.")
        self.assertEqual(formatted.evidence, original.evidence)
        self.assertEqual(formatted.confidence, original.confidence)
        self.assertEqual(formatted.latency_ms, original.latency_ms)


if __name__ == "__main__":
    unittest.main()
