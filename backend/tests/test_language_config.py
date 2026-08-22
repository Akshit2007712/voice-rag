"""Tests for application-to-provider language mappings."""

from unittest import TestCase

from app.rag.language_config import get_qdrant_target_lang, get_sarvam_language_code


class LanguageConfigTests(TestCase):
    def test_hindi_uses_distinct_sarvam_and_qdrant_codes(self) -> None:
        self.assertEqual(get_sarvam_language_code("hi"), "hi-IN")
        self.assertEqual(get_qdrant_target_lang("hi"), "hin_Deva")

    def test_unsupported_language_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_sarvam_language_code("ta")
