"""Mappings between application language codes and provider-specific values."""

LANGUAGE_CONFIG = {
    "en": {
        "qdrant_target_lang": "eng_Latn",
        "sarvam_language_code": "en-IN",
    },
    "hi": {
        "qdrant_target_lang": "hin_Deva",
        "sarvam_language_code": "hi-IN",
    },
}


def get_application_language(qdrant_target_lang: str) -> str:
    """Return the application language code for a stored Qdrant target language."""
    if not isinstance(qdrant_target_lang, str) or not qdrant_target_lang.strip():
        raise ValueError("qdrant_target_lang must be a non-empty string")
    normalized_target = qdrant_target_lang.strip()
    for language, config in LANGUAGE_CONFIG.items():
        if config["qdrant_target_lang"] == normalized_target:
            return language
    raise ValueError(f"No application language mapping configured for Qdrant target language: {qdrant_target_lang}")


def get_qdrant_target_lang(dataset_language: str) -> str:
    """Return the target_lang value stored in Qdrant for a dataset language code."""
    if not isinstance(dataset_language, str) or not dataset_language.strip():
        raise ValueError("dataset language must be a non-empty string")
    normalized_language = dataset_language.strip().lower()
    try:
        return LANGUAGE_CONFIG[normalized_language]["qdrant_target_lang"]
    except KeyError as error:
        raise ValueError(f"No Qdrant target_lang mapping configured for language: {dataset_language}") from error


def get_sarvam_language_code(dataset_language: str) -> str:
    """Return the Sarvam BCP-47 language code for an application language code."""
    if not isinstance(dataset_language, str) or not dataset_language.strip():
        raise ValueError("dataset language must be a non-empty string")
    normalized_language = dataset_language.strip().lower()
    try:
        return LANGUAGE_CONFIG[normalized_language]["sarvam_language_code"]
    except KeyError as error:
        raise ValueError(f"No Sarvam language mapping configured for language: {dataset_language}") from error
