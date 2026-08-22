import re


FILLER_WORDS_PATTERN = re.compile(r"\b(?:um|uh|hmm)\b", flags=re.IGNORECASE)


def normalize_transcript(transcript: str) -> str:
    """Produce a compact query by lowercasing and removing standalone fillers."""
    without_fillers = FILLER_WORDS_PATTERN.sub(" ", transcript.lower())
    return re.sub(r"\s+", " ", without_fillers).strip()
