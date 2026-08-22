"""Fail-closed presentation formatting for already-grounded composer output."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from app.rag.generation.answer_composer import ComposedAnswer


FormatType = Literal["distance", "quantity", "date", "location", "unchanged"]


@dataclass(frozen=True)
class FormattingResult:
    """One stateless formatting decision for an already-grounded answer."""

    answer: str
    formatted: bool
    format_type: FormatType


_EN_DISTANCE_FROM = re.compile(r"^how\s+far\s+is\s+(?P<subject>.+?)\s+from\s+(?P<reference>.+?)[?.!]*$", re.I)
_EN_DISTANCE_BETWEEN = re.compile(r"^what\s+is\s+the\s+distance\s+between\s+(?P<first>.+?)\s+and\s+(?P<second>.+?)[?.!]*$", re.I)
_HI_DISTANCE_FROM = re.compile(r"^(?P<subject>.+?)\s+(?P<reference>.+?)\s+से\s+कितनी\s+दूर\s+है[?।!]*$")
_HI_DISTANCE_BETWEEN = re.compile(r"^(?P<first>.+?)\s+और\s+(?P<second>.+?)\s+के\s+बीच\s+कितनी\s+दूरी\s+है[?।!]*$")
_DISTANCE_VALUE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>miles?|mi|kilometers?|kms?|km|मील|किमी|किलोमीटर)",
    re.I,
)
_EN_QUANTITY = re.compile(
    r"^how\s+many\s+(?P<concept>[a-z][a-z -]*?)\s+(?:does|do|is|are)\s+(?P<entity>.+?)(?:\s+have)?[?.!]*$",
    re.I,
)
_HI_QUANTITY = re.compile(r"^(?P<entity>.+?)\s+के\s+कितने\s+(?P<concept>.+?)\s+हैं[?।!]*$")
_EN_DATE = re.compile(r"^(?:when|what\s+year)\s+did\s+(?P<event>.+?)\s+happen[?.!]*$", re.I)
_HI_DATE = re.compile(r"^(?P<event>.+?)\s+कब\s+हुआ[?।!]*$")
_EN_LOCATION = re.compile(r"^where\s+is\s+(?P<entity>.+?)[?.!]*$", re.I)
_HI_LOCATION = re.compile(r"^(?P<entity>.+?)\s+कहाँ\s+है[?।!]*$")
_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _unchanged(answer: str) -> FormattingResult:
    return FormattingResult(answer=answer, formatted=False, format_type="unchanged")


def _source_contains(source: str, value: str) -> bool:
    """Require each query entity/concept token to occur in the supporting source."""
    tokens = re.findall(r"[A-Za-z0-9]+|[\u0900-\u097F]+", value.casefold())
    normalized_source = source.casefold()
    return bool(tokens) and all(token in normalized_source for token in tokens)


def _distance_values(source: str) -> tuple[tuple[str, str], tuple[str, str]] | None:
    """Accept exactly one miles and one kilometre value from the source sentence."""
    miles: list[tuple[str, str]] = []
    kilometres: list[tuple[str, str]] = []
    for match in _DISTANCE_VALUE.finditer(source):
        value, unit = match.group("value"), match.group("unit").casefold()
        if unit in {"mile", "miles", "mi", "मील"}:
            miles.append((value, unit))
        else:
            kilometres.append((value, unit))
    if len(miles) != 1 or len(kilometres) != 1:
        return None
    return miles[0], kilometres[0]


def _format_distance(query: str, source: str, language: str) -> FormattingResult | None:
    values = _distance_values(source)
    if values is None:
        return None
    miles, kilometres = values
    if language == "en":
        match = _EN_DISTANCE_FROM.fullmatch(query) or _EN_DISTANCE_BETWEEN.fullmatch(query)
        if match is None:
            return None
        subject = match.groupdict().get("subject") or match.groupdict().get("first")
        reference = match.groupdict().get("reference") or match.groupdict().get("second")
        if not subject or not reference or not _source_contains(source, subject) or not _source_contains(source, reference):
            return None
        return FormattingResult(
            f"{subject} is about {miles[0]} miles ({kilometres[0]} km) from {reference}.",
            True,
            "distance",
        )
    if language == "hi":
        match = _HI_DISTANCE_FROM.fullmatch(query) or _HI_DISTANCE_BETWEEN.fullmatch(query)
        if match is None:
            return None
        subject = match.groupdict().get("subject") or match.groupdict().get("first")
        reference = match.groupdict().get("reference") or match.groupdict().get("second")
        if not subject or not reference or not _source_contains(source, subject) or not _source_contains(source, reference):
            return None
        return FormattingResult(
            f"{subject} {reference} से लगभग {miles[0]} मील ({kilometres[0]} किमी) दूर है।",
            True,
            "distance",
        )
    return None


def _format_quantity(query: str, source: str, language: str) -> FormattingResult | None:
    match = _EN_QUANTITY.fullmatch(query) if language == "en" else _HI_QUANTITY.fullmatch(query) if language == "hi" else None
    if match is None:
        return None
    entity, concept = match.group("entity"), match.group("concept")
    if not _source_contains(source, entity) or not _source_contains(source, concept):
        return None
    numbers = re.findall(r"\d+(?:[.,]\d+)?", source)
    if len(numbers) != 1:
        return None
    if language == "en":
        return FormattingResult(f"{entity} has {numbers[0]} {concept}.", True, "quantity")
    return FormattingResult(f"{entity} के {numbers[0]} {concept} हैं।", True, "quantity")


def _format_date(query: str, source: str, language: str) -> FormattingResult | None:
    match = _EN_DATE.fullmatch(query) if language == "en" else _HI_DATE.fullmatch(query) if language == "hi" else None
    if match is None:
        return None
    event = match.group("event")
    years = sorted(set(_YEAR.findall(source)))
    if len(years) != 1 or not _source_contains(source, event):
        return None
    if language == "en":
        return FormattingResult(f"{event} happened in {years[0]}.", True, "date")
    return FormattingResult(f"{event} {years[0]} में हुआ।", True, "date")


def _format_location(query: str, source: str, language: str) -> FormattingResult | None:
    match = _EN_LOCATION.fullmatch(query) if language == "en" else _HI_LOCATION.fullmatch(query) if language == "hi" else None
    if match is None:
        return None
    entity = match.group("entity")
    if not _source_contains(source, entity):
        return None
    if language == "en":
        source_match = re.search(r"\b(?:is|was)\s+(?:located\s+)?in\s+(?P<location>[^.?!]+)", source, re.I)
        if source_match is None:
            return None
        return FormattingResult(f"{entity} is in {source_match.group('location').strip()}.", True, "location")
    source_match = re.search(
        rf"{re.escape(entity)}\s+(?P<location>[^।!?]+?)\s+में\s+स्थित\s+है",
        source,
    )
    if source_match is None:
        return None
    return FormattingResult(f"{entity} {source_match.group('location').strip()} में स्थित है।", True, "location")


def format_answer(query: str, answer: str, source_sentence: str, language: str) -> FormattingResult:
    """Format only a high-confidence single-source fact; otherwise return unchanged."""
    if not all(isinstance(value, str) for value in (query, answer, source_sentence, language)):
        return _unchanged(answer if isinstance(answer, str) else "")
    normalized_query, normalized_answer, normalized_source = _normalized(query), _normalized(answer), _normalized(source_sentence)
    if not normalized_query or not normalized_answer or not normalized_source or language not in {"hi", "en"}:
        return _unchanged(answer)
    try:
        for formatter in (_format_distance, _format_quantity, _format_date, _format_location):
            result = formatter(normalized_query, normalized_source, language)
            if result is not None:
                return result
    except (TypeError, ValueError, AttributeError):
        # Formatting is strictly optional and cannot invalidate grounded output.
        return _unchanged(answer)
    return _unchanged(answer)


def format_composed_answer(query: str, composed: ComposedAnswer, language: str) -> ComposedAnswer:
    """Change presentation only for single-source extractive facts; preserve LLM answers."""
    if not isinstance(composed, ComposedAnswer) or composed.is_no_answer or len(composed.evidence) != 1:
        return composed
    # If evidence chunk text doesn't match answer text (LLM generated answer), keep answer as-is
    source = composed.evidence[0].source_sentence
    if not any(token in source for token in composed.text.split()[:3]):
        return composed
    try:
        result = format_answer(query, composed.text, source, language)
        return replace(composed, text=result.answer) if result.formatted else composed
    except (AttributeError, TypeError, ValueError):
        return composed

