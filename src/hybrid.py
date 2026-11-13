"""Hybrid preference extraction, deterministic ranking, and grounded explanations."""

from __future__ import annotations

import json
import logging
import os
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .recommender import Song, recommend_songs, song_to_dict
from .reliability import ReliabilityReport, build_reliability_report


logger = logging.getLogger(__name__)


@dataclass
class ExtractedPreferences:
    """LLM-compatible profile with optional continuous acousticness preference."""

    genre: Optional[str] = None
    mood: Optional[str] = None
    energy: Optional[float] = None
    acousticness: Optional[float] = None
    valence: Optional[float] = None
    tempo_bpm: Optional[float] = None
    danceability: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in {
                "genre": self.genre,
                "mood": self.mood,
                "energy": self.energy,
                "acousticness": self.acousticness,
                "valence": self.valence,
                "tempo_bpm": self.tempo_bpm,
                "danceability": self.danceability,
            }.items()
            if value is not None
        }


@dataclass
class Recommendation:
    song: Dict[str, Any]
    score: float
    reasons: str
    explanation: str = ""


@dataclass
class RecommendationResponse:
    preferences: ExtractedPreferences
    recommendations: List[Recommendation]
    used_fallback: bool
    warning: Optional[str] = None
    request_id: str = ""
    reliability: Optional[ReliabilityReport] = None


class OpenAICompatibleClient:
    """Minimal Chat Completions client using only Python's standard library."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if model is None else model
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        return parsed


def _catalog_values(songs: Sequence[Song], field: str) -> List[str]:
    return sorted({str(getattr(song, field)) for song in songs})


def _clamp_number(value: Any, minimum: float = 0.0, maximum: float = 1.0) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return None


def preferences_from_payload(payload: Dict[str, Any], songs: Sequence[Song]) -> ExtractedPreferences:
    genres = _catalog_values(songs, "genre")
    moods = _catalog_values(songs, "mood")
    genre = payload.get("genre") if payload.get("genre") in genres else None
    mood = payload.get("mood") if payload.get("mood") in moods else None
    tempo = payload.get("tempo_bpm")
    danceability = _clamp_number(payload.get("danceability"))
    try:
        tempo_value = float(tempo) if tempo is not None else None
    except (TypeError, ValueError):
        tempo_value = None
    return ExtractedPreferences(
        genre=genre,
        mood=mood,
        energy=_clamp_number(payload.get("energy")),
        acousticness=_clamp_number(payload.get("acousticness")),
        valence=_clamp_number(payload.get("valence")),
        tempo_bpm=tempo_value,
        danceability=danceability,
    )


def fallback_preferences(text: str, songs: Sequence[Song]) -> ExtractedPreferences:
    """Small offline parser for demos and tests when no LLM is configured."""
    lowered = text.lower()
    genre = next((value for value in _catalog_values(songs, "genre") if value in lowered), None)
    mood_aliases = {
        "dark": "aggressive",
        "heavy": "aggressive",
        "sad": "melancholic",
        "upbeat": "happy",
    }
    mood = next(
        (mood_aliases[key] for key in mood_aliases if key in lowered and mood_aliases[key] in _catalog_values(songs, "mood")),
        None,
    )
    if "heavy" in lowered and "metal" in _catalog_values(songs, "genre"):
        genre = "metal"
    if "energetic" in lowered or "high energy" in lowered or "driving" in lowered:
        energy = 0.9
    elif "calm" in lowered or "chill" in lowered:
        energy = 0.35
    else:
        energy = 0.6
    acousticness = 0.15 if any(word in lowered for word in ("electric", "heavy", "dark")) else None
    return ExtractedPreferences(genre=genre, mood=mood, energy=energy, acousticness=acousticness)


def extract_preferences(text: str, songs: Sequence[Song], client: Optional[OpenAICompatibleClient] = None) -> tuple[ExtractedPreferences, bool, Optional[str]]:
    client = client or OpenAICompatibleClient()
    try:
        payload = client.complete_json(
            "Extract music preferences as JSON only. Use only catalog labels for genre and mood. Numeric values must be within their natural range.",
            json.dumps(
                {
                    "request": text,
                    "allowed_genres": _catalog_values(songs, "genre"),
                    "allowed_moods": _catalog_values(songs, "mood"),
                    "schema": {
                        "genre": "string or null",
                        "mood": "string or null",
                        "energy": "number 0..1 or null",
                        "acousticness": "number 0..1 or null",
                        "valence": "number 0..1 or null",
                        "tempo_bpm": "number or null",
                        "danceability": "number 0..1 or null",
                    },
                }
            ),
        )
        return preferences_from_payload(payload, songs), False, None
    except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return fallback_preferences(text, songs), True, str(exc)


def _fallback_explanation(preferences: ExtractedPreferences, song: Dict[str, Any]) -> str:
    parts: List[str] = []
    if preferences.genre and song["genre"] == preferences.genre:
        parts.append(f"the {song['genre']} genre")
    if preferences.mood and song["mood"] == preferences.mood:
        parts.append(f"a {song['mood']} mood")
    if preferences.energy is not None:
        parts.append(f"energy at {float(song['energy']):.2f}")
    if preferences.acousticness is not None:
        parts.append(f"acousticness at {float(song['acousticness']):.2f}")
    detail = ", ".join(parts) if parts else "the closest available audio profile"
    return f"{song['title']} by {song['artist']} fits through {detail}."


def generate_explanations(
    preferences: ExtractedPreferences,
    recommendations: List[Recommendation],
    client: Optional[OpenAICompatibleClient] = None,
) -> tuple[List[str], bool, Optional[str]]:
    client = client or OpenAICompatibleClient()
    if not recommendations:
        return [], False, None
    try:
        payload = client.complete_json(
            "Write one concise grounded explanation per recommendation. Use only the supplied song metadata and preferences. Return JSON with an 'explanations' array in the same order.",
            json.dumps(
                {
                    "preferences": preferences.as_dict(),
                    "recommendations": [
                        {"song": rec.song, "score": round(rec.score, 2)}
                        for rec in recommendations
                    ],
                }
            ),
        )
        explanations = payload.get("explanations")
        if not isinstance(explanations, list) or len(explanations) != len(recommendations):
            raise ValueError("LLM explanation count did not match recommendations")
        if not all(isinstance(item, str) and item.strip() for item in explanations):
            raise ValueError("LLM returned an invalid explanation")
        return explanations, False, None
    except (RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return [_fallback_explanation(preferences, rec.song) for rec in recommendations], True, str(exc)


def hybrid_recommend(text: str, songs: Sequence[Song], k: int = 5, client: Optional[OpenAICompatibleClient] = None) -> RecommendationResponse:
    request_id = uuid.uuid4().hex[:12]
    logger.info("recommendation_started", extra={"event": "recommendation_started", "request_id": request_id})
    preferences, extraction_fallback, warning = extract_preferences(text, songs, client)
    if extraction_fallback:
        logger.warning("preference_extraction_fallback", extra={"event": "preference_extraction_fallback", "request_id": request_id})
    ranked = recommend_songs(preferences.as_dict(), [song_to_dict(song) for song in songs], k=k)
    recommendations = [Recommendation(song=song, score=score, reasons=reasons) for song, score, reasons in ranked]
    explanations, explanation_fallback, explanation_warning = generate_explanations(preferences, recommendations, client)
    for recommendation, explanation in zip(recommendations, explanations):
        recommendation.explanation = explanation
    warnings = [message for message in (warning, explanation_warning) if message]
    if explanation_fallback:
        logger.warning("explanation_fallback", extra={"event": "explanation_fallback", "request_id": request_id})
    reliability = build_reliability_report(
        preferences,
        recommendations,
        extraction_fallback=extraction_fallback,
        explanation_fallback=explanation_fallback,
    )
    logger.info(
        "recommendation_completed",
        extra={
            "event": "recommendation_completed",
            "request_id": request_id,
            "recommendation_count": len(recommendations),
            "used_fallback": extraction_fallback or explanation_fallback,
            "confidence": reliability.overall_confidence,
        },
    )
    return RecommendationResponse(
        preferences=preferences,
        recommendations=recommendations,
        used_fallback=extraction_fallback or explanation_fallback,
        warning="; ".join(warnings) if warnings else None,
        request_id=request_id,
        reliability=reliability,
    )
