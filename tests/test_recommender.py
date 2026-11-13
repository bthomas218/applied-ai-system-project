from src.recommender import Song, UserProfile, Recommender, score_song
from src.hybrid import (
    OpenAICompatibleClient,
    fallback_preferences,
    hybrid_recommend,
)
from src.reliability import build_reliability_report, check_grounding

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_numeric_acousticness_preference_is_supported():
    score, reasons = score_song({"acousticness": 0.2}, {"acousticness": 0.2})
    assert score == 7.0
    assert reasons == ["acousticness similarity (+7.0)"]


def test_fallback_preferences_map_heavy_dark_energy_to_catalog():
    songs = make_small_recommender().songs
    prefs = fallback_preferences("Something heavy, dark, and energetic", songs)
    assert prefs.energy == 0.9
    assert prefs.acousticness == 0.15


def test_hybrid_fallback_prefers_heavy_catalog_match():
    songs = [
        Song(1, "Iron Horizon", "Black Meridian", "metal", "aggressive", 0.97, 168, 0.32, 0.55, 0.04),
        Song(2, "Night Drive Loop", "Neon Echo", "synthwave", "moody", 0.75, 110, 0.49, 0.73, 0.22),
    ]
    response = hybrid_recommend("Something heavy, dark, and energetic", songs, k=1, client=OpenAICompatibleClient(api_key=""))
    assert response.recommendations[0].song["title"] == "Iron Horizon"


def test_hybrid_recommend_works_without_api_key():
    songs = make_small_recommender().songs
    response = hybrid_recommend("Something energetic", songs, k=1, client=OpenAICompatibleClient(api_key=""))
    assert response.used_fallback is True
    assert len(response.recommendations) == 1
    assert response.recommendations[0].explanation


def test_hybrid_recommend_uses_mocked_llm_for_extraction_and_explanation():
    songs = make_small_recommender().songs

    class MockClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, system, user):
            self.calls += 1
            if self.calls == 1:
                return {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2}
            return {"explanations": ["Test Pop Track by Test Artist is a grounded explanation."]}

    client = MockClient()
    response = hybrid_recommend("pop and happy", songs, k=1, client=client)
    assert response.used_fallback is False
    assert response.preferences.genre == "pop"
    assert response.recommendations[0].explanation.startswith("Test Pop Track")
    assert response.reliability.level == "high"


def test_grounding_check_rejects_wrong_song():
    song = {"title": "Test Pop Track", "artist": "Test Artist"}
    check = check_grounding("This is a great match for another song.", song)
    assert check.passed is False


def test_reliability_report_marks_fallback_and_grounding():
    songs = make_small_recommender().songs
    response = hybrid_recommend("Something energetic", songs, k=1, client=OpenAICompatibleClient(api_key=""))
    report = build_reliability_report(
        response.preferences,
        response.recommendations,
        extraction_fallback=True,
        explanation_fallback=True,
    )
    assert 0 <= report.overall_confidence <= 1
    assert report.grounding_checks[0].passed is True
    assert "template explanations were used" in report.reasons
