from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from src.hybrid import hybrid_recommend
from src.recommender import load_songs, Song


load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="Signal & Sound", page_icon="♫", layout="wide")

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #090a0d; color: #f4f1ed; }
    [data-testid="stHeader"] { background: rgba(9,10,13,0.8); }
    [data-testid="stSidebar"] { background: #10131b; }
    .block-container { max-width: 1440px; padding: 2rem 3rem 3rem; }
    h1, h2, h3 { letter-spacing: -0.04em; }
    h1 { font-size: clamp(2.4rem, 5vw, 4.5rem) !important; line-height: 0.98 !important; }
    h2 { font-size: 2rem !important; }
    .brand { border-bottom: 1px solid #3a3c43; padding-bottom: 1rem; margin-bottom: 3rem; font-size: 1.45rem; font-weight: 800; }
    .brand span { color: #2e63ff; }
    .panel { border: 1px solid #2e63ff; border-radius: 10px; padding: 1.15rem 1.35rem; background: #10131b; }
    .result { border-top: 1px solid #3a3c43; padding: 1.15rem 0 1.25rem; }
    .result:first-child { border-top: 0; }
    .rank { color: #ff4ab2; font-size: 1.3rem; font-weight: 800; margin-right: .65rem; }
    .song-title { font-size: 1.35rem; font-weight: 750; }
    .meta { color: #b7b8bf; font-size: .9rem; margin-top: .3rem; }
    .why { margin-top: .6rem; line-height: 1.45; }
    .why strong { color: #ff4ab2; }
    .score { color: #2e63ff; font-size: 1.4rem; font-weight: 800; text-align: right; }
    .scorebar { background: #2b2e35; height: 9px; border-radius: 99px; overflow: hidden; margin-top: .45rem; }
    .scorebar div { background: #2e63ff; height: 100%; border-radius: 99px; }
    @media (max-width: 800px) { .block-container { padding: 1.25rem; } .brand { margin-bottom: 2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_catalog() -> list[Song]:
    records = load_songs(str(Path(__file__).parent / "data" / "songs.csv"))
    return [Song(**{key: _coerce_value(key, value) for key, value in record.items()}) for record in records]


def _coerce_value(key: str, value: str):
    if key == "id":
        return int(value)
    if key in {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}:
        return float(value)
    return value


st.markdown('<div class="brand">Signal <span>&amp;</span> Sound</div>', unsafe_allow_html=True)
left, right = st.columns([0.82, 1.18], gap="large")

with left:
    st.title("Find your next feeling")
    st.write("Describe what you are in the mood for and get ranked songs with clear, grounded explanations.")
    prompt = st.text_area(
        "Describe what you're in the mood for",
        value="Something heavy, dark, and energetic",
        height=125,
        label_visibility="visible",
    )
    recommend = st.button("Recommend", type="primary", use_container_width=True)

    if "response" not in st.session_state:
        st.session_state.response = hybrid_recommend(prompt, load_catalog(), k=5)
    if recommend:
        st.session_state.response = hybrid_recommend(prompt, load_catalog(), k=5)

    response = st.session_state.response
    st.subheader("Extracted preferences")
    prefs = response.preferences
    preference_rows = [
        ("Genre", prefs.genre or "Open"),
        ("Mood", prefs.mood or "Open"),
        ("Energy", f"{prefs.energy:.2f}" if prefs.energy is not None else "Open"),
        ("Acousticness", f"{prefs.acousticness:.2f}" if prefs.acousticness is not None else "Open"),
    ]
    st.markdown(
        '<div class="panel">' + "".join(
            f'<div style="display:flex;justify-content:space-between;margin:.55rem 0"><span style="color:#b7b8bf">{label}</span><strong>{value}</strong></div>'
            for label, value in preference_rows
        ) + "</div>",
        unsafe_allow_html=True,
    )

with right:
    st.header("Top matches")
    st.write("Ranked from the 18-song catalog by the deterministic preference engine.")
    if response.used_fallback:
        st.info("LLM unavailable, so local extraction and grounded template explanations are being used.")
    for index, recommendation in enumerate(response.recommendations, start=1):
        song = recommendation.song
        score = max(0.0, min(1.0, recommendation.score / 114.0))
        st.markdown(
            f'''<div class="result"><div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start"><div><span class="rank">{index}.</span><span class="song-title">{song["title"]}</span><div class="meta">{song["artist"]} &nbsp;·&nbsp; Genre: {song["genre"]} &nbsp;·&nbsp; Mood: {song["mood"]} &nbsp;·&nbsp; Energy: {float(song["energy"]):.2f} &nbsp;·&nbsp; Acousticness: {float(song["acousticness"]):.2f}</div></div><div class="score">{recommendation.score:.2f}<div class="scorebar"><div style="width:{score * 100:.0f}%"></div></div></div></div><div class="why"><strong>Why it fits:</strong> {recommendation.explanation}</div></div>''',
            unsafe_allow_html=True,
        )
