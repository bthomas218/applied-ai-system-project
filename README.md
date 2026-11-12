# Signal & Sound: Hybrid AI Music Recommender

## Original project

The original Modules 1–3 project was **Music Recommender Simulation**. Its goal
was to demonstrate an explainable content-based recommender using a small
catalog of 18 songs and weighted metadata such as genre, mood, energy, tempo,
valence, danceability, and acousticness. The original system accepted a
structured user profile, scored every song, sorted the results, and returned
the top recommendations with matching reasons.

## Title and summary

Signal & Sound turns a free-text request such as “Something heavy, dark, and
energetic” into ranked music recommendations. It combines an LLM for natural
language preference extraction and grounded explanations with the original
deterministic scoring engine for ranking.

This matters because the LLM makes the system easier to use without allowing
it to invent songs or silently change the ranking logic. Every recommendation
comes from the local catalog, has a deterministic score, and can be traced
back to song metadata and user preferences.

## Architecture overview

The full flow is shown in [`system_diagram.mmd`](system_diagram.mmd).

1. A user enters a natural-language request in the Streamlit interface.
2. The hybrid recommendation agent sends the request to the OpenAI-compatible
   LLM adapter, which returns structured preferences constrained by the
   catalog's genres and moods.
3. If the API key is missing, the request fails, or the response is invalid, a
   deterministic fallback parser extracts a limited local profile.
4. The existing weighted scorer evaluates all 18 songs and selects the top K.
5. The LLM receives only the selected song metadata when generating
   explanations. If it is unavailable, grounded template explanations are
   used instead.
6. The user sees ranked songs, scores, metadata, and explanations. Automated
   tests and human review check the recommendation behavior and explanation
   grounding.

Important boundary: the LLM explains the deterministic results. It does not
add tracks, invent metadata, modify scores, or rerank the catalog.

### Retrieval-augmented generation clarification

This project uses a lightweight, legitimate RAG pattern, but its retriever is
deterministic metadata retrieval rather than vector similarity search. The
system loads the complete `data/songs.csv` catalog, compares each song's
genre, mood, energy, acousticness, and other audio features against the
structured preferences, and retrieves the highest-scoring top-K songs.

Those retrieved song records and their deterministic scores are then supplied
as context to the explanation generator. The project does not create text
embeddings, use a vector database, perform nearest-neighbor search, or retrieve
tracks from an external corpus. This trade-off keeps the system transparent
and reproducible for the small catalog while giving up the semantic recall and
scalability that vector retrieval could provide for a much larger library.

## Setup instructions

The project requires Python 3.13 or newer.

### Install with uv

```bash
uv sync
```

If you prefer pip:

```bash
python -m pip install -r requirements.txt
```

### Configure the LLM

Create a local environment file from the example:

```bash
cp .env.example .env
```

Set your key in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
```

`.env` is ignored by Git. `.env.example` is safe to commit and contains no
secret value.

### Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, enter a preference description, and
select **Recommend**.

### Run the command-line simulation

```bash
uv run python src/main.py
```

### Run tests

```bash
uv run pytest
```

## Sample interactions

The examples below show the deterministic fallback mode, which works without
an API key. With a configured LLM, the extracted preferences and explanations
may be more nuanced, but the ranking still comes from the deterministic
engine.

### 1. Heavy, dark, and energetic

Input:

```text
Something heavy, dark, and energetic
```

Extracted fallback preferences:

```text
genre=metal, mood=aggressive, energy=0.90, acousticness=0.15
```

Top output:

```text
1. Iron Horizon — 85.93
   Iron Horizon by Black Meridian fits through the metal genre, an aggressive
   mood, energy at 0.97, acousticness at 0.04.
2. Storm Runner — 46.25
3. Gym Hero — 45.10
```

### 2. Lofi and chill for studying

Input:

```text
lofi chill calm music for studying
```

Extracted fallback preferences:

```text
genre=lofi, energy=0.35
```

Top output:

```text
1. Library Rain — 52.50
2. Focus Flow — 50.50
3. Midnight Coding — 49.70
```

The live LLM can additionally infer a low acousticness target when the user
explicitly asks for acoustic or unplugged music.

### 3. Happy pop for a workout

Input:

```text
pop happy energetic music for a workout
```

Extracted fallback preferences:

```text
genre=pop, energy=0.90
```

Top output:

```text
1. Gym Hero — 51.30
2. Sunrise City — 49.30
3. Storm Runner — 39.60
```

Each result is still drawn from `data/songs.csv`; no new track is generated by
the model.

## Design decisions

- **Deterministic ranking, LLM assistance:** The original weighted scorer is
  predictable and testable, while the LLM handles the parts users naturally
  express in prose.
- **Grounded explanations:** The explanation prompt receives retrieved song
  records only. This reduces hallucinated artists, tracks, and attributes.
- **Catalog-constrained extraction:** Genre and mood values are checked against
  labels present in the catalog. Unknown labels are discarded instead of
  creating unscorable categories.
- **Standard-library API client:** The project uses `urllib` for the API call
  instead of adding an OpenAI SDK. This keeps the integration small, but means
  the adapter has fewer conveniences and less typed error handling.
- **Offline fallback:** Missing credentials or API failures do not make the
  demo unusable. The fallback is intentionally simple and less expressive than
  the LLM.
- **Small catalog trade-off:** The 18-song dataset makes the scoring behavior
  easy to inspect, but it limits variety and can produce repetitive results.

## Testing summary

Implemented tests cover:

- deterministic sorting by weighted score
- non-empty recommendation explanations
- numeric acousticness scoring
- fallback preference extraction
- heavy/dark/energetic ranking behavior
- hybrid recommendations without an API key
- mocked LLM extraction and explanation responses

The direct smoke test and Python compilation checks pass. The full pytest suite
could not be executed in the development environment because pytest and the
project dependencies were not installed, and network access prevented `uv`
from downloading the locked packages. This also prevented browser-level
Streamlit verification in that environment.

The main lesson was that the LLM should improve the interface and explanation
quality without becoming the source of truth for ranking. Keeping the scorer
deterministic makes failures easier to diagnose and lets the application remain
functional when the model or network is unavailable.
