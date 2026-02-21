# Backend — DrivR API

FastAPI server that handles voice transcription, AI function-call routing, and action simulation.

## Setup

```bash
# From the project root
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r backend/requirements.txt
```

## Environment Variables

Create a `.env` file in the **project root** (not inside `backend/`):

```env
MOCK_MODE=true
GEMINI_API_KEY=your_gemini_api_key_here
```

| Variable | Description |
|---|---|
| `MOCK_MODE` | `true` = stub transcription + use Gemini cloud only. `false` = use Cactus Whisper + hybrid routing. |
| `GEMINI_API_KEY` | Your [Google AI Studio](https://aistudio.google.com/apikey) API key. Required in both modes. |

## Run

```bash
# From the project root (not from backend/)
uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

Server runs at `http://127.0.0.1:8000`. Swagger docs at `http://127.0.0.1:8000/docs`.

## API Endpoints

### `POST /api/text-command`

Text-based command — skips transcription.

```json
// Request
{ "text": "What's the weather in Tokyo?" }

// Response
{
  "transcript": "What's the weather in Tokyo?",
  "function_calls": [{ "name": "get_weather", "arguments": { "location": "Tokyo" } }],
  "actions": [{ "tool": "get_weather", "summary": "🌤️ Checked weather in Tokyo (Mock 72°F)", "success": true }],
  "source": "cloud (mock mode)",
  "latency_ms": { "transcription": 0, "routing": 450.2, "total": 450.2 }
}
```

### `POST /api/transcribe-and-act`

Voice pipeline — audio in, actions out. Send `multipart/form-data` with field `audio` (WAV/WebM blob).

Response schema is the same as above, with `latency_ms.transcription > 0`.

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI app, endpoints, mock/real mode switching |
| `tools.py` | Tool registry (JSON schemas) + action simulator |
| `requirements.txt` | Python dependencies |
