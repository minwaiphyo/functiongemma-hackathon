# 🎙️ DrivR — Build Plan

**Time:** ~2.5 hours | **Team:** Dev A (Backend/FastAPI) + Dev B (Frontend/React)

---

## Concept

Speak a command → transcribed locally (Cactus Whisper) → FunctionGemma routes the tool call on-device → action is **simulated** (no real side effects). Cloud fallback via Gemini only when needed.

> **⚠️ Current state:** Both `cactus_transcribe` and `generate_hybrid` are **stubbed** for Windows development. Transcription returns a sample command, routing uses `generate_cloud()` (Gemini) only. On Mac, swap to real Cactus by setting `MOCK_MODE=false`. All actions are always simulated — the backend never performs real actions (no actual alarms, messages, etc.), just returns confirmation strings.

---

## Tech Stack

| Layer | Tech | Owner |
|-------|------|-------|
| Frontend | **Vite + React** (fastest scaffold, no SSR overhead) | Dev B |
| Backend | **FastAPI** + uvicorn | Dev A |
| Local AI | Cactus (FunctionGemma + Whisper) | Dev A |
| Cloud AI | Gemini 2.0 Flash | Dev A |

---

## Folder Structure

```
functiongemma-hackathon/
├── main.py                # existing — generate_hybrid (DO NOT change signature)
├── benchmark.py           # existing
├── submit.py              # existing
├── PLAN.md
│
├── backend/               # ← Dev A owns everything here
│   ├── server.py          # FastAPI app
│   └── tools.py           # Tool registry + action simulator
│
└── frontend/              # ← Dev B owns everything here
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.jsx         # Main app component
        ├── App.css         # Styles
        └── main.jsx        # Entry point
```

---

## Architecture

```
┌──────────────────────────────────────┐
│         REACT APP (Dev B)            │
│                                      │
│  MicButton → records WAV             │
│  TextInput → typed commands          │
│  ActionCard → shows results          │
│  HistoryFeed → past commands         │
└──────────────┬───────────────────────┘
               │ HTTP (fetch)
               ▼
┌──────────────────────────────────────┐
│       FASTAPI SERVER (Dev A)         │
│                                      │
│  1. Transcribe  (cactus / stub)      │
│  2. Route       (generate_hybrid)    │
│  3. Simulate    (tools.py)           │
└──────────────────────────────────────┘
```

---

## API Contract

Both devs build to this. This is the only shared dependency.

---

### `POST /api/transcribe-and-act`

Voice pipeline — audio in, actions out.

**Request:** `multipart/form-data`, field `audio` (WAV blob)

**Response:**
```json
{
  "transcript": "Set an alarm for 7 AM",
  "function_calls": [
    { "name": "set_alarm", "arguments": { "hour": 7, "minute": 0 } }
  ],
  "actions": [
    { "tool": "set_alarm", "summary": "⏰ Alarm set for 7:00 AM", "success": true }
  ],
  "source": "on-device | cloud (fallback) | cloud (mock mode)",
  "latency_ms": {
    "transcription": 120.5,
    "routing": 45.3,
    "total": 165.8
  }
}
```

---

### `POST /api/text-command`

Text fallback — skip transcription.

**Request:** `application/json`
```json
{ "text": "What's the weather in Tokyo?" }
```

**Response:** Same schema. `transcript` = input text, `latency_ms.transcription` = 0.

---

### Error Response

```json
{
  "error": "Could not understand audio",
  "transcript": null,
  "function_calls": [],
  "actions": [],
  "source": null,
  "latency_ms": { "transcription": 0, "routing": 0, "total": 0 }
}
```

---

## Dev A — Backend (FastAPI)

**Owner:** Priyansh
**Files:** `backend/server.py`, `backend/tools.py`
**Deps:** `fastapi`, `uvicorn`, `python-multipart`

### tools.py

Tool registry (same dict format as benchmark.py):
- `get_weather(location)`, `set_alarm(hour, minute)`, `send_message(recipient, message)`
- `play_music(song)`, `set_timer(minutes)`, `create_reminder(title, time)`
- `search_contacts(query)`, `open_app(app_name)`, `make_call(contact)`

Action simulator function: takes function name + args → returns emoji + summary string. No real side effects.

### server.py

**Mock mode** (env var `MOCK_MODE=true`, default on Windows):
- `transcribe()` → returns random sample command string (stubbed)
- `route()` → uses `generate_cloud()` from `main.py` (`generate_hybrid` is **stubbed out** — skipped entirely in mock mode)

**Real mode** (`MOCK_MODE=false`, on Mac):
- `transcribe()` → uses `cactus_transcribe()` with Whisper model (loaded once at startup)
- `route()` → uses `generate_hybrid()` from `main.py`

**Actions are always simulated** — the backend never performs real actions. It only returns a friendly confirmation string (e.g. "⏰ Alarm set for 7:00 AM"). No alarms, messages, or calls are actually made.

**Endpoints:**
- `POST /api/transcribe-and-act` → save WAV to temp → transcribe → route → simulate → JSON
- `POST /api/text-command` → route → simulate → JSON

**CORS:** Allow all origins (React dev server runs on different port).

### Run command
```bash
# Windows
set MOCK_MODE=true && set GEMINI_API_KEY=your-key
python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000

# Mac
export MOCK_MODE=false GEMINI_API_KEY=your-key
python -m uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
```

---

## Dev B — Frontend (Vite + React)

**Owner:** Teammate
**Dir:** `frontend/`
**Deps:** None beyond Vite + React

### Setup
```bash
cd frontend
npx -y create-vite@latest ./ --template react
npm install
npm run dev
```
Dev server on `http://localhost:5173`, proxy API calls to `http://localhost:8000`.

### Components (minimalistic)

**App.jsx** — single page with 4 sections:

1. **Header** — title "DrivR", tagline "Your personal driving assistant."
2. **MicButton** — large circular button, tap to start/stop recording
   - Uses `navigator.mediaDevices.getUserMedia` + `MediaRecorder`
   - Converts blob to WAV, sends to `POST /api/transcribe-and-act`
   - Visual: pulse animation while recording, idle state otherwise
3. **ActionCard** — displays latest result
   - Shows: emoji + summary, source badge (🟢 local / ☁️ cloud), latency in ms
   - Animates in on new result
4. **HistoryFeed** — scrollable list of past commands + results
5. **TextInput** — input bar at bottom, sends to `POST /api/text-command`

### Design

- Dark theme, minimalistic, clean sans-serif font
- Mobile-first layout (demo from phone = "driver mode")
- Keep it simple — no complex state management, just `useState`/`useRef`

### Vite proxy config (`vite.config.js`)
```js
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

### Run command
```bash
cd frontend
npm run dev
```

---

## Timeline

| Time | Phase | Dev A (Backend) | Dev B (Frontend) |
|------|-------|-----------------|------------------|
| 0:00–0:15 | Scaffold | FastAPI + tools.py stubs | `create-vite`, component skeletons |
| 0:15–1:00 | Core | Pipeline: transcribe→route→simulate, mock mode | Mic recording, API calls, result display |
| 1:00–1:20 | 🔄 Sync | Integration test together | Integration test together |
| 1:20–1:50 | Polish | Error handling, logging | Animations, mobile, error states |
| 1:50–2:30 | Demo | Full run-through, phone test | Full run-through, phone test |

### Sync points
- **0:15** — Backend serves `/api/text-command`, frontend loads
- **1:00** — Full integration: text first, then voice
- **1:50** — Demo dry run

---

## Hackathon Scoring

| Criterion | Coverage |
|---|---|
| Functionality | Working voice→action demo |
| Hybrid Architecture | Uses `generate_hybrid()` — local-first + cloud fallback |
| Agentic Capability | Voice → transcribe → tool select → execute chain |
| Theme (Local-First) | Whisper + FunctionGemma on-device, offline-capable |
| Rubric 3 (Voice) | Low-latency voice-to-action via `cactus_transcribe` |
