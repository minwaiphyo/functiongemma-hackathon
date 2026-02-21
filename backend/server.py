import os
import sys
import time
import shutil
import tempfile
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load .env BEFORE anything reads env vars
load_dotenv()

# Import routing strategy and tools
from main import generate_hybrid, generate_cloud
from backend.tools import TOOLS, simulate_actions

app = FastAPI(title="DrivR API")

# Allow requests from the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_MODE = os.environ.get("MOCK_MODE", "true").lower() == "true"
print(f"[DrivR] MOCK_MODE={MOCK_MODE}")
print(f"[DrivR] GEMINI_API_KEY={'set' if os.environ.get('GEMINI_API_KEY') else 'MISSING!'}")

# Stubbed or actual Cactus models
transcription_model = None
if not MOCK_MODE:
    try:
        sys.path.insert(0, "cactus/python/src")
        from cactus import cactus_init, cactus_transcribe, cactus_destroy
        whisper_path = "cactus/weights/whisper-tiny-en"
        print(f"[DrivR] Loading Whisper model from {whisper_path}...")
        transcription_model = cactus_init(whisper_path)
    except Exception as e:
        print(f"[DrivR] Error loading whisper model: {e}")
        transcription_model = None


def _route(messages: list) -> dict:
    """Route a message through the AI pipeline and return result dict."""
    if MOCK_MODE:
        result = generate_cloud(messages, TOOLS)
        result["source"] = "cloud (mock mode)"
    else:
        result = generate_hybrid(messages, TOOLS)
        if "source" not in result:
            result["source"] = "on-device"
    return result


@app.post("/api/text-command")
async def text_command(payload: dict):
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    start_routing = time.time()

    try:
        messages = [{"role": "user", "content": text}]
        result = _route(messages)
    except Exception as e:
        traceback.print_exc()
        return {
            "error": str(e),
            "transcript": text,
            "function_calls": [],
            "actions": [],
            "source": None,
            "latency_ms": {"transcription": 0, "routing": 0, "total": 0},
        }

    routing_latency_ms = (time.time() - start_routing) * 1000
    actions = simulate_actions(result.get("function_calls", []))

    return {
        "transcript": text,
        "function_calls": result.get("function_calls", []),
        "actions": actions,
        "source": result.get("source", "unknown"),
        "latency_ms": {
            "transcription": 0,
            "routing": round(routing_latency_ms, 1),
            "total": round(routing_latency_ms, 1),
        },
    }


@app.post("/api/transcribe-and-act")
async def transcribe_and_act(audio: UploadFile = File(...)):
    # Save the audio file to a temp location
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        shutil.copyfileobj(audio.file, tmp)
        tmp.close()

        # --- Transcription ---
        start_transcription = time.time()
        if MOCK_MODE:
            transcript = "Set an alarm for 7 AM"  # stubbed
        else:
            if transcription_model:
                transcript = cactus_transcribe(transcription_model, tmp.name)
            else:
                transcript = "Could not load Whisper model"
        transcription_latency_ms = (time.time() - start_transcription) * 1000

        # --- Routing ---
        start_routing = time.time()
        messages = [{"role": "user", "content": transcript}]
        result = _route(messages)
        routing_latency_ms = (time.time() - start_routing) * 1000
        total_latency_ms = transcription_latency_ms + routing_latency_ms

        actions = simulate_actions(result.get("function_calls", []))

        return {
            "transcript": transcript,
            "function_calls": result.get("function_calls", []),
            "actions": actions,
            "source": result.get("source", "unknown"),
            "latency_ms": {
                "transcription": round(transcription_latency_ms, 1),
                "routing": round(routing_latency_ms, 1),
                "total": round(total_latency_ms, 1),
            },
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "error": "Could not understand audio",
            "transcript": None,
            "function_calls": [],
            "actions": [],
            "source": None,
            "latency_ms": {"transcription": 0, "routing": 0, "total": 0},
        }
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
