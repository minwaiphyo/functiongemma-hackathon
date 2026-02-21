import os
import sys
import time
import shutil
import tempfile
import traceback
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from elevenlabs.client import ElevenLabs

# Add parent directory to path so imports work from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

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
print(f"[DrivR] ELEVENLABS_API_KEY={'set' if os.environ.get('ELEVENLABS_API_KEY') else 'MISSING!'}")

# ElevenLabs TTS Client
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if ELEVENLABS_API_KEY:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
else:
    elevenlabs_client = None

# Cactus Models Setup
# Download on Mac with: cactus download openai/whisper-small
transcription_model = None

if not MOCK_MODE:
    try:
        sys.path.insert(0, "cactus/python/src")
        from cactus import cactus_init, cactus_transcribe, cactus_destroy
        
        # Speech-to-Text: openai/whisper-small
        # Why: Fast, accurate, Apple NPU support, 244M params vs 1.5B for medium
        # Download on Mac: cactus download openai/whisper-small
        stt_path = "weights/whisper-small"
        print(f"[DrivR] Initializing Speech-to-Text (Whisper Small)...")
        transcription_model = cactus_init(stt_path)
        print(f"[DrivR] ✓ STT loaded successfully")
    except Exception as e:
        print(f"[DrivR] ⚠ STT Error: {e}")
        print(f"[DrivR] On Mac, run: cactus download openai/whisper-small")
        transcription_model = None
    
    # Text-to-Speech: ElevenLabs API (cloud-based)
    if elevenlabs_client:
        print(f"[DrivR] ✓ TTS ready (ElevenLabs)")
    else:
        print(f"[DrivR] ⚠ TTS unavailable: Set ELEVENLABS_API_KEY environment variable")


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


def _generate_response_text(function_calls: list) -> str:
    """
    Generate a natural language response from function calls.
    This summarizes what actions will be taken.
    """
    if not function_calls:
        return "No actions needed."
    
    actions_summary = []
    for call in function_calls:
        name = call.get("name", "unknown")
        args = call.get("arguments", {})
        
        if name == "get_weather":
            actions_summary.append(f"Getting weather for {args.get('location', 'your location')}")
        elif name == "set_alarm":
            hour = args.get("hour", 0)
            minute = args.get("minute", 0)
            actions_summary.append(f"Setting alarm for {hour}:{minute:02d}")
        elif name == "send_message":
            recipient = args.get("recipient", "recipient")
            actions_summary.append(f"Sending message to {recipient}")
        elif name == "set_timer":
            minutes = args.get("minutes", 0)
            actions_summary.append(f"Setting timer for {minutes} minutes")
        elif name == "create_reminder":
            title = args.get("title", "reminder")
            time_str = args.get("time", "later")
            actions_summary.append(f"Creating reminder for {time_str}")
        elif name == "play_music":
            song = args.get("song", "song")
            actions_summary.append(f"Playing {song}")
        elif name == "search_contacts":
            query = args.get("query", "contact")
            actions_summary.append(f"Searching for {query}")
        else:
            actions_summary.append(f"Calling {name}")
    
    if len(actions_summary) == 1:
        return actions_summary[0] + "."
    else:
        return " and ".join(actions_summary) + "."


def _generate_audio_tts(text: str) -> str:
    """
    Generate audio from text using ElevenLabs TTS API.
    Returns base64-encoded MP3 audio data.
    """
    if MOCK_MODE or not elevenlabs_client:
        return ""
    
    try:
        # Generate speech using ElevenLabs (using default voice "Bella")
        # For faster responses, output format is currently MP3
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Bella: Clear, melodic, excellent for instructions
            output_format="mp3_22050",  # Smaller file size for faster transfer
        )
        
        # Convert audio stream to bytes and then to base64
        audio_bytes = b"".join(audio)
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"[TTS Error] {e}")
        return ""




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
            "response_text": None,
            "audio_base64": "",
            "function_calls": [],
            "actions": [],
            "source": None,
            "latency_ms": {"transcription": 0, "routing": 0, "total": 0},
        }

    routing_latency_ms = (time.time() - start_routing) * 1000
    function_calls = result.get("function_calls", [])
    actions = simulate_actions(function_calls)
    
    # Generate response text and audio
    response_text = _generate_response_text(function_calls)
    audio_base64 = _generate_audio_tts(response_text)

    return {
        "transcript": text,
        "response_text": response_text,
        "audio_base64": audio_base64,
        "function_calls": function_calls,
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

        function_calls = result.get("function_calls", [])
        actions = simulate_actions(function_calls)
        
        # Generate response text and audio
        response_text = _generate_response_text(function_calls)
        audio_base64 = _generate_audio_tts(response_text)

        return {
            "transcript": transcript,
            "response_text": response_text,
            "audio_base64": audio_base64,
            "function_calls": function_calls,
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
            "response_text": None,
            "audio_base64": "",
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
