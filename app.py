import os
import logging
from pathlib import Path
from typing import Generator

import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indicvoice_care")

# Required environment variables
GNANI_API_KEY = os.getenv("GNANI_API_KEY")
PRISMA_ENDPOINT = os.getenv("PRISMA_ENDPOINT")  # Speech‑to‑Text endpoint
EVON_ENDPOINT = os.getenv("EVON_ENDPOINT")      # LLM reasoning endpoint
TIMBRE_ENDPOINT = os.getenv("TIMBRE_ENDPOINT")  # Text‑to‑Speech endpoint

if not all([GNANI_API_KEY, PRISMA_ENDPOINT, EVON_ENDPOINT, TIMBRE_ENDPOINT]):
    missing = [var for var, val in {
        "GNANI_API_KEY": GNANI_API_KEY,
        "PRISMA_ENDPOINT": PRISMA_ENDPOINT,
        "EVON_ENDPOINT": EVON_ENDPOINT,
        "TIMBRE_ENDPOINT": TIMBRE_ENDPOINT,
    }.items() if not val]
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# Directory to store uploaded audio and generated TTS files
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IndicVoice Care", version="0.1.0")

HEADERS = {"X-API-Key-ID": GNANI_API_KEY}

def call_prisma(audio_path: Path) -> str:
    """Send audio to Prisma STT service and return the transcript.
    Expected response JSON: {"transcript": "..."}
    """
    files = {"file": (audio_path.name, open(audio_path, "rb"), "audio/wav")}
    try:
        resp = requests.post(PRISMA_ENDPOINT, headers=HEADERS, files=files, timeout=30)
        resp.raise_for_status()
        return resp.json().get("transcript", "")
    except Exception as exc:
        logger.exception("Prisma STT request failed")
        raise HTTPException(status_code=502, detail="Speech‑to‑Text service error") from exc

def call_evon(transcript: str) -> str:
    """Send transcript to Evon LLM and obtain a response text.
    Expected request JSON: {"prompt": transcript}
    Expected response JSON: {"response": "..."}
    """
    payload = {"prompt": transcript}
    try:
        resp = requests.post(EVON_ENDPOINT, headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        logger.exception("Evon LLM request failed")
        raise HTTPException(status_code=502, detail="LLM reasoning service error") from exc

def call_timbre(text: str, output_path: Path) -> None:
    """Convert *text* to speech using Timbre and write the audio to *output_path*.
    Expected request JSON: {"text": text}
    The service returns raw audio bytes (WAV).
    """
    payload = {"text": text}
    try:
        resp = requests.post(TIMBRE_ENDPOINT, headers=HEADERS, json=payload, timeout=30, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as exc:
        logger.exception("Timbre TTS request failed")
        raise HTTPException(status_code=502, detail="Text‑to‑Speech service error") from exc

@app.post("/process", response_class=StreamingResponse)
async def process_audio(file: UploadFile = File(...)) -> StreamingResponse:
    """Accept an audio file, run STT → LLM → TTS, and stream back the generated audio.
    The original transcript and LLM response are added as response headers.
    """
    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Save uploaded audio to a temporary file
    input_path = OUTPUT_DIR / f"input_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    logger.info("Received file %s (size %d bytes)", input_path.name, input_path.stat().st_size)

    # 1️⃣ Speech‑to‑Text
    transcript = call_prisma(input_path)
    logger.info("Transcript: %s", transcript)

    # 2️⃣ LLM reasoning
    llm_response = call_evon(transcript)
    logger.info("LLM response: %s", llm_response)

    # 3️⃣ Text‑to‑Speech
    output_path = OUTPUT_DIR / f"output_{Path(file.filename).stem}.wav"
    call_timbre(llm_response, output_path)

    # Clean up the uploaded file
    try:
        input_path.unlink()
    except Exception:
        pass

    def audio_streamer() -> Generator[bytes, None, None]:
        with open(output_path, "rb") as audio_file:
            while chunk := audio_file.read(8192):
                yield chunk

    headers = {
        "X-Transcript": transcript,
        "X-LLM-Response": llm_response,
    }
    return StreamingResponse(audio_streamer(), media_type="audio/wav", headers=headers)
