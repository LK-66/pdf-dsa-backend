from __future__ import annotations

import os
import uuid
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, ConfigDict


# ----------------------------
# Logging (must be before logger use)
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------
# Env
# ----------------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")


# ----------------------------
# MongoDB (optional)
# ----------------------------
mongo_url = os.getenv("MONGO_URL", "").strip()
db_name = os.getenv("DB_NAME", "").strip()

mongo_client: AsyncIOMotorClient | None = None
db = None

if mongo_url and db_name:
    try:
        mongo_client = AsyncIOMotorClient(mongo_url)
        db = mongo_client[db_name]
        logger.info("MongoDB connected")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        mongo_client = None
        db = None
else:
    logger.warning("MongoDB not configured (MONGO_URL/DB_NAME missing). Status endpoints will be disabled.")


# ----------------------------
# OpenAI client (official)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")


# ----------------------------
# Models
# ----------------------------
class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"
    speed: float = 1.0
    model: str = "tts-1"  # e.g. "tts-1" (standard)


class TTSResponse(BaseModel):
    audio_base64: str
    format: str = "mp3"


class OCRRequest(BaseModel):
    image_base64: str  # base64 of the image (no data: prefix)
    page_number: int = 1


class OCRResponse(BaseModel):
    text: str
    page_number: int
    success: bool


class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusCheckCreate(BaseModel):
    client_name: str


# ----------------------------
# Helpers
# ----------------------------
def require_openai_client() -> OpenAI:
    if openai_client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return openai_client


def clean_origins(raw: str) -> List[str]:
    # supports: "https://a.com,https://b.com"
    origins = [o.strip() for o in (raw or "").split(",")]
    # remove empties
    origins = [o for o in origins if o]
    return origins


# ----------------------------
# Routes
# ----------------------------
@api_router.get("/")
async def root():
    return {"message": "PDF DSA API - Text-to-Speech Ready"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    doc = status_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    if db is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check.get("timestamp"), str):
            check["timestamp"] = datetime.fromisoformat(check["timestamp"])
    return status_checks


@api_router.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Return audio as base64 (mp3)."""
    try:
        client = require_openai_client()

        if len(request.text) > 4096:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length of 4096 characters")
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        valid_voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
        if request.voice not in valid_voices:
            raise HTTPException(status_code=400, detail=f"Invalid voice. Must be one of: {', '.join(valid_voices)}")

        if request.speed < 0.25 or request.speed > 4.0:
            raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")

        audio = client.audio.speech.create(
            model=request.model,
            voice=request.voice,
            input=request.text,
            speed=request.speed,
            response_format="mp3",
        )

        audio_bytes = audio.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return TTSResponse(audio_base64=audio_base64, format="mp3")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")


@api_router.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    try:
        client = require_openai_client()

        audio = client.audio.speech.create(
            model=request.model,
            voice=request.voice,
            input=request.text,
            speed=request.speed,
        )

        audio_bytes = audio.read()

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/ocr", response_model=OCRResponse)
async def ocr_image(request: OCRRequest):
    """Extract text from an image (base64) using OpenAI vision-capable model."""
    try:
        client = require_openai_client()

        if not request.image_base64 or not request.image_base64.strip():
            raise HTTPException(status_code=400, detail="Image data is required")

        # You can switch model if you prefer:
        # model = "gpt-4o-mini" (cheap/fast) or "gpt-4o" (stronger)
        model = os.getenv("OPENAI_OCR_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

        # OpenAI Responses API expects a data URL for image input.
        data_url = f"data:image/png;base64,{request.image_base64}"

        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract ALL text from this image. Preserve line breaks and formatting. Output ONLY the extracted text."},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )

        # Extract text from response
        extracted = resp.output_text or ""
        return OCRResponse(text=extracted.strip(), page_number=request.page_number, success=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


# ----------------------------
# Middleware + Router
# ----------------------------
app.include_router(api_router)

cors_raw = os.getenv("CORS_ORIGINS", "*")
origins = clean_origins(cors_raw)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=origins if origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Shutdown
# ----------------------------
@app.on_event("shutdown")
async def shutdown_db_client():
    if mongo_client is not None:
        mongo_client.close()
