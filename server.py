from openai import OpenAI
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (optional)
mongo_url = os.getenv("MONGO_URL", "").strip()
db_name = os.getenv("DB_NAME", "").strip()

client = None
db = None

if mongo_url and db_name:
    try:
        client = AsyncIOMotorClient(mongo_url)
        db = client[db_name]
        logger.info("MongoDB connected")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        client = None
        db = None
else:
    logger.warning("MongoDB not configured (MONGO_URL/DB_NAME missing). Status endpoints will be disabled.")

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TTS Models
class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"
    speed: float = 1.0
    model: str = "tts-1"

class TTSResponse(BaseModel):
    audio_base64: str
    format: str = "mp3"

# OCR Models
class OCRRequest(BaseModel):
    image_base64: str
    page_number: int = 1

class OCRResponse(BaseModel):
    text: str
    page_number: int
    success: bool

# Status Check Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

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
    doc['timestamp'] = doc['timestamp'].isoformat()
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
   if db is None:
    raise HTTPException(status_code=503, detail="Database not configured")
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks

@api_router.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using OpenAI TTS API"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        # Validate text length
        if len(request.text) > 4096:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length of 4096 characters")

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        valid_voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
        if request.voice not in valid_voices:
            raise HTTPException(status_code=400, detail=f"Invalid voice. Must be one of: {', '.join(valid_voices)}")

        if request.speed < 0.25 or request.speed > 4.0:
            raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")

        client = OpenAI(api_key=api_key)

        # OpenAI TTS (returns bytes)
        audio = client.audio.speech.create(
            model=request.model,
            voice=request.voice,
            input=request.text,
            speed=request.speed,
            format="mp3",
        )

        audio_bytes = audio.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return TTSResponse(audio_base64=audio_base64, format="mp3")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@api_router.post("/ocr", response_model=OCRResponse)
async def ocr_image(request: OCRRequest):
    """Extract text from image using OpenAI Vision"""
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
        
        api_key = os.getenv("EMERGENT_LLM_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key not configured")
        
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="Image data is required")
        
        chat = LlmChat(
            api_key=api_key,
            session_id=f"ocr-{uuid.uuid4()}",
            system_message="You are an OCR assistant. Extract ALL text from the image exactly as it appears, preserving line breaks and formatting. Only output the extracted text, nothing else."
        ).with_model("openai", "gpt-4o")
        
        image_content = ImageContent(image_base64=request.image_base64)
        
        user_message = UserMessage(
            text="Extract all text from this image. Preserve the original formatting and line breaks. Output only the extracted text.",
            file_contents=[image_content]
        )
        
        response = await chat.send_message(user_message)
        
        return OCRResponse(
            text=response.strip(),
            page_number=request.page_number,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@api_router.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """Stream audio directly for immediate playback"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

        if len(request.text) > 4096:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length of 4096 characters")

        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        client = OpenAI(api_key=api_key)

        audio = client.audio.speech.create(
            model=request.model,
            voice=request.voice,
            input=request.text,
            speed=request.speed,
            format="mp3",
        )

        audio_bytes = audio.read()

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-cache"
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS stream failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS stream failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    if client is not None:
        client.close()
