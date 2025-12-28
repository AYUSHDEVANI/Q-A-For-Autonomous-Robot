import logging
import os
import json
from fastapi import APIRouter, Request, Form, File, UploadFile, Depends
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.services.chat_service import answer_question
from app.services.audio_service import generate_audio_file, transcribe_audio_file, AUDIO_FOLDER
from app.utils.helpers import get_groq_client

router = APIRouter()
logger = logging.getLogger(__name__)
from pydantic import BaseModel

templates = Jinja2Templates(directory="app/templates")

class TTSRequest(BaseModel):
    text: str

class TranscribeResponse(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_url: str

def flash(request: Request, message: str, category: str = "primary"):
    if "_messages" not in request.session:
        request.session["_messages"] = []
    request.session["_messages"].append({"category": category, "message": message})

def get_flashed_messages(request: Request):
    if "_messages" in request.session:
        messages = request.session.pop("_messages")
        return [(m["category"], m["message"]) for m in messages]
    return []

@router.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.abspath(os.path.join(AUDIO_FOLDER, filename))
    return FileResponse(file_path)

@router.get("/")
async def index(request: Request):
    logger.info("Serving index page.")
    question = request.session.get('question')
    answer = request.session.get('answer')
    audio_filename = request.session.get('audio_filename')
    
    # Custom flash handling for Jinja
    messages = get_flashed_messages(request)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "question": question,
        "answer": answer,
        "audio_filename": audio_filename,
        "flashed_messages": messages
    })

@router.post("/")
async def ask(request: Request, question: str = Form(...)):
    logger.info(f"Received text question via POST: {question}")
    if not question:
        logger.warning("Empty question received.")
        flash(request, "Please enter a question.", "error")
        return RedirectResponse(url="/", status_code=303)

    logger.info("Processing question...")
    answer = answer_question(question)
    logger.info("Question processed successfully.")

    request.session['question'] = question
    request.session['answer'] = answer
    
    logger.info("Generating audio for answer...")
    audio_filename = await generate_audio_file(answer)
    if audio_filename:
        logger.info(f"Audio generated: {audio_filename}")
        request.session['audio_filename'] = audio_filename
    else:
        logger.warning("Audio generation failed or skipped (error checking).")
        request.session.pop('audio_filename', None)
    
    flash(request, "Question processed successfully!", "success")
    return RedirectResponse(url="/", status_code=303)

@router.post("/ask_audio")
async def ask_audio(request: Request, audio: UploadFile = File(...)):
    logger.info("Received audio upload request.")
    audio_data = await audio.read()

    if not audio_data:
        logger.warning("Empty audio upload.")
        flash(request, "Empty audio file.", "error")
        return RedirectResponse(url="/", status_code=303)

    try:
        logger.info(f"Transcribing audio ({len(audio_data)} bytes)...")
        client = get_groq_client()
        question = transcribe_audio_file(client, audio_data)
        logger.info(f"Transcription result: {question}")
        
        if not question:
            logger.warning("No text detected in audio.")
            flash(request, "No text detected in audio.", "error")
            return RedirectResponse(url="/", status_code=303)

        logger.info(f"Processing transcribed question: {question}")
        answer = answer_question(question)
        
        request.session['question'] = question
        request.session['answer'] = answer

        logger.info("Generating audio response...")
        audio_filename = await generate_audio_file(answer)
        if audio_filename:
            request.session['audio_filename'] = audio_filename
        else:
             request.session.pop('audio_filename', None)

        flash(request, "Audio question processed successfully!", "success")

    except Exception as e:
        logger.error(f"Error processing audio request: {str(e)}")
        flash(request, f"Error transcribing audio: {str(e)}", "error")

    return RedirectResponse(url="/", status_code=303)

from fastapi.responses import StreamingResponse
from app.services.chat_service import stream_answer

@router.get("/ask_stream")
async def ask_stream(request: Request, question: str):
    logger.info(f"Stream request received for: {question}")
    
    # Retrieve History from Session
    history = request.session.get("history", [])
    logger.info(f"Loaded history: {len(history)} items")
    
    async def event_generator():
        full_answer = ""
        # Pass history to service
        async for chunk in stream_answer(question, history):
            full_answer += chunk
            yield chunk
            
        # Update History after stream
        history.append((question, full_answer))
        # Keep last 2 turns (User requested specific limit)
        request.session["history"] = history[-2:]
        logger.info("History updated in session (kept last 2 turns).")
        
    return StreamingResponse(event_generator(), media_type="text/plain")

@router.post("/api/tts", response_model=TTSResponse)
async def api_tts(request: TTSRequest):
    logger.info(f"API TTS request for {len(request.text)} chars.")
    audio_filename = await generate_audio_file(request.text)
    if not audio_filename:
        # Fallback or error
        return JSONResponse(status_code=500, content={"message": "Audio generation failed"})
    
    return {"audio_url": f"/static/audio/{audio_filename}"}

@router.post("/api/transcribe", response_model=TranscribeResponse)
async def api_transcribe(audio: UploadFile = File(...)):
    logger.info("API Transcription request received.")
    audio_data = await audio.read()
    if not audio_data:
        logger.warning("Empty audio data.")
        return JSONResponse(status_code=400, content={"message": "Empty audio"})
    
    try:
        client = get_groq_client()
        text = transcribe_audio_file(client, audio_data)
        logger.info(f"Transcribed text: {text}")
        return {"text": text}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})
