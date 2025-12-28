import os
import uuid
import logging
import edge_tts


logger = logging.getLogger(__name__)

# This path should be relative to the running application or absolute
# Since we moved static to app/static, and we run from root, it is app/static/audio
AUDIO_FOLDER = os.path.join('app', 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)

async def generate_audio_file(answer_text):
    if not answer_text or answer_text.startswith(("Error", "No PDFs")):
        logger.warning("Skipping audio generation due to error or empty response.")
        return None
    try:
        logger.info(f"Generating audio for text ({len(answer_text)} chars) using edge-tts.")
        # Voice: en-US-AriaNeural is a good default
        communicate = edge_tts.Communicate(answer_text, "en-US-AriaNeural")
        
        audio_filename = f"answer_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        
        await communicate.save(audio_path)
        
        logger.info(f"Audio file saved: {audio_path}")
        return audio_filename
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

def transcribe_audio_file(client, audio_data, model="whisper-large-v3"):
    """
    Transcribes audio using Groq client.
    """
    try:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=("audio.webm", audio_data, "audio/webm"),
            language="en",
            response_format="json"  # Change to json to get an object with .text, OR remove .text below. Let's use json for consistency.
        )
        return transcription.text.strip()
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise e
