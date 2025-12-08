export const API_BASE_URL = 'http://localhost:8000';

export async function streamAnswer(question, onChunk, onError) {
  try {
    const encodedQuestion = encodeURIComponent(question);
    const response = await fetch(`${API_BASE_URL}/ask_stream?question=${encodedQuestion}`);

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      onChunk(chunk);
    }
  } catch (error) {
    console.error("Streaming error:", error);
    onError(error.message);
  }
}

export async function generateTts(text) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) throw new Error("TTS Failed");
        return await response.json(); // { audio_url: ... }
    } catch (error) {
        console.error("TTS Error:", error);
        return null;
    }
}

export async function transcribeAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice_input.wav');

    const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
        method: 'POST',
        body: formData,
    });
    
    if (!response.ok) throw new Error("Transcription Failed");
    return await response.json(); // { text: "..." }
}

