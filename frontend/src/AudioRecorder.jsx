import { useState, useRef } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';
import { transcribeAudio } from './api';

export default function AudioRecorder({ onTranscription, onError }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        setIsProcessing(true);
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        
        try {
           const data = await transcribeAudio(audioBlob);
           if (data && data.text) {
               onTranscription(data.text);
           }
        } catch (err) {
            onError(err.message);
        } finally {
            setIsProcessing(false);
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
      onError("Could not access microphone.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  return (
    <div className="audio-recorder">
      <button 
        type="button"
        className={`record-btn ${isRecording ? 'recording' : ''}`}
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isProcessing}
        title={isRecording ? "Stop Recording" : "Start Recording"}
      >
        {isProcessing ? (
            <Loader2 className="spin" />
        ) : isRecording ? (
            <Square fill="currentColor" />
        ) : (
            <Mic />
        )}
      </button>
    </div>
  );
}
