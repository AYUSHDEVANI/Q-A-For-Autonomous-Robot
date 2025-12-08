import { useState, useRef, useEffect } from 'react';
import { Send, RefreshCw, Volume2, StopCircle, User, Bot, Sparkles } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';
import { streamAnswer, generateTts, API_BASE_URL } from './api';
import AudioRecorder from './AudioRecorder';
import './index.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: "Hello! I'm LabBot 🤖. Ask me anything about our lab projects!" }
  ]);
  const [question, setQuestion] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const messagesEndRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle Text Submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || isStreaming) return;
    
    stopAudio();
    const userMsg = { role: 'user', content: question };
    setMessages(prev => [...prev, userMsg]);
    setQuestion('');
    
    await processQuestion(question);
  };

  // Reusable Process Logic
  const processQuestion = async (text) => {
    setError(null);
    setAudioUrl(null);
    setIsStreaming(true);

    // If coming from audio, we need to add the user message first if not already added
    // Ideally AudioRecorder would pass the text back, and we handle it.
    // Since we reuse this for both text/audio, let's assume 'text' is what we want to send.
    // Note: If calling from AudioRecorder, we might need to manually update messages list there too.
    
    // Add temporary empty assistant message to stream into
    setMessages(prev => [...prev, { role: 'assistant', content: '', isStreaming: true }]);

    let fullAnswer = "";
    
    await streamAnswer(
      text,
      (chunk) => {
          fullAnswer += chunk;
          setMessages(prev => {
            const newMsgs = [...prev];
            const lastMsg = newMsgs[newMsgs.length - 1];
            if (lastMsg.role === 'assistant') {
                lastMsg.content = fullAnswer;
            }
            return newMsgs;
          });
      },
      (err) => {
          setError(err);
          setMessages(prev => {
              const newMsgs = [...prev];
              // Remove the empty streaming message if it failed completely? Or show error.
              return newMsgs; 
          });
      }
    );

    setIsStreaming(false);

    // Mark last message as done streaming (optional cleanup)
    setMessages(prev => {
        const newMsgs = [...prev];
        const lastMsg = newMsgs[newMsgs.length - 1];
        if (lastMsg) lastMsg.isStreaming = false;
        return newMsgs;
    });

    // After streaming, generate audio
    if (fullAnswer) {
        const data = await generateTts(fullAnswer);
        if (data && data.audio_url) {
            setAudioUrl(`${API_BASE_URL}${data.audio_url}`);
            setTimeout(() => {
                playAudio();
            }, 100);
        }
    }
  };
  
  // Audio Controls
  const playAudio = () => {
    if (audioRef.current) {
        audioRef.current.play().catch(e => console.log("Audio play failed:", e));
        setIsPlaying(true);
    }
  };

  const stopAudio = () => {
    if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
        setIsPlaying(false);
    }
  };

  // Handle Transcription from Recorder
  const handleTranscription = (text) => {
      // Add user message first
      setMessages(prev => [...prev, { role: 'user', content: text }]);
      processQuestion(text);
  };

  return (
    <div className="app-container">
      <header>
        <div className="brand">
            <h1>LabBot <span className="beta-tag">AI</span></h1>
            <p>Interactive Lab Assistant</p>
        </div>
        <div className="status-indicators">
            {isStreaming && (
                <div className="status-badge processing">
                    <RefreshCw size={14} className="spin" /> Processing
                </div>
            )}
            {isPlaying && (
                <div className="status-badge playing">
                    <Volume2 size={14} /> Speaking
                </div>
            )}
        </div>
      </header>

      <main className="chat-interface">
        <div className="messages-container">
           {messages.map((msg, idx) => (
             <motion.div 
               key={idx}
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               className={`message ${msg.role}`}
             >
                <div className="avatar">
                    {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                </div>
                <div className="message-content">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  
                  {/* Show audio controls only on the LATEST assistant message if audio is available */}
                  {msg.role === 'assistant' && idx === messages.length - 1 && audioUrl && (
                    <div className="audio-controls">
                        {isPlaying ? (
                            <button className="icon-btn stop" onClick={stopAudio} title="Stop Audio">
                                <StopCircle size={16} /> Stop
                            </button>
                        ) : (
                            <button className="icon-btn play" onClick={playAudio} title="Replay">
                                <Volume2 size={16} /> Replay
                            </button>
                        )}
                    </div>
                  )}
                </div>
             </motion.div>
           ))}
           
           {error && (
               <div className="error-banner">
                   ⚠️ {error}
               </div>
           )}
           <div ref={messagesEndRef} />
        </div>

        <div className="input-wrapper">
            <form onSubmit={handleSubmit} className="input-container">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask about our lab projects..."
                disabled={isStreaming}
              />
              
              {/* Audio Recorder Button */}
              <AudioRecorder 
                onTranscription={handleTranscription} 
                onError={setError} 
              />
              
            <button type="submit" disabled={isStreaming || !question.trim()}>
                {isStreaming ? <RefreshCw size={20} className="spin" /> : <Send size={20} />}
              </button>
            </form>
        </div>
      </main>

      <audio 
        ref={audioRef} 
        src={audioUrl || ""} 
        style={{ display: 'none' }} 
        onEnded={() => setIsPlaying(false)}
        onPause={() => setIsPlaying(false)}
        onPlay={() => setIsPlaying(true)}
      />
    </div>
  );
}

export default App;
