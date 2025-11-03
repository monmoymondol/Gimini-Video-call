
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Blob as GenaiBlob } from '@google/genai';

// --- Type Definitions ---
type Transcript = {
  speaker: 'user' | 'model';
  text: string;
};

// --- Helper Functions ---
const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64data = (reader.result as string).split(',')[1];
            resolve(base64data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
};

function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): GenaiBlob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}


// --- UI Components (defined outside App to prevent re-creation) ---

// FIX: Changed JSX.Element to React.ReactElement to resolve namespace issue.
const VideoPlaceholder: React.FC<{ icon: React.ReactElement; text: string }> = ({ icon, text }) => (
    <div className="w-full h-full bg-gray-800 rounded-lg flex flex-col items-center justify-center">
        {icon}
        <p className="mt-4 text-gray-400">{text}</p>
    </div>
);

const UserVideo: React.FC<{ stream: MediaStream | null }> = ({ stream }) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    useEffect(() => {
        if (videoRef.current && stream) {
            videoRef.current.srcObject = stream;
        }
    }, [stream]);

    return (
        <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover rounded-lg transform -scale-x-100"
        />
    );
};

const TranscriptionPanel: React.FC<{ transcripts: Transcript[], currentInput: string, currentOutput: string }> = ({ transcripts, currentInput, currentOutput }) => {
    const scrollRef = useRef<HTMLDivElement>(null);
    
    useEffect(() => {
        if(scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [transcripts, currentInput, currentOutput]);
    
    return (
        <div className="w-full lg:w-96 bg-gray-800/80 backdrop-blur-sm p-4 rounded-lg flex flex-col">
            <h2 className="text-xl font-bold mb-4 border-b border-gray-600 pb-2">Transcription</h2>
            <div ref={scrollRef} className="flex-grow space-y-4 overflow-y-auto pr-2">
                {transcripts.map((t, i) => (
                    <div key={i} className={`p-3 rounded-lg ${t.speaker === 'user' ? 'bg-blue-600/50 text-right' : 'bg-gray-700/50'}`}>
                        <p className="font-bold capitalize text-sm mb-1">{t.speaker}</p>
                        <p>{t.text}</p>
                    </div>
                ))}
                 {currentInput && (
                    <div className="p-3 rounded-lg bg-blue-600/30 text-right opacity-70">
                        <p className="font-bold capitalize text-sm mb-1">User</p>
                        <p>{currentInput}</p>
                    </div>
                )}
                {currentOutput && (
                    <div className="p-3 rounded-lg bg-gray-700/30 opacity-70">
                        <p className="font-bold capitalize text-sm mb-1">Model</p>
                        <p>{currentOutput}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

// --- Main App Component ---

export default function App() {
    const [isConnecting, setIsConnecting] = useState(false);
    const [isCallActive, setIsCallActive] = useState(false);
    const [transcripts, setTranscripts] = useState<Transcript[]>([]);
    const [currentInputTranscription, setCurrentInputTranscription] = useState('');
    const [currentOutputTranscription, setCurrentOutputTranscription] = useState('');

    // Refs for streams, contexts, and session
    const localStreamRef = useRef<MediaStream | null>(null);
    const sessionPromiseRef = useRef<Promise<any> | null>(null);
    const inputAudioContextRef = useRef<AudioContext | null>(null);
    const outputAudioContextRef = useRef<AudioContext | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const frameIntervalRef = useRef<number | null>(null);
    const audioSourcesRef = useRef(new Set<AudioBufferSourceNode>());
    const nextStartTimeRef = useRef(0);
    const localVideoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));

    const cleanUp = useCallback(() => {
        console.log("Cleaning up resources...");
        
        if (sessionPromiseRef.current) {
            sessionPromiseRef.current.then(session => session.close()).catch(e => console.error("Error closing session:", e));
            sessionPromiseRef.current = null;
        }

        if (localStreamRef.current) {
            localStreamRef.current.getTracks().forEach(track => track.stop());
            localStreamRef.current = null;
        }

        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }

        if (scriptProcessorRef.current) {
            scriptProcessorRef.current.disconnect();
            scriptProcessorRef.current = null;
        }

        inputAudioContextRef.current?.close().catch(e => console.error("Error closing input audio context:", e));
        inputAudioContextRef.current = null;
        
        outputAudioContextRef.current?.close().catch(e => console.error("Error closing output audio context:", e));
        outputAudioContextRef.current = null;

        audioSourcesRef.current.forEach(source => source.stop());
        audioSourcesRef.current.clear();
        nextStartTimeRef.current = 0;

        setIsCallActive(false);
        setIsConnecting(false);
    }, []);

    const handleStartCall = async () => {
        setIsConnecting(true);
        setTranscripts([]);
        setCurrentInputTranscription('');
        setCurrentOutputTranscription('');

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
            localStreamRef.current = stream;
            if (localVideoRef.current) {
                localVideoRef.current.srcObject = stream;
            }
            
            // FIX: Cast window to `any` to access vendor-prefixed webkitAudioContext
            outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });

            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
            sessionPromiseRef.current = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => {
                        console.log('Session opened.');

                        // --- Audio Input Streaming ---
                        // FIX: Cast window to `any` to access vendor-prefixed webkitAudioContext
                        inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
                        const audioSource = inputAudioContextRef.current.createMediaStreamSource(stream);
                        scriptProcessorRef.current = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
                        
                        scriptProcessorRef.current.onaudioprocess = (audioProcessingEvent) => {
                            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                            const pcmBlob = createBlob(inputData);
                            if (sessionPromiseRef.current) {
                                sessionPromiseRef.current.then((session) => {
                                    session.sendRealtimeInput({ media: pcmBlob });
                                });
                            }
                        };
                        audioSource.connect(scriptProcessorRef.current);
                        scriptProcessorRef.current.connect(inputAudioContextRef.current.destination);

                        // --- Video Input Streaming ---
                        if (localVideoRef.current) {
                            const videoEl = localVideoRef.current;
                            const canvasEl = canvasRef.current;
                            const ctx = canvasEl.getContext('2d');
                            
                            frameIntervalRef.current = window.setInterval(() => {
                                if (ctx && videoEl.readyState >= 2) {
                                    canvasEl.width = videoEl.videoWidth;
                                    canvasEl.height = videoEl.videoHeight;
                                    ctx.drawImage(videoEl, 0, 0, videoEl.videoWidth, videoEl.videoHeight);
                                    canvasEl.toBlob(
                                        async (blob) => {
                                            if (blob && sessionPromiseRef.current) {
                                                const base64Data = await blobToBase64(blob);
                                                sessionPromiseRef.current.then((session) => {
                                                    session.sendRealtimeInput({ media: { data: base64Data, mimeType: 'image/jpeg' } });
                                                });
                                            }
                                        }, 'image/jpeg', 0.8
                                    );
                                }
                            }, 1000 / 15); // 15 FPS
                        }

                        setIsConnecting(false);
                        setIsCallActive(true);
                    },
                    onmessage: async (message: LiveServerMessage) => {
                        // Handle transcriptions
                        if (message.serverContent?.inputTranscription) {
                            setCurrentInputTranscription(prev => prev + message.serverContent.inputTranscription.text);
                        }
                        if (message.serverContent?.outputTranscription) {
                            setCurrentOutputTranscription(prev => prev + message.serverContent.outputTranscription.text);
                        }
                        if (message.serverContent?.turnComplete) {
                            const fullInput = currentInputTranscription + (message.serverContent?.inputTranscription?.text || '');
                            const fullOutput = currentOutputTranscription + (message.serverContent?.outputTranscription?.text || '');
                            
                            setTranscripts(prev => [...prev, { speaker: 'user', text: fullInput }, { speaker: 'model', text: fullOutput }]);
                            setCurrentInputTranscription('');
                            setCurrentOutputTranscription('');
                        }

                        // Handle audio output
                        const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
                        if (base64Audio && outputAudioContextRef.current) {
                            const audioCtx = outputAudioContextRef.current;
                            nextStartTimeRef.current = Math.max(nextStartTimeRef.current, audioCtx.currentTime);
                            const audioBuffer = await decodeAudioData(decode(base64Audio), audioCtx, 24000, 1);
                            const source = audioCtx.createBufferSource();
                            source.buffer = audioBuffer;
                            source.connect(audioCtx.destination);
                            source.addEventListener('ended', () => audioSourcesRef.current.delete(source));
                            source.start(nextStartTimeRef.current);
                            nextStartTimeRef.current += audioBuffer.duration;
                            audioSourcesRef.current.add(source);
                        }
                         if (message.serverContent?.interrupted) {
                            audioSourcesRef.current.forEach(source => source.stop());
                            audioSourcesRef.current.clear();
                            nextStartTimeRef.current = 0;
                        }
                    },
                    onerror: (e: ErrorEvent) => {
                        console.error('Session error:', e);
                        alert(`An error occurred: ${e.message}. Please try again.`);
                        cleanUp();
                    },
                    onclose: () => {
                        console.log('Session closed.');
                        cleanUp();
                    },
                },
                config: {
                    responseModalities: [Modality.AUDIO],
                    inputAudioTranscription: {},
                    outputAudioTranscription: {},
                    speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
                    systemInstruction: 'You are a friendly video call assistant. Keep your responses concise and conversational.'
                },
            });

        } catch (error) {
            console.error('Failed to start call:', error);
            alert('Could not access camera and microphone. Please check your permissions and try again.');
            cleanUp();
        }
    };
    
    const handleEndCall = useCallback(() => {
        cleanUp();
    }, [cleanUp]);
    
    // Cleanup on component unmount
    useEffect(() => {
        return () => {
            cleanUp();
        };
    }, [cleanUp]);

    const GeminiIcon = <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-16 h-16 text-gray-600"><path d="M12.848 9.852a.75.75 0 0 0-1.06 0L9.853 11.788a.75.75 0 1 0 1.06 1.06l1.935-1.935a.75.75 0 0 0 0-1.06Z" /><path d="m14.28 11.96 4.253 4.253a.75.75 0 0 1-1.06 1.06L13.22 13.02a.75.75 0 0 1 1.06-1.06Z" /><path fillRule="evenodd" d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM5.25 12a6.75 6.75 0 1 1 13.5 0 6.75 6.75 0 0 1-13.5 0Z" clipRule="evenodd" /></svg>;
    const StartIcon = <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 mr-2"><path d="M4.5 4.5a3 3 0 0 0-3 3v9a3 3 0 0 0 3 3h8.25a3 3 0 0 0 3-3v-9a3 3 0 0 0-3-3H4.5ZM19.5 18a1.5 1.5 0 0 0 1.5-1.5v-6a1.5 1.5 0 0 0-1.5-1.5h-1.5v9h1.5Z" /></svg>;
    const EndIcon = <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 mr-2"><path d="M3.596 2.548a.75.75 0 0 0-1.058 1.09l3.475 3.142a2.25 2.25 0 0 0-.82 1.637v9a2.25 2.25 0 0 0 2.25 2.25h8.25a2.25 2.25 0 0 0 2.25-2.25v-2.121l4.158 3.961a.75.75 0 0 0 1.082-.996L3.596 2.548ZM18 10.5l-1.79 1.707-6.04-5.712L12 4.5h3.75a2.25 2.25 0 0 1 2.25 2.25v3.75Z" /></svg>;

    return (
        <div className="min-h-screen bg-gray-900 text-white flex flex-col p-4 font-sans">
            <header className="w-full max-w-7xl mx-auto mb-4">
                <h1 className="text-3xl font-bold text-center flex items-center justify-center gap-3">
                    {GeminiIcon} Gemini Video Call
                </h1>
            </header>

            <main className="flex-grow flex flex-col lg:flex-row gap-4 w-full max-w-7xl mx-auto">
                <div className="flex-grow flex flex-col gap-4 relative">
                    <div className="flex-grow bg-black rounded-lg overflow-hidden aspect-video relative">
                        {isCallActive ? (
                            <img src="https://picsum.photos/1280/720?grayscale" alt="AI Placeholder" className="w-full h-full object-cover"/>
                        ) : (
                            <VideoPlaceholder icon={GeminiIcon} text="Call has not started" />
                        )}
                        <div className="absolute bottom-4 right-4 w-1/4 max-w-xs aspect-video rounded-lg overflow-hidden border-2 border-gray-700 shadow-lg">
                           <video ref={localVideoRef} autoPlay muted playsInline className="w-full h-full object-cover transform -scale-x-100" />
                        </div>
                    </div>
                    
                    <div className="bg-gray-800/80 backdrop-blur-sm rounded-lg p-4 flex items-center justify-center gap-4">
                         {!isCallActive ? (
                            <button
                                onClick={handleStartCall}
                                disabled={isConnecting}
                                className="bg-green-600 hover:bg-green-700 disabled:bg-gray-500 text-white font-bold py-3 px-6 rounded-lg flex items-center transition-colors duration-200"
                            >
                                {isConnecting ? 'Connecting...' : <>{StartIcon} Start Call</>}
                            </button>
                        ) : (
                            <button
                                onClick={handleEndCall}
                                className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg flex items-center transition-colors duration-200"
                            >
                               {EndIcon} End Call
                            </button>
                        )}
                    </div>
                </div>

                <div className="h-full lg:h-auto lg:max-h-[calc(100vh-120px)] w-full lg:w-96 flex-shrink-0">
                    <TranscriptionPanel transcripts={transcripts} currentInput={currentInputTranscription} currentOutput={currentOutputTranscription} />
                </div>
            </main>
        </div>
    );
}
