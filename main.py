import asyncio
import os
import numpy as np
import time
import threading
from dotenv import load_dotenv
from google import genai
from google.genai.types import LiveConnectConfig, HttpOptions
import sounddevice as sd
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)

# Load environment variables
load_dotenv(override=True)

# Set the API key
gemini_api_key = os.environ.get('GEMINI_API_KEY')
if not gemini_api_key:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set")

# Audio settings
SAMPLE_RATE = 24000
BUFFER_SIZE = 3  # Number of chunks to buffer before playback

class ReliableAudioPlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.int16)
        self.buffer_lock = threading.Lock()
        self.playback_thread = None
        self.stop_event = threading.Event()
        self.new_data_event = threading.Event()
        self.initial_buffer_filled = False
        self.initial_buffer_size = 4000  # ~170ms of audio
        
    def add_chunk(self, chunk):
        """Add audio chunk to buffer"""
        with self.buffer_lock:
            self.buffer = np.append(self.buffer, chunk)
        self.new_data_event.set()
        
        # Mark when we have enough data to start
        if not self.initial_buffer_filled and len(self.buffer) >= self.initial_buffer_size:
            self.initial_buffer_filled = True
    
    def start_playback(self):
        """Start audio playback in a separate thread"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.stop_event.clear()
            self.playback_thread = threading.Thread(target=self._playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def _playback_worker(self):
        """Worker thread for audio playback"""
        # Wait until we have enough data before starting
        while not self.initial_buffer_filled and not self.stop_event.is_set():
            self.new_data_event.wait(timeout=0.1)
            self.new_data_event.clear()
            
        if self.stop_event.is_set():
            return
        
        # Start audio stream with callback
        def audio_callback(outdata, frames, time, status):
            with self.buffer_lock:
                if len(self.buffer) == 0:
                    # No data yet, output silence
                    outdata.fill(0)
                    return
                
                # Get data from buffer
                if len(self.buffer) >= frames:
                    # Convert from int16 to float32 (-1.0 to 1.0)
                    outdata[:, 0] = self.buffer[:frames].astype(np.float32) / 32768.0
                    # Remove used data from buffer
                    self.buffer = self.buffer[frames:]
                else:
                    # Not enough data, use what we have and fill rest with silence
                    outdata[:len(self.buffer), 0] = self.buffer.astype(np.float32) / 32768.0
                    outdata[len(self.buffer):, 0] = 0
                    self.buffer = np.array([], dtype=np.int16)
        
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=1024,  # Smaller block size for better responsiveness
                dtype='float32'
            ) as stream:
                # Keep stream alive until stopped
                while not self.stop_event.is_set():
                    # Check if we need more data
                    with self.buffer_lock:
                        buffer_empty = len(self.buffer) == 0
                    
                    if buffer_empty:
                        # Wait for more data or stop signal
                        self.new_data_event.wait(timeout=0.1)
                        self.new_data_event.clear()
                    else:
                        # Sleep a bit to reduce CPU usage
                        time.sleep(0.01)
                
                # Let the stream play out remaining buffer
                time.sleep(0.2)
                
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def stop(self):
        """Stop audio playback"""
        self.stop_event.set()
        self.new_data_event.set()  # Wake up thread if waiting
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

# Gemini configuration
config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(
                voice_name="Puck",
            )
        )
    ),
)

async def main():
    client = genai.Client(
        api_key=gemini_api_key, 
        http_options=HttpOptions(api_version="v1alpha")  
    )
    
    model_id = "gemini-2.0-flash-exp"
    audio_player = ReliableAudioPlayer(sample_rate=SAMPLE_RATE)

    print("Connecting to Gemini...")
    async with client.aio.live.connect(model=model_id, config=config) as session:
        
        async def send() -> bool:
            text_input = input("Input > ")
            if text_input.lower() in ("q", "quit", "exit"):
                return False
            await session.send(input=text_input, end_of_turn=True)
            return True
            
        async def receive() -> None:
            chunks_received = 0
            playback_started = False
            
            print("Receiving audio response...")
            async for message in session.receive():
                if hasattr(message, 'server_content') and message.server_content:
                    if message.server_content.model_turn:
                        for part in message.server_content.model_turn.parts:
                            if part.inline_data:
                                chunk = np.frombuffer(part.inline_data.data, dtype=np.int16)
                                chunks_received += 1
                                
                                # Add chunk to player
                                audio_player.add_chunk(chunk)
                                
                                # Start playback after buffer threshold
                                if not playback_started and chunks_received >= BUFFER_SIZE:
                                    print(f"Starting playback after {BUFFER_SIZE} chunks")
                                    audio_player.start_playback()
                                    playback_started = True
                    
                    if hasattr(message.server_content, 'turn_complete') and message.server_content.turn_complete:
                        print("Response complete")
                        
                        # Start playback if we haven't already
                        if not playback_started and chunks_received > 0:
                            audio_player.start_playback()
                            
                        # Wait a bit for audio to finish
                        await asyncio.sleep(0.5)
                        
                        # Stop playback
                        audio_player.stop()
                        break
                        
            return
            
        while True:
            if not await send():
                break
            await receive()

# Run the async function with asyncio
if __name__ == "__main__":
    asyncio.run(main())
