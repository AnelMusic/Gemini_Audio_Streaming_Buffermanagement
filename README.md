# Low-Latency Audio Streaming from Gemini API

This repository demonstrates how to implement low-latency audio streaming when working with the Gemini API's speech synthesis capabilities. The implementation addresses the common issue of audio playback delay by using a buffer-and-stream approach instead of waiting for the entire audio response before playback.

## The Problem

When working with voice-enabled AI models like Gemini, audio responses are typically delivered in chunks. A naive implementation might look like this:

```python
audio_data = []
async for message in session.receive():
    # ... process message ...
    if part.inline_data:  # Audio chunk received
        chunk = np.frombuffer(part.inline_data.data, dtype=np.int16)
        audio_data.append(chunk)
        
# Play after receiving all chunks
if audio_data:
    combined_audio = np.concatenate(audio_data)
    sd.play(combined_audio, sample_rate)
    sd.wait()
```
```
Input > Hi tell me a joke
Receiving audio response...
Received audio chunk: 4800 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 960 samples
Received audio chunk: 4800 samples
Received audio chunk: 4800 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 4800 samples
Received audio chunk: 4800 samples
Received audio chunk: 4800 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 5760 samples
Received audio chunk: 960 samples
Received audio chunk: 4800 samples
Response complete, playing audio...
```

This approach has a significant drawback: **the user must wait for the entire response to be received before playback begins**, creating an unpleasant delay and reducing the conversational feel of the interaction.

## The Solution: Buffer and Stream

Instead of waiting for all audio chunks to be received, our solution uses a buffer-and-stream approach that:

1. Collects a small number of initial chunks to create a buffer
2. Begins playback as soon as the buffer reaches a minimum threshold
3. Continues receiving and processing chunks in the background
4. Streams the audio continuously without interruption

This approach dramatically reduces perceived latency while maintaining smooth playback.

## Implementation Details

### The Audio Player Class

The core of the solution is the streamlined `ReliableAudioPlayer` class, which manages the audio buffer and playback:

```python
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
```

Key components:
- A single numpy array buffer to store audio data
- Thread synchronization mechanisms (lock and events)
- Buffer size tracking to determine when to start playback

### Thread-Safe Buffer Management

Audio chunks are added to the buffer in a thread-safe manner:

```python
def add_chunk(self, chunk):
    """Add audio chunk to buffer"""
    with self.buffer_lock:
        self.buffer = np.append(self.buffer, chunk)
    self.new_data_event.set()
    
    # Mark when we have enough data to start
    if not self.initial_buffer_filled and len(self.buffer) >= self.initial_buffer_size:
        self.initial_buffer_filled = True
```

This ensures that chunks can be added while playback is occurring without data corruption or race conditions.

### Audio Streaming with Callbacks

Instead of using sounddevice's `play()` function (which blocks until completion), we use the streaming API with callbacks:

```python
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
```

This callback:
1. Locks the buffer to prevent concurrent modification
2. Checks if there's enough data to fill the requested frame size
3. Extracts the needed samples and updates the buffer
4. Handles the case where there's not enough data by outputting silence

### Smooth Playback Timing

To ensure smooth playback start, we wait for a minimum buffer size before beginning:

```python
# Wait until we have enough data before starting
while not self.initial_buffer_filled and not self.stop_event.is_set():
    self.new_data_event.wait(timeout=0.1)
    self.new_data_event.clear()
```

This ensures we have enough audio data (~170ms worth) before starting playback, which helps prevent stuttering.

### Stream Configuration for Low Latency

We configure the audio stream with parameters optimized for low-latency streaming:

```python
with sd.OutputStream(
    samplerate=self.sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=1024,  # Smaller block size for better responsiveness
    dtype='float32'
) as stream:
    # ... stream management ...
```

The smaller `blocksize` (1024 samples) provides better responsiveness while maintaining smooth playback.

### Managing Playback Start

In the main receive loop, we monitor chunks and start playback after a configurable threshold:

```python
# Start playback after buffer threshold
if not playback_started and chunks_received >= BUFFER_SIZE:
    print(f"Starting playback after {BUFFER_SIZE} chunks")
    audio_player.start_playback()
    playback_started = True
```

## Why This Approach Works

This implementation successfully reduces audio playback delay while preventing stuttering by:

1. **Balancing Buffer Size**: Using a buffer that's large enough to handle network jitter but small enough to minimize perceived delay

2. **Continuous Streaming**: Using the streaming API instead of discrete playback to provide seamless audio

3. **Thread Safety**: Properly synchronizing access to shared resources to prevent data corruption

4. **Adaptive Buffer Management**: Handling varying chunk sizes and network conditions gracefully

5. **Responsive Audio Blocks**: Using appropriately sized audio blocks (1024 samples) for real-time processing

6. **Efficient Resource Usage**: Minimizing state tracking to only what's necessary (e.g., checking if buffer is empty rather than tracking exact size)

## Tuning Parameters

The implementation includes several parameters that can be tuned to optimize performance:

- `BUFFER_SIZE` (default: 3): Number of chunks to buffer before starting playback
- `initial_buffer_size` (default: 4000): Minimum samples (~170ms) required in buffer before playback starts
- `blocksize` (default: 1024): Audio processing block size - smaller values reduce latency but may increase CPU usage

## Usage

To use this code:

1. Set up your environment variables (GEMINI_API_KEY) - .env
2. Run the script to start an interactive session with Gemini
3. Type your query and hear the response with minimal delay

## Conclusion

This implementation strikes a balance between low latency and smooth playback when working with streaming audio from the Gemini API. By using a buffer-and-stream approach, we create a more natural conversational experience while avoiding audio glitches and stuttering. The code is kept minimal and focused by removing unnecessary variables and simplifying state tracking.
