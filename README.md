# WhisperLive Client

A lightweight client for real-time transcription and translation with the `WhisperLive` server.

## Features

- Real-time audio transcription/translation.
- Supports Whisper models and Voice Activity Detection (VAD).
- Streams audio to the server; optionally saves `.wav` output.

## Setup

1. Install Python 3.8+ and dependencies.
   Make sure `portaudio` or its development package e.g. `portaudio19-dev` is
   installed.

   - via `pip`:

   ```bash
   pip install loguru numpy pyaudio websocket-client
   ```

   - or via `conda`:

   ```bash
   conda env create -f py311-wslclient.yaml
   ```

2. Start the `WhisperLive` server.

## Usage

Run the client:

```bash
python client.py
```

Configure options (server, task, output) in the `__main__` block.
