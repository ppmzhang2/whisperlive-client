"""WhisperLive Client for real-time audio transcription and translation."""

import json
import os
import shutil
import subprocess
import threading
import time
import typing
import uuid
import wave

import loguru
import numpy as np
import pyaudio
import websocket


def clear_screen() -> None:
    """Clears the console screen."""
    # Define the commands based on OS
    cmd_txt = "cls" if os.name == "nt" else "clear"

    # Validate if the command exists
    if shutil.which(cmd_txt) is None:
        loguru.logger.error(f"Command '{cmd_txt}' not found.")
        raise FileNotFoundError()

    try:
        subprocess.run([cmd_txt], check=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        loguru.logger.error(f"Failed to execute '{cmd_txt}': {e}")
        raise


def print_transcript(text: list[str]) -> None:
    """Prints formatted transcript text."""
    print("\n".join(text))


class Client:
    """Handles communication with a server using WebSocket."""

    INSTANCES: typing.ClassVar = {}
    END_OF_AUDIO = "END_OF_AUDIO"

    def __init__(  # noqa: PLR0913
        self,
        host: str,
        port: int,
        *,
        lang: str | None = None,
        translate: bool = False,
        model: str = "small",
        use_vad: bool = True,
        log_transcription: bool = True,
        max_clients: int = 4,
        max_connection_time: int = 600,
        out_srt: str = "output.srt",
    ):
        """Init a client for audio recording and streaming to a server.

        If host and port are not provided, the WebSocket connection will not be
        established. When translate is True, the task will be set to
        "translate" instead of "transcribe". The audio recording starts
        immediately upon initialization.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number for the WebSocket server.
            lang (str, optional): The selected language for transcription.
                Default is None.
            translate (bool, optional): Specifies if the task is translation.
                Default is False.
            model (str, optional): The model to use for transcription. Default
                "small".
            use_vad (bool, optional): Specifies if VAD should be used. Default
                True.
            log_transcription (bool, optional): Specifies if the transcription
                should be logged. Default True.
            max_clients (int, optional): The maximum number of clients that can
                connect to the server. Default 4.
            max_connection_time (int, optional): The maximum connection time
                allowed. Default 600 seconds.
            out_srt (str, optional): The output file path for the
                transcription, Default "output.srt".
        """
        self.recording = False
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.last_response_received = None
        self.disconnect_if_no_response_for = 15
        self.language = lang
        self.model = model
        self.server_error = False
        self.srt_file_path = out_srt
        self.use_vad = use_vad
        self.last_segment = None
        self.last_received_segment = None
        self.log_transcription = log_transcription
        self.max_clients = max_clients
        self.max_connection_time = max_connection_time

        if translate:
            self.task = "translate"

        self.audio_bytes = None

        # ensure host and port are provided
        socket_url = f"ws://{host}:{port}"
        self.client_socket = websocket.WebSocketApp(
            socket_url,
            on_open=lambda ws: self.on_open(ws),
            on_message=lambda ws, message: self.on_message(ws, message),
            on_error=lambda ws, error: self.on_error(ws, error),
            on_close=lambda ws, close_status_code, close_msg: self.on_close(
                ws, close_status_code, close_msg),
        )

        Client.INSTANCES[self.uid] = self

        # start websocket client in a thread
        self.ws_thread = threading.Thread(
            target=self.client_socket.run_forever)
        # self.ws_thread.setDaemon(True)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.transcript = []
        loguru.logger.info("[INFO]: Client initialized")

    def handle_status_msg(self, msg: dict) -> None:
        """Handles server status messages."""
        status = msg["status"]
        if status == "WAIT":
            self.waiting = True
            loguru.logger.info("[INFO]: Server is full. Estimated wait time "
                               f"{round(msg['message'])} minutes.")
        elif status == "ERROR":
            loguru.logger.error(f"Error from Server: {msg['message']}")
            self.server_error = True
        elif status == "WARNING":
            loguru.logger.warning(f"Warning from Server: {msg['message']}")

    def process_segments(self, segments: list[dict]) -> None:
        """Processes transcript segments."""
        text = []
        for i, seg in enumerate(segments):
            if not text or text[-1] != seg["text"]:
                text.append(seg["text"])
                if i == len(segments) - 1 and not seg.get("completed", False):
                    self.last_segment = seg
                elif (self.server_backend == "faster_whisper"
                      and seg.get("completed", False)
                      and (not self.transcript or float(seg["start"]) >= float(
                          self.transcript[-1]["end"]))):
                    self.transcript.append(seg)
        # update last received segment and last valid response time
        if (self.last_received_segment is None
                or self.last_received_segment != segments[-1]["text"]):
            self.last_response_received = time.time()
            self.last_received_segment = segments[-1]["text"]

        if self.log_transcription:
            # Truncate to last 3 entries for brevity.
            text = text[-3:]
            clear_screen()
            print_transcript(text)

    def on_message(self, ws: websocket.WebSocket, msg: str) -> None:
        """Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received
        message, including recording status, language detection, and server
        messages. If a disconnect message is received, it sets the recording
        status to False.

        Args:
            ws (websocket.WebSocket): The WebSocket client instance.
            msg (str): The received message from the server.
        """
        dict_msg = json.loads(msg)

        if self.uid != dict_msg.get("uid"):
            loguru.logger.error("invalid client uid")
            return

        if "status" in dict_msg:
            self.handle_status_msg(dict_msg)
            return

        if "message" in dict_msg and dict_msg["message"] == "DISCONNECT":
            loguru.logger.info("Server disconnected due to overtime.")
            self.recording = False

        if "message" in dict_msg and dict_msg["message"] == "SERVER_READY":
            self.last_response_received = time.time()
            self.recording = True
            self.server_backend = dict_msg["backend"]
            loguru.logger.info(f"Server Running backend {self.server_backend}")
            return

        if "language" in dict_msg:
            self.language = dict_msg.get("language")
            lang_prob = dict_msg.get("language_prob")
            loguru.logger.info("Server detected language "
                               f"{self.language} with probability {lang_prob}")
            return

        if "segments" in dict_msg:
            self.process_segments(dict_msg["segments"])

    def on_error(self, ws: websocket.WebSocket, error: Exception) -> None:
        """Callback when an error occurs in the WebSocket."""
        loguru.logger.error(f"WS Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(self, ws: websocket.WebSocket, close_status_code: int,
                 close_msg: str) -> None:
        """Callback when the WebSocket connection is closed."""
        loguru.logger.info("WS connection closed: "
                           f"{close_status_code}: {close_msg}")
        self.recording = False
        self.waiting = False

    def on_open(self, ws: websocket.WebSocket) -> None:
        """Callback when the WebSocket connection is successfully opened.

        Sends an initial configuration message to the server, including client
        UID, language selection, and task type.

        Args:
            ws (websocket.WebSocketApp): The WebSocket client instance.
        """
        loguru.logger.info("Opened connection")
        ws.send(
            json.dumps({
                "uid": self.uid,
                "language": self.language,
                "task": self.task,
                "model": self.model,
                "use_vad": self.use_vad,
                "max_clients": self.max_clients,
                "max_connection_time": self.max_connection_time,
            }))

    def send_packet_to_server(self, data: bytes) -> None:
        """Send an audio packet to the server using WebSocket.

        Args:
            data (bytes): Audio data packet in bytes to be sent to the server.
        """
        try:
            self.client_socket.send(data, websocket.ABNF.OPCODE_BINARY)
        except Exception as e:
            loguru.logger.error(f"Error sending packet to server: {e}")

    def close_websocket(self) -> None:
        """Close the WebSocket connection and join the WebSocket thread.

        First attempts to close the WebSocket connection using
        `self.client_socket.close()`. After closing the connection, it joins
        the WebSocket thread to ensure proper termination.
        """
        try:
            self.client_socket.close()
        except Exception as e:
            loguru.logger.error("Error closing WS:", e)

        try:
            self.ws_thread.join()
        except Exception as e:
            loguru.logger.error("Error joining WS thread:", e)

    def get_client_socket(self) -> websocket.WebSocketApp:
        """Get the WebSocket client socket instance.

        Returns:
            WebSocketApp: The WebSocket client socket instance currently in use
            by the client.
        """
        return self.client_socket

    def wait_before_disconnect(self) -> None:
        """Wait for a bit before disconnecting.

        In order to process pending responses.
        """
        assert self.last_response_received
        while (time.time() - self.last_response_received
               < self.disconnect_if_no_response_for):
            continue


class TranscriptionTeeClient:
    """Client for handling audio recording, streaming, and transcription tasks.

    It is done via one or more WebSocket connections. Acts as a high-level
    client for audio transcription tasks using a WebSocket connection. It can
    be used to send audio data for transcription to one or more servers, and
    receive transcribed text segments.

    Args:
        clients (list): one or more previously initialized Client instances

    Attributes:
        clients (list): the underlying Client instances responsible for
            handling WebSocket connections.
    """

    def __init__(
        self,
        clients: list[Client],
        *,
        save_recording: bool = False,
        out_recording: str = "./output_recording.wav",
    ):
        self.clients = clients
        if not self.clients:
            loguru.logger.error("At least one client is required.")
            raise ValueError()
        self.chunk = 4096
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 60000
        self.save_recording = save_recording
        self.out_recording = out_recording
        self.frames = b""
        self.p = pyaudio.PyAudio()
        # no try
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def __call__(self):
        """Start the transcription process.

        Initiates the transcription process by connecting to the server via a
        WebSocket. It waits for the server to be ready to receive audio data
        and then sends audio for transcription. If an audio file is provided,
        it will be played and streamed to the server; otherwise, it will
        perform live recording.

        Args:
            audio (str, optional): Path to an audio file for transcription.
            Default is None, which triggers live recording.
        """
        loguru.logger.info("Waiting for server ready ...")
        for client in self.clients:
            while not client.recording:
                if client.waiting or client.server_error:
                    self.close_all_clients()
                    return

        loguru.logger.info("Server Ready!")
        self.record()

    def close_all_clients(self) -> None:
        """Closes all client websockets."""
        for client in self.clients:
            client.close_websocket()

    def multicast_packet(
        self,
        packet: bytes,
        *,
        unconditional: bool = False,
    ) -> None:
        """Sends an identical packet via all clients.

        Args:
            packet (bytes): The audio data packet in bytes to be sent.
            unconditional (bool, optional): If true, send regardless of whether
                clients are recording.  Default is False.
        """
        for client in self.clients:
            if unconditional or client.recording:
                client.send_packet_to_server(packet)

    def save_chunk(self, n_audio_file: int) -> None:
        """Saves the current audio frames to a WAV file in a separate thread.

        Args:
            n_audio_file (int): The index of the audio file which determines
            the filename. This helps in maintaining the order and uniqueness of
            each chunk.
        """
        t = threading.Thread(
            target=self.write_audio_frames_to_file,
            args=(
                self.frames[:],
                f"chunks/{n_audio_file}.wav",
            ),
        )
        t.start()

    def finalize_recording(self, n_audio_file: int) -> None:
        """Finalizes the recording process.

        It finalizes the recording process by saving any remaining audio
        frames, closing the audio stream, and terminating the process.

        Args:
            n_audio_file (int): The file index to be used if there are
            remaining audio frames to be saved. This index is incremented
            before use if the last chunk is saved.
        """
        if self.save_recording and len(self.frames):
            self.write_audio_frames_to_file(self.frames[:],
                                            f"chunks/{n_audio_file}.wav")
            n_audio_file += 1
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.close_all_clients()
        if self.save_recording:
            self.write_output_recording(n_audio_file)

    def record(self) -> None:
        """Record audio data from the input stream and save it to a WAV file.

        Continuously records audio data from the input stream, sends it to the
        server via a WebSocket connection, and simultaneously saves it to
        multiple WAV files in chunks. It stops recording when the
        `RECORD_SECONDS` duration is reached or when the `RECORDING` flag is
        set to `False`.

        Audio data is saved in chunks to the "chunks" directory. Each chunk is
        saved as a separate WAV file. The recording will continue until the
        specified duration is reached or until the `RECORDING` flag is set to
        `False`. The recording process can be interrupted by sending a
        KeyboardInterrupt (e.g., pressing Ctrl+C). After recording, the method
        combines all the saved audio chunks into the specified `out_file`.
        """
        n_audio_file = 0
        if self.save_recording:
            if os.path.exists("chunks"):
                shutil.rmtree("chunks")
            os.makedirs("chunks")
        try:
            for _ in range(int(self.rate / self.chunk * self.record_seconds)):
                if not any(client.recording for client in self.clients):
                    break
                data = self.stream.read(self.chunk,
                                        exception_on_overflow=False)
                self.frames += data

                audio_array = self.bytes_to_float_array(data)

                self.multicast_packet(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > 60 * self.rate:
                    if self.save_recording:
                        self.save_chunk(n_audio_file)
                        n_audio_file += 1
                    self.frames = b""

        except KeyboardInterrupt:
            self.finalize_recording(n_audio_file)

    def write_audio_frames_to_file(self, frames: bytes, filename: str) -> None:
        """Write audio frames to a WAV file.

        The WAV file is created or overwritten with the specified name. The
        audio frames should be in the correct format and match the specified
        channel, sample width, and sample rate.

        Args:
            frames (bytes): The audio frames to be written to the file.
            filename (str): The name of the WAV file to which the frames will
                be written.
        """
        with wave.open(filename, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            wavfile.writeframes(frames)

    def write_output_recording(self, n_audio_file: int) -> None:
        """Combine and save recorded audio chunks into a single WAV file.

        The individual audio chunk files are expected to be located in the
        "chunks" directory. Reads each chunk file, appends its audio data to
        the final recording, and then deletes the chunk file. After combining
        and saving, the final recording is stored in the specified `out_file`.

        Args:
            n_audio_file (int): The number of audio chunk files to combine.
            out_file (str): The name of the output WAV file to save the final
            recording.
        """
        input_files = [
            f"chunks/{i}.wav" for i in range(n_audio_file)
            if os.path.exists(f"chunks/{i}.wav")
        ]
        with wave.open(self.out_recording, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for in_file in input_files:
                with wave.open(in_file, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                # remove this file
                os.remove(in_file)
        wavfile.close()
        # clean up temporary directory to store chunks
        if os.path.exists("chunks"):
            shutil.rmtree("chunks")

    @staticmethod
    def bytes_to_float_array(audio_bytes: bytes) -> np.ndarray:
        """Convert audio data from bytes to a NumPy float array.

        It assumes that the audio data is in 16-bit PCM format. The audio data
        is normalized to have values between -1 and 1.

        Args:
            audio_bytes (bytes): Audio data in bytes.

        Returns:
            np.ndarray: A NumPy array containing the audio data as float values
            normalized between -1 and 1.
        """
        raw_data = np.frombuffer(buffer=audio_bytes, dtype=np.int16)
        return raw_data.astype(np.float32) / 32768.0


class TranscriptionClient(TranscriptionTeeClient):
    """Client Handling audio transcription tasks via a single ws conn.

    Acts as a high-level client for audio transcription tasks using a WebSocket
    connection. It can be used to send audio data for transcription to a server
    and receive transcribed text segments.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number to connect to on the server.
        lang (str, optional): The primary language for transcription. Default
            is None, which defaults to English ('en').
        translate (bool, optional): Indicates whether translation tasks are
            required (default is False).
        save_recording (bool, optional): Indicates whether to save
            recording from microphone.
        out_recording (str, optional): File to save the output
            recording.
        output_transcription_path (str, optional): File to save the output
            transcription.

    Attributes:
        client (Client): An instance of the underlying Client class responsible
            for handling the WebSocket connection.

    Example:
        To create a TranscriptionClient and start transcription on microphone
        audio:

        ```python
        transcription_client = TranscriptionClient(host="localhost", port=9090)
        transcription_client()
        ```
    """

    def __init__(  # noqa: PLR0913
        self,
        host: str,
        port: int,
        *,
        lang: str | None = None,
        translate: bool = False,
        model: str = "small",
        use_vad: bool = True,
        save_recording: bool = False,
        out_recording: str = "./output_recording.wav",
        output_transcription_path: str = "./output.srt",
        log_transcription: bool = True,
        max_clients: int = 4,
        max_connection_time: int = 600,
    ):
        self.client = Client(
            host,
            port,
            lang=lang,
            translate=translate,
            model=model,
            out_srt=output_transcription_path,
            use_vad=use_vad,
            log_transcription=log_transcription,
            max_clients=max_clients,
            max_connection_time=max_connection_time,
        )

        if save_recording and not out_recording.endswith(".wav"):
            loguru.logger.error("Please provide a valid `out_recording`: "
                                f"{out_recording}")
            raise ValueError()
        if not output_transcription_path.endswith(".srt"):
            loguru.logger.error(
                "Please provide a valid `output_transcription_path`: "
                f"{output_transcription_path}. The file extension should be "
                "`.srt`.")
            raise ValueError()
        TranscriptionTeeClient.__init__(
            self,
            [self.client],
            save_recording=save_recording,
            out_recording=out_recording,
        )


if __name__ == "__main__":
    client = TranscriptionClient(
        "localhost",
        9090,
        lang="en",
        translate=False,
        model="large-v3",
        use_vad=False,
        save_recording=True,
        max_clients=4,
        max_connection_time=600,
    )
    client()
