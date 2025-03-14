"""WhisperLive Client for real-time audio transcription and translation."""

import json
import os
import shutil
import subprocess
import threading
import typing
import uuid
import wave

import loguru
import numpy as np
import pyaudio
import websocket

DTYPE_TXT = "U256"
DTYPE_SRT = np.dtype({
    "names": ("eos", "beg", "end", "txt"),
    "formats": (np.int8, np.int32, np.int32, DTYPE_TXT),
})

logger = loguru.logger


def seg2arr(seg: dict) -> np.ndarray:
    """Converts a segment into an numpy array of SRT dtype."""
    return np.array(
        (
            int(seg.get("completed", False)),
            int(float(seg["start"]) * 1000),
            int(float(seg["end"]) * 1000),
            seg["text"],
        ),
        dtype=DTYPE_SRT,
    )


def get_intvl_union(arr: np.ndarray) -> list[tuple[int, int]]:
    """Compute the union of completed intervals.

    Args:
        arr (np.ndarray): A structured numpy array with the following fields:
            - np.int32, start timestamp of the interval.
            - np.int32, end timestamp of the interval.

    Returns:
        list[tuple[int, int]]: A list of disjoint intervals covering all
            completed records.
    """
    # Compute the union of completed intervals
    ret = []
    beg_cur, end_cur = arr[0]

    for beg_, end_ in arr[1:]:
        # Overlapping or touching
        if beg_ <= end_cur:
            end_cur = max(end_cur, end_)
        # Disjoint
        else:
            ret.append((beg_cur, end_cur))
            beg_cur, end_cur = beg_, end_
    ret.append((beg_cur, end_cur))
    return ret


def interp_minutes(arr: np.ndarray) -> np.ndarray:
    """Interpolate transcript text for incomplete intervals.

    Modify a structured numpy array of text intervals by filling gaps in the
    union of completed intervals using incomplete intervals, based on overlap
    and proportional matching.

    Args:
        arr (np.ndarray): A structured numpy array with the following fields:
            - 'eos': np.int8, indicating whether the record is completed (1) or
              incomplete (0).
            - 'beg': np.int32, start timestamp of the interval.
            - 'end': np.int32, end timestamp of the interval.
            - 'txt': str, text associated with the interval.

    Logic:
    - Extract `completed` records (where `eos` == 1) and compute the union of
      their intervals. The union is represented as a list of disjoint intervals
      covering all completed records.
    - Identify the gaps between these union intervals. A gap is defined as the
      time span between the end of one union interval and the start of next.
    - Match each gap with an appropriate `incomplete` record (where `eos` == 0)
      using the following rules:
      - If the incomplete record's interval exactly matches a gap or is a
        subset of it, use the record's full text.
      - If the incomplete record's interval partially overlaps a gap, calculate
        the overlap ratio and fill the gap with a proportional segment of the
        record's text.
      - If an incomplete record does not overlap with any gap, discard it.
    - Add trailing incomplete records that extend beyond the last completed
    - Combine all completed records with the filled gaps, ensuring the final
      output is sorted by the `beg` field.
    """
    # Return if no records
    if arr.size == 0:
        return arr

    # Extract completed and incomplete records
    arr_compl = arr[arr["eos"] == 1]
    arr_incompl = arr[arr["eos"] == 0]

    # Return all incomplete records if no completed records
    if arr_compl.size == 0:
        return arr_incompl

    # Compute the union of completed intervals
    intvl_union = get_intvl_union(arr_compl[["beg", "end"]])

    # Compute gaps from the union intervals
    gaps = []
    for i in range(len(intvl_union) - 1):
        gaps.append((intvl_union[i][1], intvl_union[i + 1][0]))

    filled_gaps = []

    # Handle trailing incomplete records beyond the last completed interval
    end_compl_last = intvl_union[-1][1]
    arr_incompl_latest = arr_incompl[arr_incompl["beg"] >= end_compl_last]

    # Match each gap with incomplete records
    for beg_gap, end_gap in gaps:
        txt_best = "..."  # default text for the gap
        r_best = 0  # best overlap ratio

        for _, beg_incompl, end_incompl, txt_incompl in arr_incompl:
            beg_overlap = max(beg_gap, beg_incompl)
            end_overlap = min(end_gap, end_incompl)

            # 1. Skip if no overlap
            if beg_overlap >= end_overlap:
                continue

            # Calculate overlap ratio
            len_overlap = end_overlap - beg_overlap
            len_incompl = end_incompl - beg_incompl
            r_overlap = len_overlap / len_incompl

            # 2. Exact match or subset
            if beg_incompl >= beg_gap and end_incompl <= end_gap:
                txt_best = txt_incompl
                break
            # 3. Partial overlap, track best fit
            if r_overlap > r_best:
                r_best = r_overlap
                # Get only the first (100 * r_best)% words, or even fewer
                n_word = txt_incompl.count(" ")
                txt_best = " ".join(txt_incompl.split()[:int(n_word * r_best)])

        # Fill the gap with the best fit text
        filled_gaps.append((0, beg_gap, end_gap, txt_best))

    # Add trailing incomplete records
    for _, beg_latest, end_latest, txt_latest in arr_incompl_latest:
        filled_gaps.append((0, beg_latest, end_latest, txt_latest))

    arr_filled = np.array(filled_gaps, dtype=DTYPE_SRT)

    arr_ = np.concatenate((arr_compl, arr_filled), axis=0)
    arr_.sort(order="beg")
    return arr_


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
    logger.info("\n".join(text))


class WsClient:
    """Handles communication with a server using WebSocket."""

    INSTANCES: typing.ClassVar = {}
    END_OF_AUDIO = "END_OF_AUDIO"
    BACKEND_VALID = "faster_whisper"

    def __init__(  # noqa: PLR0913
        self,
        host: str,
        port: int,
        *,
        lang: str | None = None,
        translate: bool = False,
        model: str = "small",
        use_vad: bool = True,
        max_clients: int = 4,
        max_connection_time: int = 600,
    ):
        """Init a WsClient for audio recording and streaming to a server.

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
            max_clients (int, optional): The maximum number of clients that can
                connect to the server. Default 4.
            max_connection_time (int, optional): The maximum connection time
                allowed. Default 600 seconds.
        """
        self.recording = False
        self.task = "transcribe"
        self.uid = str(uuid.uuid4())
        self.waiting = False
        self.language = lang
        self.model = model
        self.server_error = False
        self.use_vad = use_vad
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
            on_message=lambda _, message: self.on_message(message),
            on_error=lambda _, error: self.on_error(error),
            on_close=lambda _, close_status_code, close_msg: self.on_close(
                close_status_code, close_msg),
        )

        WsClient.INSTANCES[self.uid] = self

        # start websocket client in a thread
        self.ws_thread = threading.Thread(
            target=self.client_socket.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.minutes = {}
        self.lines = np.empty(0, dtype=DTYPE_SRT)
        loguru.logger.info("WsClient initialized")

    def update_minutes(self, segments: list[dict]) -> None:
        """Updates the minutes dictionary with the latest segments."""
        # filter out invalid segments
        # - uncompleted segments that are not the last segment
        # - invalid timestamps
        arr_seg = np.array([seg2arr(seg) for seg in segments])
        arr_seg = arr_seg[(arr_seg["eos"] == 1)
                          | ((arr_seg["eos"] == 0) &
                             (arr_seg["beg"] < arr_seg["end"]))]

        for _, seg in enumerate(arr_seg):
            self.minutes[seg["beg"]] = seg

        arr = np.array(list(self.minutes.values()))
        arr.sort(order=["beg", "eos"])
        arr_interp = interp_minutes(arr)
        self.lines = arr_interp

    def print_minutes(self) -> None:
        """Prints the transcript segments in the minutes dictionary."""
        # Log transcript; truncate to last 3 entries for brevity.
        # print_transcript()
        if self.lines.size == 0:
            return
        clear_screen()
        for _, _, _, txt in self.lines:
            loguru.logger.info(txt)

    def handle_status_msg(self, msg: dict) -> None:
        """Handles server status messages."""
        status = msg["status"]
        if status == "WAIT":
            self.waiting = True
            loguru.logger.info("Server is full. Estimated wait time "
                               f"{round(msg['message'])} minutes.")
        elif status == "ERROR":
            loguru.logger.error(f"Error from Server: {msg['message']}")
            self.server_error = True
        elif status == "WARNING":
            loguru.logger.warning(f"Warning from Server: {msg['message']}")

    def on_message(self, msg: str) -> None:
        """Callback function called when a message is received from the server.

        It updates various attributes of the client based on the received
        message, including recording status, language detection, and server
        messages. If a disconnect message is received, it sets the recording
        status to False.

        Args:
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
            self.recording = True
            self.server_backend = dict_msg["backend"]
            if self.server_backend != self.BACKEND_VALID:
                loguru.logger.error(
                    f"Server backend {self.server_backend} not supported.")
                raise ValueError()
            loguru.logger.info(f"Server Running backend {self.server_backend}")
            return

        if "language" in dict_msg:
            self.language = dict_msg.get("language")
            lang_prob = dict_msg.get("language_prob")
            loguru.logger.info("Server detected language "
                               f"{self.language} with probability {lang_prob}")
            return

        if "segments" in dict_msg:
            self.update_minutes(dict_msg["segments"])
            self.print_minutes()

    def on_error(self, error: Exception) -> None:
        """Callback when an error occurs in the WebSocket."""
        loguru.logger.error(f"WS Error: {error}")
        self.server_error = True
        self.error_message = error

    def on_close(
        self,
        close_status_code: int | None,
        close_msg: str | None,
    ) -> None:
        """Callback when the WebSocket connection is closed."""
        loguru.logger.info(f"WS closed: {close_status_code}: {close_msg}")
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

    def send_server_packet(self, data: bytes) -> None:
        """Send an audio packet to the server using WebSocket.

        Args:
            data (bytes): Audio data packet in bytes to be sent to the server.
        """
        try:
            self.client_socket.send(data, websocket.ABNF.OPCODE_BINARY)
        except Exception:
            logger.exception("Error sending packet to server")

    def close_websocket(self) -> None:
        """Close the WebSocket connection and join the WebSocket thread.

        First attempts to close the WebSocket connection using
        `self.client_socket.close()`. After closing the connection, it joins
        the WebSocket thread to ensure proper termination.
        """
        try:
            self.client_socket.close()
        except Exception:
            logger.exception("Error closing WS")

        try:
            self.ws_thread.join()
        except Exception:
            logger.exception("Error joining WS thread")


class AudioClient:
    """Client for handling audio recording, streaming, and transcription tasks.

    It is done via one or more WebSocket connections. Acts as a high-level
    client for audio transcription tasks using a WebSocket connection. It can
    be used to send audio data for transcription to one or more servers, and
    receive transcribed text segments.

    Args:
        clients (list): one or more previously initialized WsClient instances

    Attributes:
        clients (list): the underlying WsClient instances responsible for
            handling WebSocket connections.
    """

    # directory to store audio chunks
    DIR_WAV = "tmp_audios"
    # seconds to record audio for each chunk
    SEC_CHUNK = 60

    def __init__(
        self,
        clients: list[WsClient],
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

    @classmethod
    def audio_chunk_path(cls, n: int) -> str:
        """Get the path for the audio chunk file.

        Args:
            n (int): The index of the audio chunk file.

        Returns:
            str: The path to the audio chunk file.
        """
        return os.path.join(cls.DIR_WAV, f"{n}.wav")

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
                    self.close_clients()
                    return

        loguru.logger.info("Server Ready!")
        self.record()

    def close_clients(self) -> None:
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
                client.send_server_packet(packet)

    def save_chunk_thread(self, chunk_idx: int) -> None:
        """Saves the current audio frames to a WAV file in a separate thread.

        Args:
            chunk_idx (int): The index of the audio file which determines the
            filename. This helps in maintaining the order and uniqueness of
            each chunk.
        """
        t = threading.Thread(
            target=self.save_chunk,
            args=(self.frames[:], self.audio_chunk_path(chunk_idx)),
        )
        t.start()

    def post_process(self, chunk_idx: int) -> None:
        """Finalizes the recording process.

        It finalizes the recording process by saving any remaining audio
        frames, closing the audio stream, and terminating the process.

        Args:
            chunk_idx (int): The file index to be used if there are remaining
            audio frames to be saved. This index is incremented before use if
            the last chunk is saved.
        """
        if self.save_recording and len(self.frames):
            self.save_chunk(
                self.frames[:],
                self.audio_chunk_path(chunk_idx),
            )
            chunk_idx += 1
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.close_clients()
        if self.save_recording:
            self.concat_chunks(chunk_idx)

    def record(self) -> None:
        """Record audio data from the input stream and save it to a WAV file.

        Continuously records audio data from the input stream, sends it to the
        server via a WebSocket connection, and simultaneously saves it to
        multiple WAV files in chunks. It stops recording when the
        `RECORD_SECONDS` duration is reached or when the `RECORDING` flag is
        set to `False`.

        Audio data is saved in chunks to the `DIR_WAV` directory. Each chunk is
        saved as a separate WAV file. The recording will continue until the
        specified duration is reached or until the `RECORDING` flag is set to
        `False`. The recording process can be interrupted by sending a
        KeyboardInterrupt (e.g., pressing Ctrl+C). After recording, the method
        combines all the saved audio chunks into the specified `out_file`.
        """
        chunk_idx_ = 0
        if self.save_recording:
            if os.path.exists(self.DIR_WAV):
                shutil.rmtree(self.DIR_WAV)
            os.makedirs(self.DIR_WAV)
        try:
            for _ in range(int(self.rate / self.chunk * self.record_seconds)):
                if not any(client.recording for client in self.clients):
                    break
                dat = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames += dat
                audio_array = self.bytes2arr(dat)
                self.multicast_packet(audio_array.tobytes())

                # save frames if more than a minute
                if len(self.frames) > self.rate * self.SEC_CHUNK:
                    if self.save_recording:
                        self.save_chunk_thread(chunk_idx_)
                        chunk_idx_ += 1
                    self.frames = b""

        except KeyboardInterrupt:
            self.post_process(chunk_idx_)

    def save_chunk(self, frames: bytes, filename: str) -> None:
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

    def concat_chunks(self, n_audio_file: int) -> None:
        """Combine and save recorded audio chunks into a single WAV file.

        The individual audio chunk files are expected to be located in the
        `DIR_WAV` directory. Reads each chunk file, appends its audio data to
        the final recording, and then deletes the chunk file. After combining
        and saving, the final recording is stored in the specified `out_file`.

        Args:
            n_audio_file (int): The number of audio chunk files to combine.
            out_file (str): The name of the output WAV file to save the final
            recording.
        """
        src_chunks = [
            self.audio_chunk_path(i) for i in range(n_audio_file)
            if os.path.exists(self.audio_chunk_path(i))
        ]
        with wave.open(self.out_recording, "wb") as wavfile:
            wavfile: wave.Wave_write
            wavfile.setnchannels(self.channels)
            wavfile.setsampwidth(2)
            wavfile.setframerate(self.rate)
            for chunk in src_chunks:
                with wave.open(chunk, "rb") as wav_in:
                    while True:
                        data = wav_in.readframes(self.chunk)
                        if data == b"":
                            break
                        wavfile.writeframes(data)
                # remove this file
                os.remove(chunk)
        wavfile.close()
        # clean up temporary directory to store audio chunks
        if os.path.exists(self.DIR_WAV):
            shutil.rmtree(self.DIR_WAV)

    @staticmethod
    def bytes2arr(audio_bytes: bytes) -> np.ndarray:
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


if __name__ == "__main__":
    model = "small.en"
    ws_client = WsClient(
        "localhost",
        9090,
        lang="en",
        translate=False,
        model=model,
        use_vad=True,
        max_clients=4,
        max_connection_time=600,
    )
    audio_client = AudioClient(
        [ws_client],
        save_recording=False,
        out_recording="./output_recording.wav",
    )
    audio_client()
