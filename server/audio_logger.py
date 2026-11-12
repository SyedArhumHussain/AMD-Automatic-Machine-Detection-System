#!/usr/bin/env python3

import os
import sys
import time
import threading
import queue
import wave
import traceback
import numpy as np
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
STORAGE_SERVER   = os.getenv("STORAGE_SERVER",   "u563163.your-storagebox.de")
STORAGE_PORT     = int(os.getenv("STORAGE_PORT",  "23"))
STORAGE_USER     = os.getenv("STORAGE_USER",      "u563163")
STORAGE_PASSWORD = os.getenv("STORAGE_PASSWORD",  "AMD$server!1122@ARC")

STORAGE_PATH     = os.getenv("STORAGE_PATH",      "/home/audio_storage")

AUDIO_SAVE_ENABLED = os.getenv("AUDIO_SAVE_ENABLED", "true").lower() == "true"
# MP3 bitrate (ignored if not using MP3, but we always use MP3)
BITRATE            = os.getenv("MP3_BITRATE",  "64k")
MAX_QUEUE_SIZE     = int(os.getenv("AUDIO_QUEUE_SIZE", "1000"))
SAMPLE_RATE        = 8000

MAX_RETRIES        = int(os.getenv("AUDIO_MAX_RETRIES", "3"))
RETRY_DELAY        = float(os.getenv("AUDIO_RETRY_DELAY", "2.0"))
CONN_TIMEOUT       = int(os.getenv("AUDIO_CONN_TIMEOUT", "15"))
KEEPALIVE_SEC      = int(os.getenv("AUDIO_KEEPALIVE", "30"))

AUDIO_DEBUG = os.getenv("AUDIO_DEBUG", "true").lower() == "true"


def _dbg(msg: str):
    if AUDIO_DEBUG:
        print(f"[AUDIO-DBG] {msg}", flush=True)

def _info(msg: str):
    print(f"[AUDIO] {msg}", flush=True)

def _err(msg: str):
    print(f"[AUDIO ERROR] {msg}", flush=True)


# ---------------------------------------------------------------------------
#  Lazy imports
# ---------------------------------------------------------------------------
try:
    import paramiko
    _PARAMIKO_OK = True
except ImportError:
    _PARAMIKO_OK = False
    _err("paramiko not installed! pip install paramiko")

try:
    from pydub import AudioSegment
    _PYDUB_OK = True
except ImportError:
    _PYDUB_OK = False


# ---------------------------------------------------------------------------
#  MP3 capability probe – runs at import time, MUST succeed
# ---------------------------------------------------------------------------
def _probe_mp3_support() -> bool:
    """Actually encode a tiny buffer to MP3 to verify pydub + ffmpeg work."""
    if not _PYDUB_OK:
        return False
    try:
        sample = (np.zeros(800, dtype=np.int16)).tobytes()  # 0.1s silence
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            wav_path = wf.name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mf:
            mp3_path = mf.name
        try:
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(sample)
            seg = AudioSegment.from_wav(wav_path)
            seg.export(mp3_path, format="mp3", bitrate="64k")
            return os.path.getsize(mp3_path) > 0
        finally:
            for p in (wav_path, mp3_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
    except Exception as e:
        _err(f"MP3 probe failed: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
#  Enforce MP3 format – no fallback
# ---------------------------------------------------------------------------
_MP3_OK = _probe_mp3_support()

if not _MP3_OK:
    _err("=" * 70)
    _err("FATAL: MP3 encoding is NOT available (pydub or ffmpeg missing).")
    _err("This module requires MP3 output. Please install:")
    _err("  apt-get install ffmpeg")
    _err("  pip install pydub")
    _err("Exiting now.")
    _err("=" * 70)
    sys.exit(1)

# Force MP3 format regardless of any environment variable
EFFECTIVE_AUDIO_FORMAT = "mp3"

# Make this the source of truth for BOTH audio_logger AND elastic_logger
os.environ["EFFECTIVE_AUDIO_FORMAT"] = EFFECTIVE_AUDIO_FORMAT
os.environ["AUDIO_FORMAT"] = EFFECTIVE_AUDIO_FORMAT

_info(f"EFFECTIVE_AUDIO_FORMAT = {EFFECTIVE_AUDIO_FORMAT} (MP3 forced, probe OK)")


# ---------------------------------------------------------------------------
#  Shared naming helpers – always produce .mp3
# ---------------------------------------------------------------------------
def sanitize_vid(vid: str) -> str:
    """Keep only [A-Za-z0-9_-]. Must match elastic_logger exactly."""
    return "".join(c for c in (vid or "") if c.isalnum() or c in "-_")


def build_audio_filename(vid: str, conn_id: int = 0, ts: float = None) -> str:
    """
    Single source of truth for the filename.
    Always returns "{VID}.mp3" or "conn_{conn_id}_{timestamp}.mp3" fallback.
    """
    safe = sanitize_vid(vid)
    if safe:
        return f"{safe}.mp3"
    dt = datetime.fromtimestamp(ts or time.time())
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    return f"conn_{conn_id}_{stamp}.mp3"


class AudioStorage:
    """Non-blocking audio file uploader (SFTP) driven by a background thread.
       Guarantees MP3 output only.
    """

    def __init__(self):
        self._queue   = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._running = True
        self._thread  = None
        self._ssh     = None
        self._sftp    = None
        self._lock    = threading.Lock()
        self._dir_ok  = False

        self.stats: Dict[str, Any] = {
            "stored": 0,
            "failed": 0,
            "queue_size": 0,
            "last_error": None,
            "last_success": None,
            "reconnects": 0,
        }

        if not AUDIO_SAVE_ENABLED:
            _info("Storage DISABLED")
            return
        if not _PARAMIKO_OK:
            _err("Cannot start — paramiko missing")
            return

        _info("=" * 70)
        _info(f"  Storage ENABLED (v10.1 – MP3 only, customer_name added)")
        _info(f"  Server      : {STORAGE_USER}@{STORAGE_SERVER}:{STORAGE_PORT}")
        _info(f"  Save to     : {STORAGE_PATH}  (flat, no subdirs)")
        _info(f"  Format      : MP3 (forced)")
        _info(f"  Bitrate     : {BITRATE}")
        _info(f"  Naming      : {{VID}}.mp3")
        _info(f"  Debug       : {AUDIO_DEBUG}")
        _info("=" * 70)
        self._start_worker()

    # ------------------------------------------------------------------
    def _close(self):
        for obj in (self._sftp, self._ssh):
            if obj:
                try:
                    obj.close()
                except Exception:
                    pass
        self._sftp = None
        self._ssh  = None
        self._dir_ok = False

    def _connect(self) -> bool:
        with self._lock:
            self._close()
            try:
                _dbg(f"Connecting to {STORAGE_SERVER}:{STORAGE_PORT} ...")
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(
                    hostname=STORAGE_SERVER,
                    port=STORAGE_PORT,
                    username=STORAGE_USER,
                    password=STORAGE_PASSWORD,
                    timeout=CONN_TIMEOUT,
                    banner_timeout=CONN_TIMEOUT,
                    allow_agent=False,
                    look_for_keys=False,
                )
                transport = ssh.get_transport()
                if transport:
                    transport.set_keepalive(KEEPALIVE_SEC)
                sftp = ssh.open_sftp()
                self._ssh  = ssh
                self._sftp = sftp
                self.stats["reconnects"] += 1
                _info("Connected to storage server")
                return True
            except paramiko.AuthenticationException:
                _err("Auth failed - check user/password")
                self.stats["last_error"] = "Auth failed"
                return False
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                _err(f"Connect failed: {msg}")
                self.stats["last_error"] = msg
                return False

    def _alive(self) -> bool:
        if self._sftp is None or self._ssh is None:
            return False
        try:
            t = self._ssh.get_transport()
            if t is None or not t.is_active():
                return False
            self._sftp.normalize(".")
            return True
        except Exception:
            return False

    def _ensure_connected(self) -> bool:
        if self._alive():
            return True
        _dbg("Connection dead - reconnecting")
        return self._connect()

    def _ensure_dir(self):
        if self._dir_ok:
            return
        parts = STORAGE_PATH.strip("/").split("/")
        path = ""
        for part in parts:
            path = f"{path}/{part}"
            try:
                self._sftp.stat(path)
            except FileNotFoundError:
                try:
                    self._sftp.mkdir(path)
                    _dbg(f"Created dir: {path}")
                except Exception:
                    pass
            except Exception:
                pass
        self._dir_ok = True

    # ------------------------------------------------------------------
    @staticmethod
    def _to_pcm16(audio: np.ndarray) -> bytes:
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    @staticmethod
    def _write_wav(pcm: bytes, path: str):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)

    @staticmethod
    def _wav_to_mp3(wav_path: str, mp3_path: str) -> bool:
        if not _PYDUB_OK:
            return False
        try:
            seg = AudioSegment.from_wav(wav_path)
            seg.export(mp3_path, format="mp3", bitrate=BITRATE, parameters=["-q:a", "2"])
            return True
        except Exception as e:
            _err(f"MP3 encode failed: {e}")
            return False

    # ------------------------------------------------------------------
    def _upload(self, local: str, remote: str) -> bool:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if not self._ensure_connected():
                    raise ConnectionError("No SFTP connection")
                self._ensure_dir()
                _dbg(f"Uploading: {remote} (attempt {attempt}/{MAX_RETRIES})")
                self._sftp.put(local, remote)
                try:
                    st = self._sftp.stat(remote)
                    _dbg(f"Verified: {remote} ({st.st_size} bytes)")
                except Exception:
                    pass
                return True
            except Exception as e:
                _err(f"Upload attempt {attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}")
                self._dir_ok = False
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    self._connect()
        return False

    # ------------------------------------------------------------------
    def _process(self, task: Dict[str, Any]) -> Optional[str]:
        audio: np.ndarray = task["audio_data"]
        vid       = task.get("vid", "")
        conn_id   = task.get("conn_id", 0)
        timestamp = task.get("timestamp", time.time())
        number    = task.get("customer_number", "?")
        decision  = task.get("decision", "?")
        server_ip = task.get("server_ip", "?")
        cust_name = task.get("customer_name", "")

        if audio is None or len(audio) == 0:
            _dbg("Empty audio - skipping")
            return None

        dur = len(audio) / SAMPLE_RATE

        # === Always .mp3 filename ===
        fname  = build_audio_filename(vid, conn_id, timestamp)
        remote = f"{STORAGE_PATH}/{fname}"

        _dbg(f"Processing: VID={vid!r} -> safe={sanitize_vid(vid)!r} | Num={number} | SrvIP={server_ip} | Cust={cust_name} | {decision} | {dur:.1f}s -> {fname}")

        pcm = self._to_pcm16(audio)
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(mp3_fd)
        upload_path = None

        try:
            self._write_wav(pcm, wav_path)
            _dbg(f"WAV tmp: {os.path.getsize(wav_path)} bytes")

            if not self._wav_to_mp3(wav_path, mp3_path):
                # Should never happen because startup probe passed
                _err(f"CRITICAL: MP3 encoding failed for {fname} despite startup probe passing")
                self.stats["failed"] += 1
                self.stats["last_error"] = f"MP3 encode failed mid-call: {fname}"
                return None

            _dbg(f"MP3 tmp: {os.path.getsize(mp3_path)} bytes")
            upload_path = mp3_path

            if self._upload(upload_path, remote):
                self.stats["stored"] += 1
                self.stats["last_success"] = time.time()
                _info(f"STORED: {remote} | VID={vid!r}→{sanitize_vid(vid)} | SrvIP={server_ip} | Cust={cust_name}")
                return fname
            else:
                self.stats["failed"] += 1
                self.stats["last_error"] = f"Upload failed: {fname}"
                return None

        except Exception as e:
            self.stats["failed"] += 1
            self.stats["last_error"] = str(e)
            _err(f"Process error: {traceback.format_exc()}")
            return None
        finally:
            for p in (wav_path, mp3_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    def _worker(self):
        _info("Worker thread started")
        if not self._connect():
            _err("Initial connection failed - will retry on first upload")

        while self._running:
            try:
                task = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue
            if task is None:
                continue

            fname = self._process(task)
            self.stats["queue_size"] = self._queue.qsize()

            if fname:
                _info(f"SAVED: {fname} | VID={task.get('vid','?')} | Num={task.get('customer_number','?')} | SrvIP={task.get('server_ip','?')} | Cust={task.get('customer_name','')}")
            else:
                _err(f"FAILED: VID={task.get('vid','?')} | Num={task.get('customer_number','?')} | SrvIP={task.get('server_ip','?')} | Cust={task.get('customer_name','')}")

        self._close()
        _info("Worker thread stopped")

    def _start_worker(self):
        self._thread = threading.Thread(target=self._worker, daemon=True, name="audio-sftp")
        self._thread.start()
        _info(f"Worker started (queue={MAX_QUEUE_SIZE})")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def log_audio(self, audio_data: np.ndarray, conn_id: int,
                  customer_number: str, decision: str,
                  transcript: str = "", confidence: float = 0.0,
                  vid: str = "", server_ip: str = "",
                  customer_name: str = ""):          # <-- NEW parameter
        """
        Enqueue audio for background SFTP upload. Non-blocking.
        Always saves as .mp3.
        """
        if not AUDIO_SAVE_ENABLED or not _PARAMIKO_OK:
            _dbg("log_audio: storage disabled or paramiko missing")
            return
        if audio_data is None or len(audio_data) == 0:
            _dbg("log_audio: empty audio")
            return

        dur = len(audio_data) / SAMPLE_RATE
        sanitized = sanitize_vid(vid)
        _dbg(f"Queuing: VID={vid!r}→{sanitized!r} | Num={customer_number} | SrvIP={server_ip} | Cust={customer_name} | {decision} | {dur:.1f}s")

        if not sanitized:
            _err(f"WARNING: empty/invalid VID {vid!r} — file will use conn_id fallback naming, elastic URL will NOT match!")

        task = {
            "audio_data": audio_data.copy(),
            "conn_id": conn_id,
            "customer_number": customer_number,
            "vid": vid,
            "server_ip": server_ip,
            "customer_name": customer_name,
            "decision": decision,
            "transcript": transcript,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        try:
            self._queue.put(task, timeout=1.0)
            self.stats["queue_size"] = self._queue.qsize()
            _dbg(f"Queued OK (size={self._queue.qsize()})")
        except queue.Full:
            _err(f"Queue full! Dropping: {customer_number} VID={vid} SrvIP={server_ip} Cust={customer_name}")
            self.stats["failed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "queue_size": self._queue.qsize(),
            "enabled": AUDIO_SAVE_ENABLED,
            "effective_format": EFFECTIVE_AUDIO_FORMAT,
            "mp3_available": _MP3_OK,
            "thread_alive": self._thread.is_alive() if self._thread else False,
            "connected": self._alive(),
            "storage_path": STORAGE_PATH,
        }

    def stop(self):
        _info("Stopping ...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=10.0)
        _info(f"Done. stored={self.stats['stored']} failed={self.stats['failed']}")


# Singleton
_audio_logger_instance = None

def get_audio_logger():
    global _audio_logger_instance
    if _audio_logger_instance is None:
        _audio_logger_instance = AudioStorage()
    return _audio_logger_instance
