#!/usr/bin/env python3
import os
import sys
import json
import time
import base64
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from urllib.parse import quote

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
ELASTICSEARCH_URL      = os.getenv("ELASTICSEARCH_URL",      "")
ELASTIC_API_KEY        = os.getenv("ELASTIC_API_KEY",        "")
ELASTIC_INDEX_PREFIX   = os.getenv("ELASTIC_INDEX_PREFIX",   "")
ELASTIC_BULK_SIZE      = int(os.getenv("ELASTIC_BULK_SIZE",  "100"))    # ↑ from 50
ELASTIC_FLUSH_INTERVAL = float(os.getenv("ELASTIC_FLUSH_INTERVAL", "2.0"))  # ↓ from 5.0
ELASTIC_MAX_QUEUE      = int(os.getenv("ELASTIC_MAX_QUEUE",  "50000"))  # ↑ from 10000
ELASTIC_NUM_WORKERS    = int(os.getenv("ELASTIC_NUM_WORKERS", "3"))     # NEW: parallel workers
ELASTIC_BULK_TIMEOUT   = int(os.getenv("ELASTIC_BULK_TIMEOUT", "30"))   # ↓ from 60
ELASTIC_MAX_RETRIES    = int(os.getenv("ELASTIC_MAX_RETRIES", "5"))     # NEW: per-batch retries
ELASTIC_DEAD_LETTER    = os.getenv("ELASTIC_DEAD_LETTER", "/var/log/amd_elastic_deadletter.jsonl")
ELASTIC_QUEUE_WARN_PCT = float(os.getenv("ELASTIC_QUEUE_WARN_PCT", "0.7"))  # warn at 70% full

# Audio URL configuration (matches audio_logger.py storage box)
AUDIO_URL_USER     = os.getenv("AUDIO_URL_USER",     "")
AUDIO_URL_PASSWORD = os.getenv("AUDIO_URL_PASSWORD",  "")
AUDIO_URL_HOST     = os.getenv("AUDIO_URL_HOST",     "")
AUDIO_URL_BASE_DIR = os.getenv("AUDIO_URL_BASE_DIR", "")

_ENCODED_PASSWORD = quote(AUDIO_URL_PASSWORD, safe="")
_AUDIO_URL_PREFIX = f"https://{AUDIO_URL_USER}:{_ENCODED_PASSWORD}@{AUDIO_URL_HOST}/{AUDIO_URL_BASE_DIR}"

ELASTIC_DEBUG = os.getenv("ELASTIC_DEBUG", "false").lower() == "true"


def _dbg(msg: str):
    if ELASTIC_DEBUG:
        print(f"[ELASTIC-DBG] {msg}", flush=True)


def _info(msg: str):
    print(f"[ELASTIC] {msg}", flush=True)


def _err(msg: str):
    print(f"[ELASTIC ERROR] {msg}", flush=True)


# ---------------------------------------------------------------------------
#  SHARED NAMING — import from audio_logger if possible, else enforce MP3
# ---------------------------------------------------------------------------
try:
    from audio_logger import build_audio_filename, sanitize_vid, EFFECTIVE_AUDIO_FORMAT
    _info(f"Shared naming imported from audio_logger (format={EFFECTIVE_AUDIO_FORMAT})")
except Exception as e:
    _err(f"Could not import naming from audio_logger ({e}); using local MP3-only fallback")
    EFFECTIVE_AUDIO_FORMAT = "mp3"
    os.environ["EFFECTIVE_AUDIO_FORMAT"] = "mp3"
    os.environ["AUDIO_FORMAT"] = "mp3"

    def sanitize_vid(vid: str) -> str:
        return "".join(c for c in (vid or "") if c.isalnum() or c in "-_")

    def build_audio_filename(vid: str, conn_id: int = 0, ts: float = None) -> str:
        safe = sanitize_vid(vid)
        if safe:
            return f"{safe}.{EFFECTIVE_AUDIO_FORMAT}"
        dt = datetime.fromtimestamp(ts or time.time())
        stamp = dt.strftime("%Y%m%d_%H%M%S")
        return f"conn_{conn_id}_{stamp}.{EFFECTIVE_AUDIO_FORMAT}"

# ---------------------------------------------------------------------------
#  Lazy import elasticsearch
# ---------------------------------------------------------------------------
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    _ES_OK = True
except ImportError:
    _ES_OK = False
    _err("elasticsearch not installed! pip install elasticsearch")


# ===========================================================================
def build_audio_url(vid: str, conn_id: int = 0, ts: float = None) -> str:
    """Build the direct-download URL for an audio file on the storage box."""
    fname = build_audio_filename(vid, conn_id=conn_id, ts=ts)
    if not fname:
        return ""
    return f"{_AUDIO_URL_PREFIX}/{fname}"


# ===========================================================================
class ElasticLogger:
    """Non-blocking Elasticsearch logger with multiple workers + dead letter."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=ELASTIC_MAX_QUEUE)
        self._running = True
        self._workers: List[threading.Thread] = []
        self._deadletter_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self.stats: Dict[str, Any] = {
            "logged": 0,
            "failed": 0,
            "dropped_queue_full": 0,    # NEW: tracks silent drops
            "deadletter_count": 0,      # NEW: docs written to dead letter
            "bulk_operations": 0,
            "bulk_errors": 0,
            "reconnects": 0,
            "last_success_ts": 0,
            "last_error_ts": 0,
            "last_error_msg": "",
        }

        if not _ES_OK:
            _err("Cannot start — elasticsearch package missing")
            return

        # Ensure dead-letter dir exists
        try:
            os.makedirs(os.path.dirname(ELASTIC_DEAD_LETTER) or ".", exist_ok=True)
        except Exception as e:
            _err(f"Could not create dead-letter dir: {e}")

        _info("=" * 70)
        _info(f"  Elasticsearch Logger ENABLED (v9.5 — HEAVY LOAD HARDENED)")
        _info(f"  URL          : {ELASTICSEARCH_URL}")
        _info(f"  Index        : {ELASTIC_INDEX_PREFIX}-YYYY.MM.DD")
        _info(f"  Workers      : {ELASTIC_NUM_WORKERS} parallel")
        _info(f"  Bulk         : size={ELASTIC_BULK_SIZE}  flush={ELASTIC_FLUSH_INTERVAL}s")
        _info(f"  Queue cap    : {ELASTIC_MAX_QUEUE:,}")
        _info(f"  Dead letter  : {ELASTIC_DEAD_LETTER}")
        _info(f"  Audio        : {_AUDIO_URL_PREFIX}/{{VID}}.{EFFECTIVE_AUDIO_FORMAT}")
        _info("=" * 70)

        for i in range(ELASTIC_NUM_WORKERS):
            self._start_worker(i)

    # ------------------------------------------------------------------
    def _process_api_key(self):
        if not ELASTIC_API_KEY:
            return None
        try:
            decoded = base64.b64decode(ELASTIC_API_KEY).decode("utf-8")
            if ":" in decoded:
                key_id, key_secret = decoded.split(":", 1)
                return (key_id, key_secret)
            return ELASTIC_API_KEY
        except Exception:
            return ELASTIC_API_KEY

    def _create_client(self) -> Optional["Elasticsearch"]:
        try:
            api_key = self._process_api_key()
            kwargs: Dict[str, Any] = {
                "hosts": [ELASTICSEARCH_URL],
                "request_timeout": ELASTIC_BULK_TIMEOUT,
                "max_retries": 2,
                "retry_on_timeout": True,
                "verify_certs": False,
                # Keep connection pool small per worker — too many connections
                # to a single ES node is counterproductive.
                "http_compress": True,
            }
            if api_key:
                kwargs["api_key"] = api_key

            client = Elasticsearch(**kwargs)
            if client.ping():
                _info("Connected to Elasticsearch")
            else:
                _err("Elasticsearch ping failed — will retry per batch")
            with self._stats_lock:
                self.stats["reconnects"] += 1
            return client
        except Exception as e:
            _err(f"Client creation failed: {type(e).__name__}: {e}")
            with self._stats_lock:
                self.stats["last_error_ts"] = time.time()
                self.stats["last_error_msg"] = f"client_create: {e}"
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _index_name(ts: float = None) -> str:
        dt = datetime.fromtimestamp(ts or time.time())
        return f"{ELASTIC_INDEX_PREFIX}-{dt.strftime('%Y.%m.%d')}"

    @staticmethod
    def _parse_bracket_ts(s: str) -> Optional[int]:
        if not s:
            return None
        try:
            clean = s.strip("[]")
            year = datetime.now().year
            dt = datetime.strptime(f"{year} {clean}", "%Y %b %d %H:%M:%S")
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def _prepare(self, doc: Dict) -> Dict:
        out = doc.copy()

        ts_sec = out.pop("_ts_sec", time.time())
        out["timestamp"] = int(ts_sec * 1000)
        out["@timestamp"] = datetime.fromtimestamp(ts_sec).isoformat()

        for key in ("audio_start_time_str", "audio_end_time_str"):
            val = out.get(key)
            if val:
                ms = self._parse_bracket_ts(val)
                if ms:
                    base = key.replace("_str", "")
                    out[f"{base}_epoch_ms"] = ms
                    out[f"{base}_iso"] = datetime.fromtimestamp(ms / 1000).isoformat()

        start_ms = out.get("audio_start_time_epoch_ms")
        end_ms = out.get("audio_end_time_epoch_ms")
        if start_ms and end_ms:
            out["audio_processing_duration_ms"] = end_ms - start_ms

        for f in ("confidence", "proba_human", "proba_machine", "audio_duration_sec"):
            if f in out:
                try:
                    out[f] = float(out[f])
                except (ValueError, TypeError):
                    out[f] = 0.0
        for f in ("response_time_ms", "bytes_received", "transcript_length"):
            if f in out:
                try:
                    out[f] = int(out[f])
                except (ValueError, TypeError):
                    out[f] = 0
        if "has_transcript" in out:
            out["has_transcript"] = bool(out["has_transcript"])

        return out

    # ------------------------------------------------------------------
    def _write_to_deadletter(self, docs: List[Dict], reason: str):
        """Persist unrecoverable docs to disk so they're never silently lost."""
        if not docs:
            return
        try:
            with self._deadletter_lock:
                with open(ELASTIC_DEAD_LETTER, "a", encoding="utf-8") as f:
                    for doc in docs:
                        record = {
                            "deadletter_ts": datetime.now().isoformat(),
                            "reason": reason,
                            "doc": doc,
                        }
                        f.write(json.dumps(record, default=str) + "\n")
            with self._stats_lock:
                self.stats["deadletter_count"] += len(docs)
            _err(f"Wrote {len(docs)} doc(s) to dead-letter [{reason}]: {ELASTIC_DEAD_LETTER}")
        except Exception as e:
            _err(f"DEAD-LETTER WRITE FAILED ({e}) — losing {len(docs)} docs!")

    # ------------------------------------------------------------------
    def _bulk_index(self, client, batch: List[Dict]) -> Tuple[int, List[Dict]]:
        """
        Index a batch. Returns (success_count, failed_docs_to_retry).
        Failed docs are returned for retry, not silently dropped.
        """
        if not batch:
            return 0, []
        if client is None:
            return 0, batch  # all need retry

        actions = []
        for doc in batch:
            prepared = self._prepare(doc)
            actions.append({
                "_index": self._index_name(doc.get("_ts_sec", time.time())),
                "_source": prepared,
            })

        try:
            success, errors = bulk(
                client, actions,
                stats_only=False,
                raise_on_error=False,
                raise_on_exception=False,
                max_retries=2,
                request_timeout=ELASTIC_BULK_TIMEOUT,
                chunk_size=ELASTIC_BULK_SIZE,
            )
            with self._stats_lock:
                self.stats["bulk_operations"] += 1
                self.stats["logged"] += success
                if success > 0:
                    self.stats["last_success_ts"] = time.time()

            if errors:
                # Determine which docs failed. The 'errors' list contains
                # detail about which actions failed — we map them back.
                # Simpler: if any error, retry the failed portion.
                err_count = len(errors)
                with self._stats_lock:
                    self.stats["bulk_errors"] += err_count

                # Identify retryable vs permanent errors
                retryable_docs = []
                permanent_docs = []
                # `errors` length is the count that failed; we don't have
                # a reliable index map back to `batch`, so for safety we
                # treat the *last* err_count docs as the failures only if
                # success + err_count == len(batch). Otherwise, retry all.
                if success + err_count == len(batch):
                    # The bulk helper preserves order; failures are the
                    # tail when raise_on_error=False. Be conservative:
                    # inspect each error for permanence.
                    for i, err in enumerate(errors):
                        doc_idx = success + i
                        if doc_idx >= len(batch):
                            break
                        err_info = err.get("index", {}).get("error", {}) if isinstance(err, dict) else {}
                        err_type = err_info.get("type", "")
                        # Permanent errors: bad mapping, validation
                        if err_type in ("mapper_parsing_exception",
                                        "illegal_argument_exception",
                                        "strict_dynamic_mapping_exception",
                                        "document_parsing_exception"):
                            permanent_docs.append(batch[doc_idx])
                        else:
                            retryable_docs.append(batch[doc_idx])
                else:
                    # Can't reliably map — retry all that didn't succeed
                    retryable_docs = batch[success:]

                first = errors[0] if errors else {}
                if isinstance(first, dict) and "index" in first:
                    et = first["index"].get("error", {}).get("type", "?")
                    er = str(first["index"].get("error", {}).get("reason", "?"))[:120]
                    _err(f"Bulk: {success} ok, {err_count} err — first: {et}: {er}")

                if permanent_docs:
                    self._write_to_deadletter(permanent_docs, "permanent_index_error")

                return success, retryable_docs

            return success, []

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            _err(f"Bulk exception: {err_msg}")
            with self._stats_lock:
                self.stats["last_error_ts"] = time.time()
                self.stats["last_error_msg"] = err_msg
                self.stats["bulk_errors"] += 1
            # Network / transient error — retry whole batch
            return 0, batch

    # ------------------------------------------------------------------
    def _drain_batch(self, max_size: int) -> List[Dict]:
        """Pull up to max_size items from queue without blocking long."""
        batch = []
        deadline = time.time() + 0.5  # max 500ms collection window
        while len(batch) < max_size and time.time() < deadline:
            try:
                # First item: wait briefly. Subsequent: don't wait.
                timeout = 0.5 if not batch else 0.0
                item = self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                if batch:
                    break
                # No items at all — return empty
                return batch
        return batch

    # ------------------------------------------------------------------
    def _worker(self, worker_id: int):
        _info(f"Worker {worker_id} started")
        client = self._create_client()
        consecutive_failures = 0
        retry_buffer: List[Dict] = []  # Docs that failed and need re-attempt

        while self._running:
            try:
                # First, try retry buffer (failed batches from previous attempt)
                if retry_buffer:
                    batch = retry_buffer
                    retry_buffer = []
                    _dbg(f"W{worker_id} retrying {len(batch)} buffered docs")
                else:
                    batch = self._drain_batch(ELASTIC_BULK_SIZE)

                if not batch:
                    continue

                # Ensure client healthy
                if client is None:
                    client = self._create_client()
                    if client is None:
                        # Can't connect — don't lose batch, push to retry buffer
                        retry_buffer = batch
                        consecutive_failures += 1
                        # Exponential backoff capped at 30s
                        sleep_s = min(2 ** min(consecutive_failures, 5), 30)
                        _err(f"W{worker_id} no ES client, sleeping {sleep_s}s (retry buffer={len(retry_buffer)})")
                        time.sleep(sleep_s)
                        continue

                success, failed_docs = self._bulk_index(client, batch)

                if failed_docs:
                    # Track retry attempts per-doc
                    for doc in failed_docs:
                        doc["_retry_count"] = doc.get("_retry_count", 0) + 1

                    # Split: those still under retry limit vs dead
                    still_retry = [d for d in failed_docs if d.get("_retry_count", 0) < ELASTIC_MAX_RETRIES]
                    dead = [d for d in failed_docs if d.get("_retry_count", 0) >= ELASTIC_MAX_RETRIES]

                    if dead:
                        self._write_to_deadletter(dead, "max_retries_exceeded")

                    if still_retry:
                        retry_buffer.extend(still_retry)
                        if len(retry_buffer) > ELASTIC_BULK_SIZE * 3:
                            # Retry buffer growing — back off briefly
                            time.sleep(1.0)

                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        # Force client recreation
                        _err(f"W{worker_id} 3 failures, recreating client")
                        try:
                            if client:
                                client.close()
                        except Exception:
                            pass
                        client = None
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0

            except Exception as e:
                # NEVER let the worker die
                _err(f"W{worker_id} unexpected error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

        # Shutdown: flush retry buffer
        if retry_buffer:
            _info(f"W{worker_id} shutdown — flushing {len(retry_buffer)} retry docs")
            if client:
                self._bulk_index(client, retry_buffer)
            else:
                self._write_to_deadletter(retry_buffer, "shutdown_no_client")

        try:
            if client:
                client.close()
        except Exception:
            pass
        _info(f"Worker {worker_id} stopped")

    def _start_worker(self, worker_id: int):
        t = threading.Thread(
            target=self._worker, args=(worker_id,),
            daemon=True, name=f"elastic-logger-{worker_id}"
        )
        t.start()
        self._workers.append(t)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def log_response(
        self,
        response_data: Dict[str, Any],
        conn_id: int,
        customer_number: str,
        audio_duration: float,
        bytes_received: int,
        response_time_ms: int,
        vid: str = "",
        server_ip: str = "",
        audio_start_time_str: str = None,
        audio_end_time_str: str = None,
        customer_name: str = "",
    ):
        """Queue one AMD result. NON-BLOCKING. Drops to dead-letter if queue full."""
        try:
            now = time.time()
            decision = response_data.get("label", response_data.get("decision", "UNKNOWN")).upper()
            audio_url = build_audio_url(vid, conn_id=conn_id, ts=now)

            entry: Dict[str, Any] = {
                "_ts_sec": now,
                "connection_id": str(conn_id),
                "customer_number": customer_number or "unknown",
                "vid": vid or "",
                "vid_sanitized": sanitize_vid(vid),
                "server_ip": server_ip or "",
                "customer_name": customer_name or "",
                "decision": decision,
                "decision_type": "human" if decision == "HUMAN" else "machine",
                "confidence": float(response_data.get("confidence", 0)),
                "proba_human": float(response_data.get("proba_human", 0)),
                "proba_machine": float(response_data.get("proba_machine", 0)),
                "method": response_data.get("method", ""),
                "transcript": (response_data.get("transcript", "") or "")[:500],
                "transcript_length": len(response_data.get("transcript", "") or ""),
                "has_transcript": len(response_data.get("transcript", "") or "") > 0,
                "audio_duration_sec": round(float(audio_duration), 3),
                "bytes_received": int(bytes_received),
                "response_time_ms": int(response_time_ms),
                "audio_url": audio_url,
                "audio_format": EFFECTIVE_AUDIO_FORMAT,
                "audio_start_time_str": audio_start_time_str or "",
                "audio_end_time_str": audio_end_time_str or "",
            }

            if response_data.get("audio_duration_sec"):
                entry["reported_audio_duration_sec"] = float(response_data["audio_duration_sec"])

            # Try to enqueue. If full, write directly to dead letter
            # so we NEVER silently lose a record.
            try:
                self._queue.put_nowait(entry)
            except queue.Full:
                with self._stats_lock:
                    self.stats["dropped_queue_full"] += 1
                # Write to dead letter so it can be replayed later
                self._write_to_deadletter([entry], "queue_full")
                if self.stats["dropped_queue_full"] % 100 == 1:
                    _err(f"QUEUE FULL — total drops: {self.stats['dropped_queue_full']}")
                return

            # Warn if queue is filling up (early signal before drops)
            qsize = self._queue.qsize()
            if qsize > ELASTIC_MAX_QUEUE * ELASTIC_QUEUE_WARN_PCT:
                if qsize % 500 == 0:  # rate-limit log spam
                    _err(f"QUEUE PRESSURE: {qsize}/{ELASTIC_MAX_QUEUE} ({qsize*100//ELASTIC_MAX_QUEUE}%)")

            _dbg(f"Queued: VID={vid!r} | Cust={customer_name!r} | {decision} | qsize={qsize}")

        except Exception as e:
            _err(f"log_response error: {e}")
            with self._stats_lock:
                self.stats["failed"] += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            base_stats = dict(self.stats)
        return {
            **base_stats,
            "queue_size": self._queue.qsize(),
            "queue_max": ELASTIC_MAX_QUEUE,
            "queue_pct": round(self._queue.qsize() * 100 / ELASTIC_MAX_QUEUE, 1),
            "audio_format": EFFECTIVE_AUDIO_FORMAT,
            "is_running": self._running,
            "workers_alive": sum(1 for w in self._workers if w.is_alive()),
            "workers_total": len(self._workers),
            "last_success_age_sec": round(time.time() - base_stats["last_success_ts"], 1) if base_stats["last_success_ts"] else None,
        }

    def stop(self):
        _info("Stopping ...")
        self._running = False
        for t in self._workers:
            t.join(timeout=15.0)
        with self._stats_lock:
            s = dict(self.stats)
        _info(f"Done. logged={s['logged']} failed={s['failed']} "
              f"dropped={s['dropped_queue_full']} deadletter={s['deadletter_count']}")


# ---------------------------------------------------------------------------
_instance = None
_instance_lock = threading.Lock()

def get_elastic_logger():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ElasticLogger()
    return _instance