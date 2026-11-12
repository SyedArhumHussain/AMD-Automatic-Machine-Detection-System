#!/usr/bin/python3

# =============================================================================
# Changes I made Today :  
# Todays Goals = Want to remove the SEND_TIMES to SEND_INTERVAL to Improve the overall connectivity Also Reduced the MAX_WAIT_TIME = 2.6   
# Old AMD.py : { VD_amd.agi hangup the MACHINE calls SEND_INTERVAL = 2.0, FALLBACK_CHUNK_SIZE = 4000 AND process_audio_stream } 
# New AMD.p : { AMDCAUSE and AMDSTAT Send to VD_amd.agi , AMDSTATUS values recognized: "HUMAN"   -> vicidial_log.amd_status = 'HUMAN', returns to dialplan, "MACHINE" -> vicidial_log.amd_status = 'A' / 'AA' / 'AB', drops per campaign, NOAUDIO / NOTSURE mapped to MACHINE
VD.amd.agi only understands two AMDCAUSE , removed safe_set_variable()}
# ISSUES ON LIVE CALLS = Average Call Length for all HA in seconds is Not Improving , Waiting Time Increased , Not Connecting So many calls, 
# Todays Improvements = Agents are going to Long calls, Agents are Getting very less MACHINES, Overall Response Time is 2.6 MS 
# CHANGES IN THE CODE : 295 lines removed from old, 407 lines added in new = 702 total lines touched , +112 lines (old had 348, new has 460)    
# =============================================================================
import os
import sys
import fcntl
import time
import json
import select
import socket
from websocket import create_connection
from asterisk.agi import AGI

# =============================================================================
# CONFIGURATION
# =============================================================================

AUDIO_FD = 3
AUDIO_READ_SIZE = 9500
SAMPLE_RATE = 8000

WS_URL = os.getenv("AMD_WS_URL", "ws://g-lab.arcinventador.com:8989/ws/amd")
CONNECTION_TIMEOUT = 5
WS_RESPONSE_TIMEOUT = 3.0

CHUNK_SIZE = 3200
MAX_WAIT_TIME = 2.6
SELECT_TIMEOUT = 0.02

SERVER_IP_OVERRIDE = os.getenv("AMD_SERVER_IP", "").strip()
DEFAULT_CUSTOMER_NAME = os.getenv("AMD_CUSTOMER_NAME", "MT Marketing")

ASTGUICLIENT_CONF = '/etc/astguiclient.conf'

# AMDCAUSE values that VD_amd.agi understands natively (machine causes)
# These map cleanly to vicidial_log.amd_status = 'A' / 'AA' / 'AB'
VALID_MACHINE_CAUSES = (
    "MACHINE",
    "INITIALSILENCE",
    "LONGGREETING",
    "MAXWORDLENGTH",
    "MAXWORDS",
    "TOOLONG",
)

# AMDCAUSE values that trigger dialplan fallback to built-in AMD()
FALLBACK_CAUSES = ("CONNECTION_ERROR", "INTERR")


# =============================================================================
# SERVER IP DETECTION
# =============================================================================

def detect_server_ip():
    if SERVER_IP_OVERRIDE:
        return SERVER_IP_OVERRIDE
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and ip != "127.0.0.1":
            return ip
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1":
            return ip
    except Exception:
        pass
    return "0.0.0.0"


SERVER_IP = detect_server_ip()


# =============================================================================
# DATABASE
# =============================================================================

def parse_astguiclient_conf():
    config = {}
    try:
        with open(ASTGUICLIENT_CONF, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('VARDB_'):
                    if '=>' in line:
                        key, val = line.split('=>', 1)
                        config[key.strip()] = val.strip()
    except Exception:
        pass
    return config


def get_phone_from_db(callerid, agi=None, vid=None):
    def _log(msg):
        if agi:
            agi.verbose("AMD[{}]: {}".format(vid, msg) if vid else "AMD: {}".format(msg))

    _log("DB lookup: {}".format(callerid))

    config = parse_astguiclient_conf()
    if not config:
        _log("DB ERROR: no config")
        return (None, None)

    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=config.get('VARDB_server', 'localhost'),
            user=config.get('VARDB_user', 'cron'),
            password=config.get('VARDB_pass', ''),
            database=config.get('VARDB_database', 'asterisk'),
            port=int(config.get('VARDB_port', 3306)),
            connection_timeout=2
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT phone_code,phone_number FROM vicidial_auto_calls "
            "WHERE callerid=%s ORDER BY auto_call_id DESC LIMIT 1",
            (callerid,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            _log("DB FOUND: phone={}, code={}".format(row[1], row[0]))
            return (row[1], row[0])
        _log("DB NOT FOUND")
        return (None, None)
    except Exception as err:
        _log("DB ERROR: {}".format(err))
        return (None, None)


# =============================================================================
# LOGGING & VARIABLE HELPERS
# =============================================================================

# Global timer for AMDTOTALTIME calculation
_AMD_START_TIME = None


def log_verbose(agi, message, vid=None):
    msg = str(message)[:300]
    if vid:
        agi.verbose("AMD[{}]: {}".format(vid, msg))
    else:
        agi.verbose("AMD: {}".format(msg))


def _safe_var_value(value):
    """
    VD_amd.agi reads channel variables via Asterisk AGI. The value MUST:
      - Not contain quotes, newlines, carriage returns, or tabs
      - Be reasonably short (Asterisk truncates very long values)
      - Be a plain ASCII string
    """
    if value is None:
        return ""
    s = str(value)
    # Strip anything that could break Asterisk SET VARIABLE parsing
    s = ''.join(
        ch for ch in s
        if ch == ' ' or (ch.isprintable() and ch not in ('"', '\n', '\r', '\t', '\\'))
    )
    s = s.replace('"', "'")
    return s[:200]


def safe_set_variable(agi, name, value, vid=None, verify=False):
    """
    Set an Asterisk channel variable with verification.

    For critical AMD variables (AMDSTATUS, AMDCAUSE), set verify=True to
    confirm the value was actually written. This catches the rare case where
    the AGI socket is half-broken and SET VARIABLE silently fails.
    """
    clean = _safe_var_value(value)
    try:
        agi.set_variable(name, clean)
        if verify:
            try:
                readback = agi.get_variable(name)
                if readback != clean:
                    log_verbose(
                        agi,
                        "WARN: {} readback mismatch: set='{}' got='{}'".format(
                            name, clean, readback
                        ),
                        vid,
                    )
                    # Retry once
                    agi.set_variable(name, clean)
            except Exception:
                pass
        return True
    except Exception as err:
        try:
            agi.verbose("AMD: set_variable({}) failed: {}".format(name, err))
        except Exception:
            pass
        return False


def _normalize_machine_cause(cause):
    """
    Map any internal cause string to a VD_amd.agi-recognized AMDCAUSE.

    VD_amd.agi looks at the FIRST TOKEN of AMDCAUSE (split on '-') and maps it
    against a known list. Causes not in the list still log as MACHINE but
    may not map cleanly to A/AA/AB dispositions.
    """
    if not cause:
        return "MACHINE"

    cause_upper = str(cause).upper().strip()

    # Already a valid machine cause? Pass through.
    for valid in VALID_MACHINE_CAUSES:
        if cause_upper.startswith(valid):
            return cause_upper[:80]

    # Map common ws_server method names to VD_amd.agi vocabulary
    if "SILENCE" in cause_upper or "NOAUDIO" in cause_upper or "FD3_EOF" in cause_upper:
        return "INITIALSILENCE"
    if "BEEP" in cause_upper or "TONE" in cause_upper:
        return "LONGGREETING"
    if "GREETING" in cause_upper or "LONG" in cause_upper:
        return "LONGGREETING"
    if "WORD" in cause_upper:
        return "MAXWORDLENGTH"
    if "TIMEOUT" in cause_upper or "TOOLONG" in cause_upper:
        return "TOOLONG"

    # Default: prepend MACHINE- so VD_amd.agi's first-token parse hits MACHINE
    return ("MACHINE-" + cause_upper)[:80]


def _normalize_human_cause(cause):
    """Map internal cause to VD_amd.agi-friendly HUMAN cause."""
    if not cause:
        return "HUMAN"
    cause_upper = str(cause).upper().strip()
    if cause_upper.startswith("HUMAN"):
        return cause_upper[:80]
    return ("HUMAN-" + cause_upper)[:80]


def set_amd_result(agi, decision, cause, response=None, vid=None):
    """
    Set AMD channel variables for ViciDial's VD_amd.agi to consume.

    STRICT MAPPING:
      AMDSTATUS:
        HUMAN              -> VD_amd.agi returns control to dialplan
        MACHINE            -> VD_amd.agi runs campaign drop logic
        NOAUDIO/NOTSURE    -> Mapped to MACHINE with INITIALSILENCE cause
        HANGUP             -> Mapped to MACHINE

      AMDCAUSE (machine):
        Normalized to VD_amd.agi vocabulary so vicidial_log.amd_status
        properly becomes 'A', 'AA', or 'AB' per campaign settings.

      AMDCAUSE (fallback):
        CONNECTION_ERROR / INTERR -> dialplan jumps to amd_fallback BEFORE
        reaching VD_amd.agi. AMDSTATUS is set to HUMAN as a safe default
        in case the dialplan check fails.

    Also sets AMDTOTALTIME (ms) which VD_amd.agi logs.
    """
    global _AMD_START_TIME

    raw_decision = (decision or "HUMAN").upper()
    raw_cause = (cause or "UNKNOWN").upper()

    # ── DIALPLAN FALLBACK PATH ────────────────────────────────────────────────
    # These causes are intercepted by the dialplan and routed to built-in AMD().
    # We still set AMDSTATUS=HUMAN as a safe default in case the dialplan
    # GotoIf check fails for any reason (e.g. variable not propagated).
    if raw_cause in FALLBACK_CAUSES:
        final_status = "HUMAN"
        final_cause = raw_cause
        log_verbose(
            agi,
            "FALLBACK PATH | AMDSTATUS={} | AMDCAUSE={} (dialplan -> built-in AMD)".format(
                final_status, final_cause
            ),
            vid,
        )

    # ── NORMAL DECISION PATH ──────────────────────────────────────────────────
    else:
        # Normalize decision to VD_amd.agi's two-value vocabulary
        if raw_decision == "HUMAN":
            final_status = "HUMAN"
            final_cause = _normalize_human_cause(raw_cause)
        elif raw_decision == "MACHINE":
            final_status = "MACHINE"
            final_cause = _normalize_machine_cause(raw_cause)
        elif raw_decision in ("NOAUDIO", "NOTSURE", "HANGUP"):
            # VD_amd.agi only understands HUMAN/MACHINE. Map ambiguous
            # results to MACHINE so the campaign drop logic runs (safer
            # than connecting an agent to silence/IVR).
            final_status = "MACHINE"
            if raw_decision == "NOAUDIO":
                final_cause = "INITIALSILENCE"
            elif raw_decision == "HANGUP":
                final_cause = "MACHINE-HANGUP"
            else:
                final_cause = _normalize_machine_cause(raw_cause) if raw_cause != "UNKNOWN" else "MACHINE-NOTSURE"
        else:
            # Unknown decision — fail safe to HUMAN so call connects to agent
            final_status = "HUMAN"
            final_cause = _normalize_human_cause(raw_cause)

    # ── WRITE CHANNEL VARIABLES (with verification on the critical two) ───────
    safe_set_variable(agi, "AMDSTATUS", final_status, vid, verify=True)
    safe_set_variable(agi, "AMDCAUSE",  final_cause,  vid, verify=True)

    # AMDTOTALTIME (ms) — VD_amd.agi reads this for vicidial_log timing
    if _AMD_START_TIME is not None:
        elapsed_ms = int((time.time() - _AMD_START_TIME) * 1000)
    else:
        elapsed_ms = 0
    safe_set_variable(agi, "AMDTOTALTIME", str(elapsed_ms), vid)

    # ── LOGGING ───────────────────────────────────────────────────────────────
    if response and isinstance(response, dict):
        confidence = float(response.get("confidence", 0.5))
        transcript = str(response.get("transcript", ""))[:100]
        method = str(response.get("method", "unknown"))
        srv_elapsed = response.get("elapsed_ms", 0)

        log_verbose(
            agi,
            "SET AMDSTATUS={} | AMDCAUSE={} | AMDTOTALTIME={}ms | conf={:.3f} method={} srv={}ms".format(
                final_status, final_cause, elapsed_ms, confidence, method, srv_elapsed
            ),
            vid,
        )
        if transcript:
            log_verbose(agi, "TRANSCRIPT={}".format(transcript), vid)
    else:
        log_verbose(
            agi,
            "SET AMDSTATUS={} | AMDCAUSE={} | AMDTOTALTIME={}ms (no response data)".format(
                final_status, final_cause, elapsed_ms
            ),
            vid,
        )


def parse_server_response(raw):
    if not raw:
        return None
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        if data.get("type") == "final":
            return data
        if "label" in data or "decision" in data:
            return data
        return None
    except Exception:
        return None


def handle_final_response(agi, response, vid=None):
    """
    Process a final response from the WebSocket server.

    Does NOT hangup — VD_amd.agi handles the drop based on AMDSTATUS/AMDCAUSE
    so vicidial_log and vicidial_auto_calls get the correct AMD disposition.

    Returns True if a final decision was made.
    """
    if not response:
        return False

    label = str(response.get("label", response.get("decision", ""))).upper()
    method = str(response.get("method", "unknown"))

    if label == "NOTSURE":
        log_verbose(agi, "Server NOTSURE -> mapped to MACHINE", vid)
        set_amd_result(agi, "MACHINE", "MACHINE-NOTSURE_{}".format(method), response, vid)
    elif label == "MACHINE":
        log_verbose(agi, "*** MACHINE DETECTED *** via {}".format(method), vid)
        # Let the method name flow through normalization
        set_amd_result(agi, "MACHINE", "MACHINE-{}".format(method.upper()), response, vid)
    elif label == "HUMAN":
        log_verbose(agi, "*** HUMAN DETECTED *** via {}".format(method), vid)
        set_amd_result(agi, "HUMAN", "HUMAN-{}".format(method.upper()), response, vid)
    else:
        log_verbose(agi, "Unknown label '{}' -> MACHINE".format(label), vid)
        set_amd_result(agi, "MACHINE", "MACHINE-UNKNOWN_LABEL", response, vid)

    return True


# =============================================================================
# WEBSOCKET
# =============================================================================

def create_websocket_connection(agi, caller_name, phone, customer_name, vid=None):
    """
    Connect to ws_server.py and send config handshake.
    On failure: sets AMDCAUSE=CONNECTION_ERROR so dialplan falls back to AMD().
    """
    try:
        ws = create_connection(WS_URL, timeout=CONNECTION_TIMEOUT)

        config_data = {
            "config": {
                "Number": phone or "",
                "VID": caller_name or "",
                "ServerIP": SERVER_IP,
                "CustomerName": customer_name or "",
            }
        }
        ws.send(json.dumps(config_data))
        log_verbose(agi, "WebSocket connected to {}".format(WS_URL), vid)
        log_verbose(
            agi,
            "Config: Number={} | VID={} | ServerIP={} | CustomerName={}".format(
                phone or "N/A", caller_name or "N/A", SERVER_IP, customer_name or "N/A"
            ),
            vid,
        )

        # Short ACK wait — audio clock is already ticking
        try:
            ws.settimeout(0.3)
            ack_raw = ws.recv()
            if ack_raw:
                try:
                    if isinstance(ack_raw, (bytes, bytearray)):
                        ack_raw = ack_raw.decode("utf-8", errors="replace")
                    ack = json.loads(ack_raw)
                    if isinstance(ack, dict) and ack.get("type") == "config_ack":
                        log_verbose(agi, "Config ack: {}".format(ack.get("applied", {})), vid)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                ws.settimeout(None)
            except Exception:
                pass

        return ws

    except Exception as err:
        log_verbose(agi, "WebSocket connection failed: {}".format(err), vid)
        log_verbose(agi, "AMD service unreachable - dialplan will fallback to AMD()", vid)
        set_amd_result(agi, "HUMAN", "CONNECTION_ERROR", None, vid)
        return None


def cleanup_websocket(ws):
    if not ws:
        return
    try:
        try:
            ws.send(json.dumps({"type": "flush"}))
        except Exception:
            pass
        ws.close()
    except Exception:
        pass


# =============================================================================
# AUDIO STREAM
# =============================================================================

def setup_audio_stream():
    fcntl.fcntl(AUDIO_FD, fcntl.F_SETFL, os.O_NONBLOCK)


def send_audio_chunk(agi, ws, audio_buffer, vid=None):
    """
    Send binary audio and immediately check for an early server decision.
    """
    try:
        ws.send_binary(audio_buffer)
        log_verbose(agi, "Sent {}b".format(len(audio_buffer)), vid)
    except Exception as err:
        log_verbose(agi, "Audio send error: {}".format(err), vid)
        return {"__error__": "send_failed"}

    try:
        ws.settimeout(0.02)
        raw = ws.recv()
        ws.settimeout(None)
    except Exception:
        try:
            ws.settimeout(None)
        except Exception:
            pass
        return None

    response = parse_server_response(raw)
    if response:
        log_verbose(
            agi,
            "Early decision: label={} method={} conf={}".format(
                response.get("label", response.get("decision", "?")),
                response.get("method", "?"),
                response.get("confidence", "?"),
            ),
            vid,
        )
    return response


def request_flush_and_wait(agi, ws, timeout=WS_RESPONSE_TIMEOUT, vid=None):
    """Send flush and wait for final decision."""
    try:
        ws.send(json.dumps({"type": "flush"}))
        log_verbose(agi, "Flush sent, waiting {}s".format(timeout), vid)
    except Exception as err:
        log_verbose(agi, "Flush send error: {}".format(err), vid)
        return None

    try:
        ws.settimeout(timeout)
        raw = ws.recv()
        ws.settimeout(None)
        return parse_server_response(raw)
    except Exception as err:
        log_verbose(agi, "Flush wait timeout: {}".format(err), vid)
        try:
            ws.settimeout(None)
        except Exception:
            pass
        return None


def process_audio_stream(agi, ws, channel, vid=None):
    """
    Size-based send: every CHUNK_SIZE bytes (~200ms at 8kHz 16-bit).
    """
    start_time = time.time()
    total_data_received = 0
    audio_buffer = b''

    log_verbose(agi, "Audio stream started | channel={}".format(channel), vid)
    log_verbose(agi, "chunk={}b | max_wait={}s".format(CHUNK_SIZE, MAX_WAIT_TIME), vid)

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # ── Hard timeout: flush and decide ────────────────────────────────────
        if elapsed_time >= MAX_WAIT_TIME:
            log_verbose(agi, "Max wait {}s reached, flushing".format(MAX_WAIT_TIME), vid)

            if len(audio_buffer) > 0:
                total_data_received += len(audio_buffer)
                try:
                    ws.send_binary(audio_buffer)
                    audio_buffer = b''
                except Exception:
                    pass

            if total_data_received > 0:
                response = request_flush_and_wait(agi, ws, WS_RESPONSE_TIMEOUT, vid)
                if response and handle_final_response(agi, response, vid):
                    return

            # No decision received — map to MACHINE (TOOLONG) for safety
            set_amd_result(agi, "MACHINE", "TOOLONG", None, vid)
            return

        # ── Read audio from FD3 ───────────────────────────────────────────────
        try:
            ready, _, _ = select.select([AUDIO_FD], [], [], SELECT_TIMEOUT)

            if ready:
                try:
                    audio_chunk = os.read(AUDIO_FD, AUDIO_READ_SIZE)
                    if audio_chunk:
                        audio_buffer += audio_chunk
                    else:
                        # FD3 closed — far end hung up or end of audio
                        log_verbose(agi, "FD3 EOF | total={}b".format(total_data_received), vid)
                        if len(audio_buffer) > 0:
                            total_data_received += len(audio_buffer)
                            try:
                                ws.send_binary(audio_buffer)
                                audio_buffer = b''
                            except Exception:
                                pass
                        if total_data_received > 0:
                            response = request_flush_and_wait(agi, ws, WS_RESPONSE_TIMEOUT, vid)
                            if response and handle_final_response(agi, response, vid):
                                return
                        # FD3 EOF with no decision -> initial silence (machine)
                        set_amd_result(agi, "MACHINE", "INITIALSILENCE", None, vid)
                        return
                except OSError as err:
                    if err.errno == 11:  # EAGAIN
                        pass
                    else:
                        log_verbose(agi, "FD3 error errno={}: {}".format(err.errno, err), vid)
                        raise

        except KeyboardInterrupt:
            log_verbose(agi, "Interrupted", vid)
            # INTERR -> dialplan fallback to built-in AMD()
            set_amd_result(agi, "HUMAN", "INTERR", None, vid)
            return
        except Exception as err:
            log_verbose(agi, "Audio loop error: {}".format(err), vid)
            set_amd_result(agi, "HUMAN", "INTERR", None, vid)
            return

        # ── Size-based send ──────────────────────────────────────────────────
        while len(audio_buffer) >= CHUNK_SIZE:
            chunk_to_send = audio_buffer[:CHUNK_SIZE]
            audio_buffer = audio_buffer[CHUNK_SIZE:]
            total_data_received += len(chunk_to_send)

            log_verbose(
                agi,
                "Send chunk: {}b total={}b elapsed={:.2f}s".format(
                    len(chunk_to_send), total_data_received, elapsed_time
                ),
                vid,
            )

            response = send_audio_chunk(agi, ws, chunk_to_send, vid)

            if response:
                if response.get("__error__"):
                    log_verbose(agi, "Send failed -> CONNECTION_ERROR fallback", vid)
                    set_amd_result(agi, "HUMAN", "CONNECTION_ERROR", None, vid)
                    return
                if handle_final_response(agi, response, vid):
                    return


# =============================================================================
# MAIN
# =============================================================================

def start_agi():
    global _AMD_START_TIME
    _AMD_START_TIME = time.time()

    devnull = open('/dev/null', 'w')
    agi = AGI(stderr=devnull)
    ws = None
    vid = None

    # Pre-set safe defaults IMMEDIATELY so even a fatal crash leaves
    # the channel in a known state for VD_amd.agi / dialplan.
    try:
        safe_set_variable(agi, "AMDSTATUS", "HUMAN")
        safe_set_variable(agi, "AMDCAUSE",  "INTERR")
        safe_set_variable(agi, "AMDTOTALTIME", "0")
    except Exception:
        pass

    try:
        caller_id = agi.env.get('agi_callerid', 'Unknown')
        caller_name = agi.env.get('agi_calleridname', 'Unknown')
        extension = agi.env.get('agi_extension', 'Unknown')
        channel = agi.env.get('agi_channel', 'Unknown')

        vid = caller_name

        # Optional dialplan override of server IP
        try:
            dp_ip = agi.get_variable("SERVER_IP")
            if dp_ip:
                global SERVER_IP
                SERVER_IP = dp_ip.strip() or SERVER_IP
        except Exception:
            pass

        # Customer name resolution
        customer_name = ""
        try:
            customer_name = (agi.get_variable("CUSTOMER_NAME") or "").strip()
        except Exception:
            pass
        if not customer_name:
            customer_name = DEFAULT_CUSTOMER_NAME

        # DB phone lookup
        phone, phone_code = get_phone_from_db(caller_name, agi, vid)

        # Header
        log_verbose(agi, "=" * 50, vid)
        log_verbose(agi, "AMD DETECTION STARTED", vid)
        log_verbose(agi, "Caller ID: {} ({})".format(caller_id, caller_name), vid)
        log_verbose(agi, "Extension: {}".format(extension), vid)
        log_verbose(agi, "Channel: {}".format(channel), vid)
        log_verbose(agi, "Phone: {} (Code: {})".format(phone or "N/A", phone_code or "N/A"), vid)
        log_verbose(agi, "ServerIP: {} | CustomerName: {}".format(SERVER_IP, customer_name), vid)
        log_verbose(agi, "=" * 50, vid)

        # Connect
        ws = create_websocket_connection(agi, caller_name, phone, customer_name, vid)
        if not ws:
            # AMDCAUSE=CONNECTION_ERROR already set — dialplan will fallback
            return

        # Audio FD
        try:
            setup_audio_stream()
        except Exception as err:
            log_verbose(agi, "Audio FD setup failed: {}".format(err), vid)
            # INTERR triggers dialplan fallback
            set_amd_result(agi, "HUMAN", "INTERR", None, vid)
            return

        # Stream + decide
        process_audio_stream(agi, ws, channel, vid)

        log_verbose(agi, "=" * 50, vid)
        log_verbose(agi, "AMD DETECTION COMPLETE - handing off to VD_amd.agi", vid)
        log_verbose(agi, "=" * 50, vid)

    except KeyboardInterrupt:
        try:
            log_verbose(agi, "Top-level interrupt", vid)
            set_amd_result(agi, "HUMAN", "INTERR", None, vid)
        except Exception:
            pass

    except Exception as err:
        try:
            log_verbose(agi, "FATAL ERROR: {}".format(err), vid)
        except Exception:
            pass
        try:
            # INTERR -> dialplan fallback to built-in AMD()
            set_amd_result(agi, "HUMAN", "INTERR", None, vid)
        except Exception:
            pass
    finally:
        cleanup_websocket(ws)


if __name__ == "__main__":
    start_agi()
