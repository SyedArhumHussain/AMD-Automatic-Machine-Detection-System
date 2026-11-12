#!/usr/bin/env python3
# ============================================================
# CHANGES (Concurrency Fixes):
# 1. Removed whisper_transcription_lock — was serialising all
#    transcriptions, causing latency under load.
# 2. Added num_workers=4 to WhisperModel — enables parallel
#    transcription lanes in CTranslate2.
# 3. preprocess_audio + silero_has_speech now run via
#    asyncio.to_thread — keeps event loop responsive.
# ============================================================
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

original_stderr_fd = os.dup(2)
original_stderr = os.fdopen(original_stderr_fd, 'w')

devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull_fd, 2)

import warnings
import json
import time
import asyncio
import concurrent.futures
import threading
import numpy as np
import joblib
import soxr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

import torch


elastic_logger = None
audio_logger = None

ELASTIC_ENABLED = os.getenv("ELASTIC_ENABLED", "true").lower() == "true"
AUDIO_SAVE_ENABLED = os.getenv("AUDIO_SAVE_ENABLED", "true").lower() == "true"
AUDIO_FORMAT = os.getenv("AUDIO_FORMAT", "mp3")
AUDIO_SAVE_FILTER = os.getenv("AUDIO_SAVE_FILTER", "all").lower()

if ELASTIC_ENABLED:
    try:
        from elastic_logger import get_elastic_logger
        elastic_logger = get_elastic_logger()
        print(f"[INIT] Elasticsearch Logging: ENABLED")
    except Exception as e:
        print(f"[INIT] Elasticsearch Logging: DISABLED (import/init failed: {e})")
        ELASTIC_ENABLED = False
else:
    print(f"[INIT] Elasticsearch Logging: DISABLED (env)")

if AUDIO_SAVE_ENABLED:
    try:
        from audio_logger import get_audio_logger
        audio_logger = get_audio_logger()
        print(f"[INIT] Audio Storage: ENABLED ({AUDIO_FORMAT.upper()}, filter={AUDIO_SAVE_FILTER})")
    except Exception as e:
        print(f"[INIT] Audio Storage: DISABLED (import/init failed: {e})")
        AUDIO_SAVE_ENABLED = False
else:
    print(f"[INIT] Audio Storage: DISABLED (env)")


ASTERISK_SR = 8000
WHISPER_SR = 16000
SILERO_SR = 16000

MAX_CALL_SEC = float(os.getenv("MAX_CALL_SEC", "2.5"))

WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", "small.en")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

# Number of parallel transcription workers allowed inside CTranslate2.
# Increase carefully based on GPU VRAM / CPU cores.
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "4"))

CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "../models/text_cls_amd_v2.joblib")

SILERO_VAD_THRESHOLD = float(os.getenv("SILERO_VAD_THRESHOLD", "0.3"))
SILERO_MIN_SPEECH_MS = int(os.getenv("SILERO_MIN_SPEECH_MS", "100"))

SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.003"))
TARGET_RMS = float(os.getenv("TARGET_RMS", "0.08"))
MAX_GAIN = float(os.getenv("MAX_GAIN", "2.5"))
SOXR_QUALITY = os.getenv("SOXR_QUALITY", "HQ")

INITIAL_PROMPT = os.getenv(
    "INITIAL_PROMPT",
    "voicemail, message, beep, tone, press, mailbox, answering machine, automated attendant"
)

MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.15"))
MIN_CONFIDENCE_GREETING = float(os.getenv("MIN_CONFIDENCE_GREETING", "0.50"))
MIN_SPEECH_MS_HUMAN = int(os.getenv("MIN_SPEECH_MS_HUMAN", "700"))
MIN_P2A_RATIO = float(os.getenv("MIN_P2A_RATIO", "3.5"))
NOISE_P2A_THRESHOLD = float(os.getenv("NOISE_P2A_THRESHOLD", "3.2"))
SHORT_SPEECH_THRESHOLD = int(os.getenv("SHORT_SPEECH_THRESHOLD", "500"))
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.52"))

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


HALLUCINATION_SUBSTRINGS = {
    "[music]", "[silence]", "[inaudible]", "[applause]", "[laughter]",
    "thanks for watching", "please subscribe",
    "subtitle by", "translated by",
    "thank you for watching", "like and subscribe",
}

HALLUCINATION_EXACT = {
    "you", "the", "i", "a", "it", "so", "oh", "um", "uh", "ah",
    "...", "..", ".", ",", "!", "?", "-",
    "thank you.", "bye.", "okay.", "yes.", "no.",
    "thank you", "bye", "okay",
    "hi", "hey",
}

GREETING_ONLY_PATTERNS = {
    "hello", "hi", "hey", "hi there", "hello there",
    "yes", "yeah", "yep", "speaking", "who is this", "who is it",
    "hey there", "yo", "whats up", "sup", "yellow",
}

GREETING_WORDS = {
    "hello", "hi", "hey", "yo", "yeah", "yep", "yes", "speaking",
    "sup", "whats", "up", "there",
}


def is_greeting_only(text_clean: str) -> bool:
    if not text_clean:
        return False
    if text_clean in GREETING_ONLY_PATTERNS:
        return True
    words = text_clean.split()
    if not words:
        return False
    return all(w in GREETING_WORDS for w in words)


def has_greeting_prefix(text_clean: str) -> bool:
    if not text_clean:
        return False
    words = text_clean.split()
    if len(words) < 2:
        return False
    return words[0] in {"hello", "hi", "hey", "yo"}


def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ")
    s = s.replace("'", " ").replace("-", " ").replace('"', " ")
    return " ".join(s.split())


def strip_leading_greeting(text_clean: str) -> str:
    if not text_clean:
        return text_clean
    words = text_clean.split()
    if not words:
        return text_clean
    i = 0
    for _ in range(2):
        if i < len(words) and words[i] in {"hello", "hi", "hey", "yo"}:
            i += 1
        else:
            break
    stripped = " ".join(words[i:]).strip()
    return stripped if stripped else text_clean



HUMAN_KEYWORDS = [
    "fuck", "fuck you", "stop calling", "don't call",
    "not interested", "remove me", "leave me alone",
    "who is this", "who's calling", "who am i speaking with",
    "how are you", "can you hear me", "may i help you",
    "what is this regarding", "what can i do for you",
    "hold on", "one moment", "just a second", "bear with me",
    "give me a moment", "let me check", "hang on",
    "be right with you", "i'll find out",
    "is this a good time", "do you have a moment",
    "identify yourself", "can you confirm your name",
    "goodbye",
    "how are", "how's it", "how're you", "how ya", "how you", "you doing",
    "you doing?", "you good?", "you okay?", "all good?", "everything good?",
    "everything okay?", "what's up", "what's new", "what's happening",
    "what's going", "what's good", "what's crackin'", "what's poppin'",
    "sup with", "how goes", "how's life", "how's tricks",
    "oh hey", "oh hi", "oh hello", "well hey", "well hi", "well hello",
    "hey hey", "hi hi", "there you", "there you are", "finally you",
    "you made", "you called", "you rang", "you there", "you made it",
    "oh wow", "hey wow", "hi wow", "hello wow", "well wow", "oh man",
    "hey man", "hi man", "hello man", "well man", "oh boy", "hey boy",
    "hi boy", "hello boy", "well boy", "oh girl", "hey girl", "hi girl",
    "hello girl", "well girl", "oh my", "hey my", "hi my", "hello my",
    "well my", "oh gosh", "hey gosh", "hi gosh", "hello gosh", "well gosh",
    "oh geez", "hey geez", "hi geez", "hello geez", "well geez", "oh brother",
    "hey brother", "hi brother", "hello brother", "well brother", "oh sister",
    "hey sister", "hi sister", "hello sister", "well sister",
]

MACHINE_KEYWORDS = [
    "name and number", "number", "tone", "unavaliable", "avaliable", "forwaord",
    "message", "reached", "number", "person", "trying", "calling", "called",
    "voicemail", "phone", "not avaliable", "mailbox", "not avaliable", "avaliable",
    "leave", "message",
    "box", "system", "mailbox", "automated", "recorded", "attendant",
    "please", "leave", "message", "name", "number", "beep", "tone", "after",
    "record", "end", "saved", "voicemail", "mailbox", "message", "available",
    "unavailable", "beep", "tone", "record", "menu", "extension", "automated",
    "forwarded", "directory", "transferring", "connecting", "redirecting", "goodbye",
    "not", "available", "moment", "unable", "take", "call", "sorry", "missed",
    "currently", "temporarily", "full", "set", "accept", "messages", "voice",
    "messaging", "try", "again", "later", "configured", "beep", "consultant",
    "tone", "at the tone", "you have", "hi you have", "you have reached",
    "please", "monitor",
    "hi this is", "hi, this", "hello this", "hello this is", "hello i am",
    "i am", "I am",
    "now transferring", "now connecting", "redirecting now", "please do not hang",
    "please record your message", "leave a message", "leave your message",
    "after the tone", "after the beep", "at the tone", "record your message",
    "voicemail", "voice mail", "mailbox", "is not available", "not available",
    "cannot take your call", "can't take your call", "unable to take",
    "press one", "press two", "press three", "press four", "press five",
    "press six", "press seven", "press eight", "press nine", "press zero",
    "press pound", "press star", "press hash", "dial one", "dial two",
    "automated", "answering machine", "answering service",
    "if you are calling", "if you re calling", "if you know your party",
    "if you know the extension", "please hold", "please wait",
    "please stay on the line", "your call is important", "your call will be",
    "this call may be recorded", "this call is being recorded",
    "for quality assurance", "menu options", "the person you are trying to reach",
    "the person you have called", "the number you have dialed",
    "the number you are calling", "has not set up", "is full",
    "try again later", "call back later", "office hours", "business hours",
    "currently closed", "thank you for calling", "thanks for calling",
    "if you need", "call", "need", "this is",
    "leave a message after the beep", "leave a message after the tone",
    "please leave a message", "leave your message after",
    "at the tone please record", "record your message",
    "the person you are calling", "the person you have called",
    "is not available", "cannot take your call", "can't take your call",
    "you have reached", "you've reached", "thank you for calling",
    "mailbox is full", "voicemail is full", "has not been set up", "1", "one",
    "2", "two", "3", "three", "4", "four", "5", "five", "6", "six",
    "7", "seven", "8", "eight",
    "leave a message after the beep", "leave a message after the tone",
    "please leave a message", "leave your message after",
    "at the tone please record", "record your message",
    "at the tone please leave the message", "at the tone please leave your message",
    "the person you are calling", "the person you have called",
    "is not available", "cannot take your call", "can't take your call",
    "you have reached", "you've reached", "thank you for calling",
    "mailbox is full", "voicemail is full", "has not been set up",
    "ether", "the person you are trying to call", "that was the unvalid number",
    "press 1", "press 2", "press pound", "press star",
    "for sales", "for support", "for billing", "for more options",
    "to speak to", "to reach", "to leave a message",
    "main menu", "customer service", "please hold",
    "this call may be recorded", "your call is important",
    "press 1 for more options", "press 2 for more options",
    "1 for more options", "for more options", "or press 1 for more",
    "or press 1 for", "or press 1", "or press pound for further",
    "or press pound for", "or press pound", "press pound for further options",
    "press pound for further", "pound for further options", "further options",
    "more options please press", "more options", "press any key",
    "press any key to continue",
    "leave a message", "after the beep", "after the tone",
    "not available", "unavailable", "voicemail", "voice mail",
    "answering machine", "mailbox", "beep", "tone",
    "forwarded", "sorry i missed", "can't answer", "extension", "directory",
    "return your call", "call you back", "get back to you", "pound key",
    "press pound", "record", "please leave", "leave a message",
    "can't come", "can't get", "forwarded", "sorry i miss", "sorry i missed",
    "we are not", "not available", "we're unable", "we are unable",
    "can't take", "can't answer", "voice mail", "voicemail", "leave your name",
    "after the tone", "at the tone", "beep", "pound key", "press pound", "record",
    "please leave", "leave a message", "can't come", "can't get", "forwarded",
    "sorry i miss", "sorry i missed", "we are not", "not available",
    "we're unable", "we are unable", "can't take", "can't answer",
    "voice mail", "voicemail", "leave your name", "after the tone",
    "at the tone", "beep", "leave a message",
    "after the tone", "after the beep", "press 1", "press pound", "voicemail",
    "mailbox", "not available", "sorry i missed", "can't take your call",
    "your call is important", "please hold", "press any key", "automated system",
    "ivr", "this call may be recorded", "return your call", "business hours",
    "office is closed", "welcome to", "thank you for calling", "leave a message",
    "leave your message", "leave me a message", "leave a brief message",
    "leave your name", "leave your number", "leave your name and number",
    "leave your name and phone number", "please leave", "please record",
    "record your message", "start speaking", "begin recording", "message recorded",
    "message saved", "please record your message", "at the tone please record",
    "tone please record your", "tone please record", "record your message when",
    "when you've finished recording", "when you have finished recording",
    "when you are finished recording", "finished recording you may",
    "recording you may hang", "you may hang up or press", "you may hang up or",
    "you may hang up", "may hang up or press", "hang up or press pound",
    "hang up or press 1", "hang up or press", "simply hang up or press",
    "simply hang up or", "simply hang up", "recording simply hang",
    "please leave your name and", "please leave a message", "leave your name and",
    "leave a message and", "please leave your message after",
    "please leave a brief message", "please leave a message after",
    "please leave me a message", "leave your name and phone number",
    "leave your name and a", "leave your message after the",
    "leave your message after", "leave a message after the", "leave a message after",
    "leave me a message and", "your message when you", "your message when",
    "message when you've", "message when you have", "message when you",
    "finished recording", "when you have finished", "when you've finished",
    "you have finished", "have finished recording", "up or press", "or press",
    "return the call thank you", "please press 1 to connect your call",
    "at the tone", "after the tone", "after the beep", "beep",
    "at the tone please", "after the tone please", "message after the tone",
    "message after the beep", "the beep we will", "the beep",
    "the tone please", "the tone", "at the sound of the tone",
    "at the sound of the beep",
    "please speak after the tone", "please speak after the beep",
    "record a message at the tone", "record a message at the beep",
    "when you are done recording", "when you are done",
    "end of message", "to end your recording", "to end recording",
    "to finish recording", "to complete your message",
    "your message will be recorded", "your message will be saved",
    "to save your message", "to review your message",
    "press star to review", "press pound to save",
    "press any key to continue", "press any key",
    "wait for the prompt", "wait for the tone",
    "please wait for the tone", "please wait for the beep",
    "recording will begin after", "recording begins after",
    "you may begin speaking at the tone", "you may begin speaking at the beep",
    "begin speaking at the tone", "begin speaking at the beep",
    "start talking at the tone", "start talking at the beep",
    "at the sound", "after the sound", "sound of the tone",
    "sound of the beep", "you will hear a tone", "you will hear a beep",
    "listen for the tone", "listen for the beep",
    "tone will sound", "beep will sound", "short tone",
    "short beep", "long tone", "long beep",
    "press 0 for operator", "press 0 for assistance",
    "press 9 for directory", "press star for main menu",
    "press star to return", "press star to repeat",
    "to repeat this message", "to hear these options again",
    "for spanish press 2", "para español marque dos",
    "for other languages", "language options",
    "if this is an emergency", "in case of emergency",
    "to make a payment", "for billing inquiries",
    "for technical support", "for sales department",
    "to schedule an appointment", "to cancel an appointment",
    "to speak with a representative", "to speak with someone",
    "to reach a live person", "to talk to a person",
    "is not available", "not available", "unavailable", "not able to come",
    "unable to take your call", "unable to answer", "we are unable",
    "we are not", "can't take your call", "can't answer", "can't come",
    "can't get", "sorry i missed", "sorry i am not available",
    "i'm sorry i missed your call", "we missed your call", "forwarded",
    "nothing has been recorded", "is not available at the tone",
    "is not available at this time", "is not available at",
    "not available at the tone", "not available at the",
    "not available at this", "available at the tone",
    "is not available please", "can't take your call now at",
    "can't take your call now", "cannot take your call",
    "take your call now at the", "take your call now at", "take your call now",
    "currently unavailable", "temporarily unavailable",
    "out of the office", "away from desk", "away from phone",
    "on another call", "on the phone", "busy line",
    "line is busy", "please try again later", "please call back later",
    "office is closed", "office hours are", "business hours",
    "after hours", "holiday schedule", "vacation message",
    "out of town", "traveling", "on vacation", "on leave",
    "on sick leave", "medical leave", "maternity leave",
    "paternity leave", "will be back", "return on",
    "mailbox", "voice mailbox", "voicemail", "mailbox is full",
    "voicemail is full", "the mailbox is full", "cannot accept messages",
    "cannot accept any messages", "is full and cannot accept new messages",
    "voice messaging system", "has not been set up yet goodbye",
    "has not been set up yet", "not been set up yet", "not been set up",
    "mailbox is full and cannot accept", "mailbox is full and cannot",
    "is full and cannot accept new", "is full and cannot accept",
    "cannot accept new messages", "voice mailbox that has not",
    "voice message system", "try again later goodbye",
    "mailbox has not been initialized", "mailbox not configured",
    "voicemail not set up", "personal greeting not set",
    "system greeting", "default greeting", "generic greeting",
    "this number is not in service", "number disconnected",
    "number has been changed", "new number is",
    "this number is no longer in use", "out of service",
    "circuit busy", "all circuits are busy",
    "network busy", "system busy", "high call volume",
    "unexpected error", "system error", "technical difficulties",
    "please try your call again", "unable to complete your call",
    "return your call", "i will return your call", "we will return your call",
    "call you back", "i will call you back", "call me back",
    "get back to you", "i'll get back to you", "as soon as possible",
    "as soon as you can", "right now", "will return your call as soon",
    "return your call as soon", "i'll get back to you as soon",
    "get back to you as soon as", "get back to you as soon",
    "get back with you as soon", "get back with you",
    "call you back as soon", "as soon as possible thank you",
    "soon as possible thank", "soon as possible",
    "will get back to you", "shall return your call",
    "expect a callback", "expect a return call",
    "i'll try you back", "we'll try you back",
    "at my earliest convenience", "at our earliest convenience",
    "when i return", "when we return", "upon my return",
    "as soon as i can", "as soon as we can",
    "at the next opportunity", "next chance i get",
    "when possible", "when i'm available",
    "verification required", "immediate attention required", "legal action",
    "support advisor", "unauthorized", "fraud", "medicare", "benefits",
    "qualify", "amazon", "shopify", "ebay", "to send a message",
    "to send an sms notification", "para continuar en español",
    "screened by smart call blocker", "smart call blocker",
    "say cancel or press", "your call",
    "urgent matter", "important information", "final notice",
    "account suspended", "account compromised", "security alert",
    "warranty expired", "warranty about to expire",
    "credit card offer", "loan approval", "debt consolidation",
    "student loan forgiveness", "social security",
    "irs", "internal revenue service", "tax debt",
    "free trial", "special offer", "limited time",
    "prize winner", "you have won", "congratulations you",
    "tech support", "microsoft support", "apple support",
    "windows support", "virus detected", "malware alert",
    "suspicious activity", "unusual login attempt",
    "press 1 to accept", "press 1 to claim",
    "press 1 to speak with", "press 1 now",
    "to be removed", "to unsubscribe",
    "do not press any keys", "do not hang up",
    "survey", "market research", "opinion poll",
    "political survey", "charity donation",
    "fundraising", "non-profit organization",
    "sales call", "product demonstration",
    "free estimate", "free consultation",
    "no obligation", "no cost",
    "exclusive offer", "pre-approved",
    "doctor's office", "medical practice",
    "dental office", "veterinary clinic",
    "law office", "attorney at law",
    "real estate office", "insurance agency",
    "financial advisor", "investment firm",
    "property management", "home services",
    "utility company", "cable company",
    "internet provider", "phone company",
    "bank", "credit union", "financial institution",
    "office hours", "business hours", "open from",
    "closed on", "weekends", "holidays",
    "eastern time", "pacific time", "central time",
    "mountain time", "standard time", "daylight time",
    "today is", "current date", "current time",
    "please enter", "please say", "please speak",
    "using your keypad", "using your voice",
    "followed by the pound key", "followed by pound",
    "then press pound", "then press hash",
    "to confirm press", "to verify press",
    "to continue press", "to proceed press",
    "to opt out", "to stop receiving calls",
    "transferring", "connecting", "redirecting",
    "please hold while", "one moment please",
    "momentarily", "shortly", "briefly",
    "your call is important to us",
    "all agents are currently busy",
    "next available representative",
    "you've reached", "thank you for calling", "have a blessed day",
    "have a great day", "have a wonderful day", "have a good day",
    "thank you and have a", "you've reached the voicemail", "you have reached",
    "sorry i missed your call please", "sorry we missed your call",
    "i missed your call", "welcome to", "thank you for contacting",
    "thanks for calling", "appreciate your call",
    "good day", "take care", "talk to you soon",
    "looking forward to", "speak with you soon",
    "have a nice day", "have a productive day",
    "best regards", "warm regards", "sincerely",
    "respectfully", "yours truly",
    "welcome to our automated system", "automated attendant",
    "interactive voice response", "ivr system",
    "for quality assurance", "for training purposes",
    "calls are monitored", "calls are recorded",
    "your call will be answered", "in the order it was received",
    "estimated wait time", "current wait time",
    "you are number", "in the queue",
    "please remain on the line", "please do not hang up",
    "your call is being transferred", "transferring your call",
    "connecting you now", "please wait while i connect you",
    "press 1 to", "press 2 to", "to connect your call",
    "your call is important",
    "this call may be recorded", "this is an automated call", "main menu",
    "customer service", "representative", "stay on the line", "please hold",
    "enter your", "select from", "choose from", "say your selection",
    "if you know your party's extension", "to connect your call please",
    "press 1 to connect",
    "business hours", "office hours",
    "currently closed", "after hours",
    "press 1 to accept", "press 2 to decline",
    "say yes", "say no",
    "for security", "for verification",
    "to protect", "to ensure",
    "special offer", "limited time",
    "act now", "don't miss",
    "estimated wait", "you are number",
    "in the queue", "all agents busy",
    "if you know your party's",
    "please stay on the line",
    "to speak with a representative",
    "now transferring", "now connecting",
    "redirecting now", "please do not hang", "please record your message",
    "not avaliabe at the momnet",
]

STANDALONE_NUMBER_WORDS = {
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "pound", "hash", "star", "0",
}


def _prepare_keywords(raw_list):
    cleaned = []
    seen = set()
    for kw in raw_list:
        k = normalize_text(kw)
        if not k or len(k) < 2:
            continue
        if k == "this is":
            continue
        if k not in seen:
            seen.add(k)
            cleaned.append(k)
    cleaned.sort(key=lambda x: (-len(x.split()), -len(x)))
    return cleaned


HUMAN_KEYWORDS = _prepare_keywords(HUMAN_KEYWORDS)
MACHINE_KEYWORDS = _prepare_keywords(MACHINE_KEYWORDS)


_NUMERIC_TOKENS = STANDALONE_NUMBER_WORDS | {
    "one", "e", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion",
    "double", "triple", "oh", "o",
}


def is_standalone_number(text_clean: str) -> bool:
    words = text_clean.split()
    if len(words) >= 1 and len(words) <= 2:
        return all(w in STANDALONE_NUMBER_WORDS for w in words)
    return False


def is_all_numbers(text_clean: str) -> bool:
    if not text_clean:
        return False
    words = text_clean.split()
    if not words:
        return False
    for w in words:
        if w in _NUMERIC_TOKENS:
            continue
        if w.isdigit():
            continue
        return False
    return True



app = FastAPI(title="AMD Whisper WS - Production v9.4")

# ---------------------------------------------------------------------------
#  Dedicated thread pool for Elasticsearch logging and audio saving,
#  so heavy I/O never starves the main async event loop.
# ---------------------------------------------------------------------------
log_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=16, thread_name_prefix="es-log"
)

print("=" * 70)
print(" AMD Whisper WS Server - Production v9.4")
print(f" Whisper: {WHISPER_MODEL_DIR} | Device: {DEVICE} | num_workers={WHISPER_NUM_WORKERS}")
print(f" Timing: MAX_CALL_SEC={MAX_CALL_SEC}s (single transcribe window)")
print(f" VAD: Silero (threshold={SILERO_VAD_THRESHOLD}, min_speech={SILERO_MIN_SPEECH_MS}ms)")
print(f" Silence RMS: {SILENCE_RMS_THRESHOLD}")
print(f" Hallucination Prevention: P2A>{MIN_P2A_RATIO}, Speech>{MIN_SPEECH_MS_HUMAN}ms, Conf>{MIN_CONFIDENCE_GREETING}")
print(f" Greeting-only: P2A < {NOISE_P2A_THRESHOLD} → MACHINE else HUMAN")
print(f" Default Decision: NOTSURE (when no matches found)")
print(f" Classification: MACHINE kw → HUMAN kw → ML → default NOTSURE")
print(f" Server IP tracking: ENABLED (per-connection from amd.py config)")
print(f" Customer Name tracking: ENABLED (new in v9.2)")
print(f" Concurrency: Whisper lock REMOVED — parallel transcription enabled")
print(f" Preprocessing: ASYNC (offloaded to worker threads)")
print(f" Debug: {DEBUG}")
print(f" Elasticsearch: {'ENABLED' if ELASTIC_ENABLED else 'DISABLED'}")
print(f" Audio Storage: {'ENABLED' if AUDIO_SAVE_ENABLED else 'DISABLED'} ({AUDIO_FORMAT.upper() if AUDIO_SAVE_ENABLED else 'N/A'}, filter={AUDIO_SAVE_FILTER})")
print("=" * 70)

# ---------------------------------------------------------------------------
# Whisper model — num_workers > 1 lets CTranslate2 process multiple
# transcription requests in parallel. The asyncio whisper lock has been
# removed (v9.4) so calls now process concurrently instead of single-file.
# ---------------------------------------------------------------------------
whisper_model = WhisperModel(
    WHISPER_MODEL_DIR,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    num_workers=WHISPER_NUM_WORKERS,
)
print(f"Whisper model loaded (num_workers={WHISPER_NUM_WORKERS})")

# Lock to protect Silero VAD (model is not thread‑safe across reset_states)
silero_lock = threading.Lock()

# NOTE (v9.4): whisper_transcription_lock has been INTENTIONALLY REMOVED.
# CTranslate2's WhisperModel is safe for concurrent calls when configured
# with num_workers > 1, and the previous asyncio.Lock was serialising every
# transcription, which caused cascading latency under load.

silero_vad_model = None
get_speech_timestamps = None
silero_ready = False
SILERO_LOCAL_DIR = os.getenv("SILERO_VAD_DIR", "")


def _load_silero_vad():
    if SILERO_LOCAL_DIR and os.path.isdir(SILERO_LOCAL_DIR):
        try:
            model, utils = torch.hub.load(
                repo_or_dir=SILERO_LOCAL_DIR, model='silero_vad',
                source='local', onnx=False,
            )
            print(f"Silero VAD loaded from local: {SILERO_LOCAL_DIR}")
            return model, utils[0]
        except Exception as e:
            print(f"[WARN] Silero local load failed: {e}")

    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            force_reload=False, onnx=False,
        )
        print("Silero VAD loaded from torch.hub")
        return model, utils[0]
    except Exception as e:
        print(f"[WARN] Silero torch.hub load failed: {e}")

    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            force_reload=False, onnx=False, trust_repo=True,
        )
        print("Silero VAD loaded from torch.hub (trust_repo=True)")
        return model, utils[0]
    except Exception as e:
        print(f"[WARN] Silero torch.hub (trust_repo) failed: {e}")

    return None, None


silero_vad_model, get_speech_timestamps = _load_silero_vad()
if silero_vad_model is not None and get_speech_timestamps is not None:
    silero_ready = True
    print(f"Silero VAD ready (threshold={SILERO_VAD_THRESHOLD})")
else:
    print("[WARN] Silero VAD NOT available — will rely on RMS + Whisper internal VAD only")

clf = None
vectorizer = None
classifier_ready = False

try:
    if os.path.exists(CLASSIFIER_PATH):
        bundle = joblib.load(CLASSIFIER_PATH)
        if isinstance(bundle, dict):
            clf = bundle.get("clf") or bundle.get("classifier") or bundle.get("model")
            vectorizer = bundle.get("vectorizer") or bundle.get("vec")
        if clf and vectorizer:
            try:
                test_transform = vectorizer.transform(["test"])
                actual_features = test_transform.shape[1]
                if hasattr(clf, "n_features_in_"):
                    expected = clf.n_features_in_
                elif hasattr(clf, "coef_"):
                    expected = clf.coef_.shape[1]
                else:
                    expected = actual_features
                if actual_features == expected:
                    classifier_ready = True
                    print(f"ML Classifier loaded ({actual_features} features)")
                else:
                    print(f"[WARN] Feature mismatch: Vec={actual_features}, Clf={expected}")
            except Exception as e:
                print(f"[WARN] Classifier validation failed: {e}")
        else:
            print("[WARN] Classifier or vectorizer missing - keyword-only mode")
    else:
        print("[WARN] No classifier found - keyword-only mode")
except Exception as e:
    print(f"[WARN] Classifier load error: {e}")

print(f"Keywords: HUMAN={len(HUMAN_KEYWORDS)} | MACHINE={len(MACHINE_KEYWORDS)}")
print("Ready for connections\n")



def pcm16_to_float32(b: bytes) -> np.ndarray:
    if not b:
        return np.array([], dtype=np.float32)
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def resample_8k_to_16k(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio
    return soxr.resample(audio, ASTERISK_SR, WHISPER_SR, quality=SOXR_QUALITY).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def normalize_audio(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    if len(audio) == 0:
        return audio
    current_r = rms(audio)
    if current_r < 1e-6:
        return audio
    gain = min(target_rms / current_r, MAX_GAIN)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def preprocess_audio(audio_8k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(audio_8k) == 0:
        return audio_8k, np.array([], dtype=np.float32)
    if len(audio_8k) < int(0.15 * ASTERISK_SR):
        return audio_8k, np.array([], dtype=np.float32)
    normalized = normalize_audio(audio_8k)
    resampled = resample_8k_to_16k(normalized)
    return normalized, resampled


def silero_has_speech(audio_16k: np.ndarray) -> Tuple[bool, float]:
    if not silero_ready or silero_vad_model is None or get_speech_timestamps is None:
        return True, 0.0
    if len(audio_16k) < int(0.1 * SILERO_SR):
        return False, 0.0
    try:
        audio_tensor = torch.from_numpy(audio_16k).float()
        with silero_lock:
            silero_vad_model.reset_states()
            timestamps = get_speech_timestamps(
                audio_tensor, silero_vad_model,
                sampling_rate=SILERO_SR,
                threshold=SILERO_VAD_THRESHOLD,
                min_speech_duration_ms=SILERO_MIN_SPEECH_MS,
                min_silence_duration_ms=200,
            )
        if not timestamps:
            return False, 0.0
        total_speech_samples = sum(ts['end'] - ts['start'] for ts in timestamps)
        speech_duration_ms = (total_speech_samples / SILERO_SR) * 1000.0
        return True, speech_duration_ms
    except Exception as e:
        print(f"[VAD ERROR] {type(e).__name__}: {e}")
        return True, 0.0



def is_hallucination(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    t_clean = normalize_text(t)
    if len(t_clean) < 2:
        return True
    if t_clean in HALLUCINATION_EXACT:
        return True
    for h in HALLUCINATION_SUBSTRINGS:
        if h in t:
            return True
    clean_no_space = t.replace(" ", "")
    if len(clean_no_space) > 3 and len(set(clean_no_space)) <= 1:
        return True
    words = t.split()
    if len(words) >= 5:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
    return False


def transcribe_audio(audio_16k: np.ndarray, use_vad: bool = True) -> Tuple[str, float, float]:
    if len(audio_16k) < int(0.2 * WHISPER_SR):
        return "", 0.0, 0.0
    total_duration_sec = len(audio_16k) / WHISPER_SR

    def _transcribe(vad_on: bool) -> Tuple[str, float, float]:
        segments, info = whisper_model.transcribe(
            audio_16k, language="en", beam_size=3, best_of=1,
            temperature=[0.0, 0.2, 0.4],
            initial_prompt=INITIAL_PROMPT,
            condition_on_previous_text=False,
            vad_filter=vad_on,
            vad_parameters={
                "threshold": 0.25,
                "min_speech_duration_ms": 80,
                "min_silence_duration_ms": 300,
            } if vad_on else None,
            word_timestamps=False,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.2,
            no_speech_threshold=0.5,
            repetition_penalty=1.2,
            patience=1.0,
        )
        segment_list = list(segments)
        texts = []
        confidences = []
        total_speech_time = 0.0
        for seg in segment_list:
            text = seg.text.strip()
            total_speech_time += (seg.end - seg.start)
            if text and not is_hallucination(text):
                texts.append(text)
                if hasattr(seg, "avg_logprob"):
                    conf = min(1.0, max(0.0, 1.0 + seg.avg_logprob))
                    confidences.append(conf)
                else:
                    confidences.append(0.8)
        full_text = " ".join(texts).strip()
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        speech_ratio = total_speech_time / total_duration_sec if total_duration_sec > 0 else 0.0
        if is_hallucination(full_text):
            return "", 0.0, speech_ratio
        return full_text, avg_conf, speech_ratio

    try:
        text, conf, ratio = _transcribe(vad_on=use_vad)
        if not text and use_vad and len(audio_16k) >= int(0.5 * WHISPER_SR):
            text, conf, ratio = _transcribe(vad_on=False)
        return text, conf, ratio
    except Exception as e:
        print(f"[TRANSCRIBE ERROR] {type(e).__name__}: {e}")
        return "", 0.0, 0.0


def _match_keywords(text_clean: str, text_stripped: str, kw_list: list, label: str) -> Optional[Tuple[str, float, str]]:
    padded = f" {text_clean} "
    padded_stripped = f" {text_stripped} "
    for kw in kw_list:
        kwp = f" {kw} "
        if kwp in padded or kwp in padded_stripped:
            wcount = len(kw.split())
            conf = 0.95 if wcount > 3 else (0.90 if wcount > 1 else 0.85)
            return label, conf, kw
    return None


def classify_ml(text: str) -> Tuple[str, float, bool]:
    global clf, vectorizer, classifier_ready
    if not classifier_ready or clf is None or vectorizer is None:
        return "MACHINE", 0.75, False          # fallback, actual default handled later
    if not text or len(text.strip()) < 2:
        return "MACHINE", 0.75, False
    try:
        X = vectorizer.transform([text])
        pred = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0]
        confidence = float(proba[pred])
        decision = "MACHINE" if pred == 1 else "HUMAN"
        return decision, confidence, True
    except Exception as e:
        print(f"[ML ERROR] {type(e).__name__}: {e}")
        return "MACHINE", 0.75, False


def classify_transcript(text: str, whisper_conf: float, speech_ratio: float,
                        p2a: float, speech_ms: float) -> Tuple[str, float, str]:
    if not text or len(text.strip()) < 2 or is_hallucination(text):
        print(f"[label] Empty/hallucination → MACHINE")
        return "MACHINE", 0.85, "empty_or_hallucination"

    text_clean = normalize_text(text)
    if len(text_clean) < 2:
        print(f"[label] Normalized too short → MACHINE")
        return "MACHINE", 0.85, "empty_or_hallucination"

    if is_all_numbers(text_clean):
        print(f"[label] All-numbers '{text_clean}' → MACHINE")
        return "MACHINE", 0.90, f"all_numbers:{text_clean}"

    if is_standalone_number(text_clean):
        print(f"[label] Standalone number '{text_clean}' → MACHINE")
        return "MACHINE", 0.85, f"standalone_number:{text_clean}"


    if is_greeting_only(text_clean):
        if p2a < NOISE_P2A_THRESHOLD:
            print(f"[label] Greeting-only with low P2A ({p2a:.2f}) → MACHINE")
            return "MACHINE", 0.80, "greeting_p2a_machine"
        else:
            print(f"[label] Greeting-only with high P2A ({p2a:.2f}) → HUMAN")
            return "HUMAN", 0.85, "greeting_p2a_human"

    text_stripped = strip_leading_greeting(text_clean)
    if has_greeting_prefix(text_clean) and text_stripped != text_clean:
        m = _match_keywords(text_stripped, text_stripped, MACHINE_KEYWORDS, "MACHINE")
        if m:
            print(f"[label] Greeting + MACHINE kw '{m[2]}' → MACHINE")
            return m[0], m[1], f"greeting_machine_kw:{m[2]}"
        m = _match_keywords(text_stripped, text_stripped, HUMAN_KEYWORDS, "HUMAN")
        if m:
            print(f"[label] Greeting + HUMAN kw '{m[2]}' → HUMAN")
            return m[0], m[1], f"greeting_human_kw:{m[2]}"

    m = _match_keywords(text_clean, text_stripped, MACHINE_KEYWORDS, "MACHINE")
    if m:
        print(f"[label] MACHINE keyword '{m[2]}' (conf={m[1]:.2f})")
        return m[0], m[1], f"keyword:{m[2]}"

    m = _match_keywords(text_clean, text_stripped, HUMAN_KEYWORDS, "HUMAN")
    if m:
        print(f"[label] HUMAN keyword '{m[2]}' (conf={m[1]:.2f})")
        return m[0], m[1], f"keyword:{m[2]}"

    ml_decision, ml_conf, ml_ok = classify_ml(text)
    if ml_ok and ml_conf >= 0.70:
        print(f"[label] ML → {ml_decision} (conf={ml_conf:.2f})")
        return ml_decision, ml_conf, "ml_classifier"

    # ============================================================
    #  DEFAULT: if nothing matched, return NOTSURE
    # ============================================================
    print(f"[label] No match → default NOTSURE")
    return "NOTSURE", 0.50, "default_notsure"



def build_response(
    decision: str, confidence: float, transcript: str,
    method: str, elapsed_ms: int,
    metrics: Optional[Dict] = None, number: str = "", server_ip: str = "",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    customer_name: str = "",
) -> Dict[str, Any]:
    decision = decision.upper()
    proba_human = confidence if decision == "HUMAN" else 1.0 - confidence
    proba_machine = confidence if decision == "MACHINE" else 1.0 - confidence

    resp = {
        "type": "final",
        "label": decision,
        "decision": decision,
        "confidence": round(float(confidence), 3),
        "proba_human": round(proba_human, 3),
        "proba_machine": round(proba_machine, 3),
        "transcript": (transcript or "")[:100],
        "method": method,
        "elapsed_ms": int(elapsed_ms),
        "number": number,
        "server_ip": server_ip,
    }

    if customer_name:
        resp["customer_name"] = customer_name

    if start_time:
        resp["audio_start_time"] = start_time.strftime("[%b %d %H:%M:%S]")
        resp["audio_start_timestamp"] = start_time.isoformat()
    if end_time:
        resp["audio_end_time"] = end_time.strftime("[%b %d %H:%M:%S]")
        resp["audio_end_timestamp"] = end_time.isoformat()
    if start_time and end_time:
        resp["audio_duration_sec"] = round((end_time - start_time).total_seconds(), 2)

    if DEBUG and metrics:
        resp["metrics"] = metrics
    return resp


def apply_runtime_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global MAX_CALL_SEC, SILENCE_RMS_THRESHOLD
    allowed = {
        "MAX_CALL_SEC": (1.0, 10.0),
        "SILENCE_RMS_THRESHOLD": (0.0, 0.05),
    }
    applied = {}
    for k, v in (cfg or {}).items():
        if k not in allowed:
            continue
        lo, hi = allowed[k]
        try:
            fv = max(lo, min(hi, float(v)))
            globals()[k] = fv
            applied[k] = fv
        except Exception:
            continue
    return applied



def _should_save_audio(decision: str) -> bool:
    if AUDIO_SAVE_FILTER == "all":
        return True
    if AUDIO_SAVE_FILTER == "machine" and decision == "MACHINE":
        return True
    if AUDIO_SAVE_FILTER == "human" and decision == "HUMAN":
        return True
    return False


def _log_to_elasticsearch(resp: Dict, conn_id: int, caller_number: str,
                          caller_vid: str, server_ip: str, customer_name: str,
                          raw_audio_len: int, bytes_received: int, elapsed_ms: int):
    """Safe wrapper: log to Elasticsearch. Never raises."""
    if not ELASTIC_ENABLED or elastic_logger is None:
        return
    try:
        dur = raw_audio_len / ASTERISK_SR if raw_audio_len > 0 else 0.0
        elastic_logger.log_response(
            response_data=resp,
            conn_id=conn_id,
            customer_number=caller_number,
            audio_duration=dur,
            bytes_received=bytes_received,
            response_time_ms=elapsed_ms,
            vid=caller_vid,
            server_ip=server_ip,
            customer_name=customer_name,
            audio_start_time_str=resp.get("audio_start_time"),
            audio_end_time_str=resp.get("audio_end_time"),
        )
    except Exception as e:
        print(f"[ELASTIC LOG ERROR] {type(e).__name__}: {e}")


def _save_audio_if_needed(raw_audio: np.ndarray, resp: Dict, conn_id: int,
                          caller_number: str, caller_vid: str, server_ip: str,
                          customer_name: str):
    """Safe wrapper: save audio via SFTP. Never raises."""
    if not AUDIO_SAVE_ENABLED or audio_logger is None:
        return
    if len(raw_audio) == 0:
        return
    decision = resp.get("label", resp.get("decision", "UNKNOWN")).upper()
    if not _should_save_audio(decision):
        print(f"[AUDIO-SKIP] Filter={AUDIO_SAVE_FILTER}, Decision={decision}")
        return
    try:
        print(f"[AUDIO-SAVE] Queuing: VID={caller_vid} | Num={caller_number} | SrvIP={server_ip} | Cust={customer_name} | {decision} | {len(raw_audio)/ASTERISK_SR:.1f}s")
        audio_logger.log_audio(
            audio_data=raw_audio,
            conn_id=conn_id,
            customer_number=caller_number,
            vid=caller_vid,
            server_ip=server_ip,
            decision=decision,
            transcript=resp.get("transcript", ""),
            confidence=resp.get("confidence", 0.0),
            customer_name=customer_name,
        )
    except Exception as e:
        print(f"[AUDIO-SAVE ERROR] {type(e).__name__}: {e}")



# ============================================================
#  process_final_audio – fully async, with preprocessing and
#  transcription both off‑loaded to worker threads so the
#  ASGI event loop stays responsive under heavy load.
# ============================================================
async def process_final_audio(raw_audio: np.ndarray, elapsed_ms: int, caller_number: str,
                              server_ip: str = "", audio_start_time: datetime = None,
                              customer_name: str = "") -> Dict[str, Any]:
    audio_end_time = datetime.now()
    if audio_start_time is None:
        audio_start_time = datetime.now() - timedelta(milliseconds=elapsed_ms)

    if len(raw_audio) == 0:
        print(f"[PROCESS] No audio received → MACHINE")
        return build_response("MACHINE", 0.85, "", "no_audio", elapsed_ms,
                              number=caller_number, server_ip=server_ip,
                              start_time=audio_start_time, end_time=audio_end_time,
                              customer_name=customer_name)

    duration = len(raw_audio) / ASTERISK_SR

    if len(raw_audio) < int(ASTERISK_SR * 0.15):
        print(f"[PROCESS] Audio too short ({duration:.2f}s) → MACHINE")
        return build_response("MACHINE", 0.85, "", "too_short", elapsed_ms,
                              number=caller_number, server_ip=server_ip,
                              start_time=audio_start_time, end_time=audio_end_time,
                              customer_name=customer_name)

    # ---- v9.4 CHANGE: preprocessing now async to avoid blocking event loop ----
    proc_8k, proc_16k = await asyncio.to_thread(preprocess_audio, raw_audio)
    # ---------------------------------------------------------------------------

    r = rms(proc_8k)

    peak = np.max(np.abs(proc_8k))
    avg = np.mean(np.abs(proc_8k))
    p2a = float(peak / avg) if avg > 1e-6 else 0.0

    print(f"[PROCESS] Audio={duration:.2f}s | RMS={r:.4f} | P2A={p2a:.2f}")

    if r < SILENCE_RMS_THRESHOLD:
        print(f"[PROCESS] Silence (rms={r:.4f}) → MACHINE")
        return build_response("MACHINE", 0.90, "", "silence", elapsed_ms,
                              number=caller_number, server_ip=server_ip,
                              start_time=audio_start_time, end_time=audio_end_time,
                              customer_name=customer_name)

    if len(proc_16k) == 0:
        print(f"[PROCESS] No 16k audio after preprocess → MACHINE")
        return build_response("MACHINE", 0.85, "", "preprocess_empty", elapsed_ms,
                              number=caller_number, server_ip=server_ip,
                              start_time=audio_start_time, end_time=audio_end_time,
                              customer_name=customer_name)

    # Silero VAD also runs in a worker thread (it holds its own internal lock)
    has_speech, speech_ms = await asyncio.to_thread(silero_has_speech, proc_16k)
    print(f"[VAD] has_speech={has_speech} | speech_duration={speech_ms:.0f}ms")

    if not has_speech:
        print(f"[PROCESS] Silero VAD: no speech detected → MACHINE")
        return build_response("MACHINE", 0.90, "", "silero_no_speech", elapsed_ms,
                              number=caller_number, server_ip=server_ip,
                              start_time=audio_start_time, end_time=audio_end_time,
                              customer_name=customer_name)

    # ---- v9.4 CHANGE: whisper_transcription_lock removed.
    # CTranslate2 with num_workers > 1 handles concurrent calls safely. ----
    text, wconf, speech_ratio = await asyncio.to_thread(
        transcribe_audio, proc_16k, True
    )
    # ----------------------------------------------------------------------

    print(f"[Transcript] '{text}' (whisper_conf={wconf:.2f}, speech_ratio={speech_ratio:.2%})")

    decision, conf, method = classify_transcript(text, wconf, speech_ratio, p2a, speech_ms)

    print(f"[Process] {decision} | Text: '{text}' | Conf: {wconf:.2f} | P2A: {p2a:.2f} | MS: {speech_ms:.0f} | Ratio: {speech_ratio:.2%}")
    print(f"[Final label] {decision} via {method}")

    return build_response(decision, conf, text or "", method, elapsed_ms,
                          number=caller_number, server_ip=server_ip,
                          start_time=audio_start_time, end_time=audio_end_time,
                          customer_name=customer_name)


@app.websocket("/ws/amd")
async def ws_amd(ws: WebSocket):
    await ws.accept()
    conn_id = id(ws)

    # Capture peer IP from socket as a sane default for server_ip
    peer_ip = ""
    try:
        if ws.client:
            peer_ip = ws.client.host or ""
    except Exception:
        pass

    print(f"\n[WS CONNECT] {conn_id} | peer={peer_ip}")

    raw_audio = np.array([], dtype=np.float32)
    start_time = time.time()
    decision_sent = False
    bytes_received = 0
    chunks_received = 0
    caller_number = ""
    caller_vid = ""
    server_ip = ""  # Will be filled from config; falls back to peer_ip
    customer_name = ""
    audio_start_time = datetime.now()

    try:
        while True:
            elapsed = time.time() - start_time
            elapsed_ms = int(elapsed * 1000)

            if elapsed > MAX_CALL_SEC:
                print(f"[TIMEOUT] VID={conn_id} | Number={caller_number} | CallerVID={caller_vid} | SrvIP={server_ip} | Cust={customer_name} | {MAX_CALL_SEC}s reached")
                break

            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if msg.get("type") == "websocket.disconnect":
                print(f"[WS CLOSE] {conn_id} | Number={caller_number} | VID={caller_vid} | SrvIP={server_ip} | Cust={customer_name}")
                break

            # === TEXT MESSAGES: config or flush ===
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])

                    # Config handshake
                    if isinstance(data, dict) and "config" in data:
                        cfg = data.get("config", {})
                        caller_number = str(cfg.pop("Number", "")).strip()
                        caller_vid = str(cfg.pop("VID", "")).strip()
                        client_server_ip = str(cfg.pop("ServerIP", "")).strip()
                        customer_name = str(cfg.pop("CustomerName", "")).strip()
                        server_ip = client_server_ip or peer_ip

                        applied = apply_runtime_config(cfg)
                        ack = {
                            "type": "config_ack",
                            "applied": applied,
                            "number": caller_number,
                            "server_ip": server_ip,
                            "customer_name": customer_name,
                        }
                        await ws.send_text(json.dumps(ack))
                        print(f"[CONFIG] VID={conn_id} | Number={caller_number} | CallerVID={caller_vid} | SrvIP={server_ip} | CustomerName={customer_name} | applied={applied}")
                        continue

                    # Flush signal — process all collected audio
                    if data.get("type") == "flush":
                        print(f"[FLUSH] {conn_id} | Number={caller_number} | VID={caller_vid} | SrvIP={server_ip} | Cust={customer_name}")
                        if not decision_sent:
                            elapsed_ms = int((time.time() - start_time) * 1000)

                            resp = await process_final_audio(raw_audio, elapsed_ms,
                                                              caller_number, server_ip,
                                                              audio_start_time, customer_name)
                            await ws.send_text(json.dumps(resp))
                            print(f"[FINAL] {resp['decision']} via {resp['method']} (conf={resp['confidence']:.3f})")
                            decision_sent = True

                            # Off‑load logging and saving so event loop stays responsive,
                            # using the dedicated thread pool.
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                log_executor, _log_to_elasticsearch, resp, conn_id,
                                caller_number, caller_vid, server_ip, customer_name,
                                len(raw_audio), bytes_received, elapsed_ms
                            )
                            await loop.run_in_executor(
                                log_executor, _save_audio_if_needed, raw_audio, resp,
                                conn_id, caller_number, caller_vid, server_ip,
                                customer_name
                            )
                        continue

                except json.JSONDecodeError:
                    continue

            # === BINARY AUDIO ===
            if msg.get("bytes") and not decision_sent:
                chunk = pcm16_to_float32(msg["bytes"])
                if len(chunk) == 0:
                    continue
                raw_audio = np.concatenate([raw_audio, chunk])
                bytes_received += len(msg["bytes"])
                chunks_received += 1

    except WebSocketDisconnect:
        print(f"[WS DISCONNECT] {conn_id} | Number={caller_number} | VID={caller_vid} | SrvIP={server_ip} | Cust={customer_name}")
    except Exception as e:
        print(f"[ERROR] {conn_id} | Number={caller_number} | VID={caller_vid} | SrvIP={server_ip} | Cust={customer_name} | {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if not decision_sent:
            try:
                elapsed_ms = int((time.time() - start_time) * 1000)
                resp = await process_final_audio(raw_audio, elapsed_ms,
                                                  caller_number, server_ip,
                                                  audio_start_time, customer_name)
                await ws.send_text(json.dumps(resp))
                print(f"[FINAL-FALLBACK] {resp['decision']} via {resp['method']}")
                decision_sent = True

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    log_executor, _log_to_elasticsearch, resp, conn_id,
                    caller_number, caller_vid, server_ip, customer_name,
                    len(raw_audio), bytes_received, elapsed_ms
                )
                await loop.run_in_executor(
                    log_executor, _save_audio_if_needed, raw_audio, resp,
                    conn_id, caller_number, caller_vid, server_ip,
                    customer_name
                )
            except Exception as e:
                print(f"[FALLBACK ERROR] {type(e).__name__}: {e}")

        dur = len(raw_audio) / ASTERISK_SR if len(raw_audio) > 0 else 0.0
        print(f"[WS END] {conn_id} | Number={caller_number} | VID={caller_vid} | SrvIP={server_ip} | Cust={customer_name} | Audio={dur:.2f}s | Bytes={bytes_received} | Sent={decision_sent}")
        print()



@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": WHISPER_MODEL_DIR,
        "version": "production_v9.4",
        "classifier_ready": classifier_ready,
        "silero_vad_ready": silero_ready,
        "elasticsearch_enabled": ELASTIC_ENABLED,
        "audio_storage_enabled": AUDIO_SAVE_ENABLED,
        "audio_format": AUDIO_FORMAT if AUDIO_SAVE_ENABLED else None,
        "audio_save_filter": AUDIO_SAVE_FILTER,
        "server_ip_tracking": True,
        "customer_name_tracking": True,
        "whisper_num_workers": WHISPER_NUM_WORKERS,
        "whisper_lock_removed": True,
        "preprocessing_async": True,
        "hallucination_prevention": {
            "min_speech_ratio": MIN_SPEECH_RATIO,
            "min_confidence_greeting": MIN_CONFIDENCE_GREETING,
            "min_speech_ms_human": MIN_SPEECH_MS_HUMAN,
            "min_p2a_ratio": MIN_P2A_RATIO,
            "noise_p2a_threshold": NOISE_P2A_THRESHOLD,
            "greeting_only_rule": f"P2A < {NOISE_P2A_THRESHOLD} → MACHINE else HUMAN",
            "default_decision": "NOTSURE",
        },
    }


@app.get("/status")
async def status():
    return {
        "whisper_model": WHISPER_MODEL_DIR,
        "device": DEVICE,
        "version": "production_v9.4",
        "classifier_ready": classifier_ready,
        "elasticsearch_enabled": ELASTIC_ENABLED,
        "audio_storage_enabled": AUDIO_SAVE_ENABLED,
        "audio_save_filter": AUDIO_SAVE_FILTER,
        "server_ip_tracking": True,
        "customer_name_tracking": True,
        "timing": {"MAX_CALL_SEC": MAX_CALL_SEC},
        "concurrency": {
            "whisper_num_workers": WHISPER_NUM_WORKERS,
            "whisper_transcription_lock": "REMOVED (v9.4)",
            "preprocessing": "async (off-loaded to worker thread)",
            "silero_vad": "async (off-loaded to worker thread)",
            "logging_executor_workers": 16,
        },
        "thresholds": {
            "SILENCE_RMS_THRESHOLD": SILENCE_RMS_THRESHOLD,
            "SILERO_VAD_THRESHOLD": SILERO_VAD_THRESHOLD,
            "SILERO_MIN_SPEECH_MS": SILERO_MIN_SPEECH_MS,
        },
        "hallucination_prevention": {
            "MIN_SPEECH_RATIO": MIN_SPEECH_RATIO,
            "MIN_CONFIDENCE_GREETING": MIN_CONFIDENCE_GREETING,
            "MIN_SPEECH_MS_HUMAN": MIN_SPEECH_MS_HUMAN,
            "MIN_P2A_RATIO": MIN_P2A_RATIO,
            "NOISE_P2A_THRESHOLD": NOISE_P2A_THRESHOLD,
            "SHORT_SPEECH_THRESHOLD": SHORT_SPEECH_THRESHOLD,
            "LOW_CONF_THRESHOLD": LOW_CONF_THRESHOLD,
            "greeting_only_rule": f"P2A < {NOISE_P2A_THRESHOLD} → MACHINE else HUMAN",
        },
        "keywords": {"human": len(HUMAN_KEYWORDS), "machine": len(MACHINE_KEYWORDS)},
        "features": {
            "architecture": "single-pass transcribe-only",
            "vad": "Silero VAD pre-filter (%s)" % ("ACTIVE" if silero_ready else "UNAVAILABLE"),
            "hallucination_prevention": "ACTIVE (P2A + speech_duration + confidence)",
            "default_decision": "NOTSURE",
            "classification_order": "MACHINE kw → HUMAN kw → ML → default NOTSURE",
            "server_ip_propagation": "Config handshake → response → ES → audio metadata",
            "customer_name_propagation": "Config handshake → response → ES → audio metadata",
            "elasticsearch": "ENABLED" if ELASTIC_ENABLED else "DISABLED",
            "audio_storage": "ENABLED" if AUDIO_SAVE_ENABLED else "DISABLED",
            "parallel_transcription": f"ENABLED (num_workers={WHISPER_NUM_WORKERS})",
        },
    }


@app.get("/elastic-status")
async def elastic_status():
    if not ELASTIC_ENABLED or elastic_logger is None:
        return {"enabled": False, "message": "Elasticsearch logging disabled"}
    try:
        stats = elastic_logger.get_stats()
        return {"enabled": True, "stats": stats}
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@app.get("/audio-status")
async def audio_status():
    if not AUDIO_SAVE_ENABLED or audio_logger is None:
        return {"enabled": False, "message": "Audio storage disabled"}
    try:
        stats = audio_logger.get_stats()
        return {
            "enabled": True,
            "format": AUDIO_FORMAT,
            "save_filter": AUDIO_SAVE_FILTER,
            "stats": stats,
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)