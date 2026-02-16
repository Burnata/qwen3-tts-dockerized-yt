# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from .. import Qwen3TTSModel


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _decode_audio_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}") from exc
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    return wav, int(sr)


def _decode_audio_b64(audio_b64: str) -> Tuple[np.ndarray, int]:
    try:
        raw = base64.b64decode(audio_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc
    return _decode_audio_bytes(raw)


def _encode_audio_b64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, int(sr), format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_audio_wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, int(sr), format="WAV")
    return buf.getvalue()


def _encode_audio_mp3_bytes(wav: np.ndarray, sr: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="qwen_tts_mp3_") as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "audio.wav"
        mp3_path = tmp_dir / "audio.mp3"
        sf.write(str(wav_path), wav, int(sr), format="WAV")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-f", "mp3", str(mp3_path)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="ffmpeg is not installed") from exc
        except subprocess.CalledProcessError as exc:
            err = exc.stderr.decode("utf-8", errors="ignore")
            raise HTTPException(status_code=500, detail=f"ffmpeg failed to encode mp3: {err}") from exc
        return mp3_path.read_bytes()


class TTSRequest(BaseModel):
    mode: Literal["custom_voice", "voice_design", "voice_clone"] = Field(
        ..., description="Which generation mode to use."
    )
    text: str = Field(..., description="Text to synthesize.")
    language: Optional[str] = Field(default="Auto", description="Language (Auto or explicit language).")
    speaker: Optional[str] = Field(default=None, description="Speaker name for custom_voice.")
    instruct: Optional[str] = Field(default=None, description="Instruction for custom_voice or voice_design.")
    ref_audio_b64: Optional[str] = Field(default=None, description="Base64 WAV/MP3/FLAC for voice_clone.")
    ref_text: Optional[str] = Field(default=None, description="Reference text for voice_clone (required unless x_vector_only_mode).")
    x_vector_only_mode: bool = Field(default=False, description="Voice clone using x-vector only.")
    model: Optional[str] = Field(default=None, description="Override model id/path.")

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None


class TTSResponse(BaseModel):
    mode: str
    model: str
    sample_rate: int
    audio_b64: str


class OpenAISpeechRequest(BaseModel):
    model: str = Field(..., description="Model id or override for the local TTS model.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice name.")
    response_format: Optional[str] = Field(default="mp3", description="Audio response format.")
    speed: Optional[float] = Field(default=None, description="Speech speed multiplier.")
    language: Optional[str] = Field(default="Auto", description="Language (Auto or explicit language).")


class YouTubeCloneRequest(BaseModel):
    url: str = Field(..., description="YouTube URL")
    text: str = Field(..., description="Target text to synthesize")
    language: Optional[str] = Field(default="Auto", description="Language (Auto or explicit language).")
    sub_lang: Optional[str] = Field(default="en", description="Subtitle language to prefer.")
    x_vector_only_mode: bool = Field(default=False, description="Voice clone using x-vector only.")
    model: Optional[str] = Field(default=None, description="Override model id/path.")

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None


class _ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: Dict[str, Qwen3TTSModel] = {}

        self.device = os.getenv("QWEN_TTS_DEVICE", "cpu")
        self.dtype = _dtype_from_str(os.getenv("QWEN_TTS_DTYPE", "float32"))
        self.attn_impl = os.getenv("QWEN_TTS_ATTN", "eager")

        self.default_custom = os.getenv(
            "QWEN_TTS_CUSTOM_VOICE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        self.default_design = os.getenv(
            "QWEN_TTS_VOICE_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        )
        self.default_clone = os.getenv(
            "QWEN_TTS_VOICE_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )

    def _resolve_model_id(self, mode: str, override: Optional[str]) -> str:
        if override:
            return override
        if mode == "custom_voice":
            return self.default_custom
        if mode == "voice_design":
            return self.default_design
        if mode == "voice_clone":
            return self.default_clone
        raise ValueError(f"Unknown mode: {mode}")

    def get(self, mode: str, override: Optional[str]) -> Tuple[str, Qwen3TTSModel]:
        model_id = self._resolve_model_id(mode, override)
        with self._lock:
            model = self._models.get(model_id)
            if model is None:
                model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation=self.attn_impl,
                )
                self._models[model_id] = model
        return model_id, model


def _collect_gen_kwargs(req: TTSRequest) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "subtalker_top_k": req.subtalker_top_k,
        "subtalker_top_p": req.subtalker_top_p,
        "subtalker_temperature": req.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _collect_gen_kwargs_from_values(
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: Optional[float],
    subtalker_top_k: Optional[int],
    subtalker_top_p: Optional[float],
    subtalker_temperature: Optional[float],
) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "subtalker_top_k": subtalker_top_k,
        "subtalker_top_p": subtalker_top_p,
        "subtalker_temperature": subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _extract_subtitle_text(path: Path) -> str:
    lines = []
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read subtitle file: {exc}") from exc
    for line in raw:
        s = line.strip()
        if not s:
            continue
        if s.startswith("WEBVTT"):
            continue
        if "-->" in s:
            continue
        if s.isdigit():
            continue
        s = re.sub(r"<[^>]+>", "", s).strip()
        if s:
            lines.append(s)
    return " ".join(lines).strip()


def _download_youtube_ref(url: str, sub_lang: Optional[str]) -> Tuple[Tuple[np.ndarray, int], str]:
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"yt-dlp is not installed: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="qwen_tts_ytdlp_") as tmp:
        tmp_dir = Path(tmp)
        outtmpl = str(tmp_dir / "%(id)s.%(ext)s")
        ydl_opts: Dict[str, Any] = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
            ],
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
        }
        if sub_lang:
            ydl_opts["subtitleslangs"] = [sub_lang]

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to download YouTube assets: {exc}") from exc

        wav_files = sorted(tmp_dir.glob("*.wav"))
        if not wav_files:
            raise HTTPException(status_code=400, detail="No WAV audio extracted from YouTube URL.")
        wav_path = wav_files[0]
        ref_audio = _decode_audio_bytes(wav_path.read_bytes())

        vtt_files = sorted(tmp_dir.glob("*.vtt"))
        if not vtt_files:
            srt_files = sorted(tmp_dir.glob("*.srt"))
            vtt_files = srt_files
        if not vtt_files:
            raise HTTPException(status_code=400, detail="No subtitles found for YouTube URL.")

        preferred = [p for p in vtt_files if ".auto." not in p.name]
        sub_path = preferred[0] if preferred else vtt_files[0]
        ref_text = _extract_subtitle_text(sub_path)
        if not ref_text:
            raise HTTPException(status_code=400, detail="Subtitle file is empty after parsing.")

        return ref_audio, ref_text


def _normalize_openai_response_format(response_format: Optional[str]) -> str:
    fmt = (response_format or "mp3").strip().lower()
    if fmt in ("mp3", "mpeg"):
        return "mp3"
    raise HTTPException(status_code=400, detail="Only mp3 response_format is supported")


def _load_openai_voice_map() -> Dict[str, Dict[str, Any]]:
    raw = os.getenv("QWEN_TTS_OPENAI_VOICE_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid QWEN_TTS_OPENAI_VOICE_MAP: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="QWEN_TTS_OPENAI_VOICE_MAP must be a JSON object")
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise HTTPException(status_code=500, detail="QWEN_TTS_OPENAI_VOICE_MAP entries must be objects")
    return data


_OPENAI_VOICE_MAP = _load_openai_voice_map()


def _resolve_openai_voice(voice: str) -> Dict[str, Any]:
    cfg = _OPENAI_VOICE_MAP.get(voice)
    if cfg is None:
        return {"mode": "custom_voice", "speaker": voice}
    mode = cfg.get("mode") or "custom_voice"
    resolved = dict(cfg)
    resolved["mode"] = mode
    return resolved


app = FastAPI(title="Qwen3-TTS API", version="1.0")
_manager = _ModelManager()


@app.on_event("startup")
def _preload_models() -> None:
    preload = os.getenv("QWEN_TTS_PRELOAD", "0").strip().lower() in ("1", "true", "yes")
    if not preload:
        return
    for mode in ("custom_voice", "voice_design", "voice_clone"):
        _manager.get(mode, None)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> Dict[str, str]:
    return {
        "custom_voice": _manager.default_custom,
        "voice_design": _manager.default_design,
        "voice_clone": _manager.default_clone,
    }


@app.post("/generate")
def generate(req: TTSRequest) -> Response:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    model_id, tts = _manager.get(req.mode, req.model)
    language = req.language or "Auto"
    kwargs = _collect_gen_kwargs(req)

    if req.mode == "custom_voice":
        if not req.speaker:
            raise HTTPException(status_code=400, detail="speaker is required for custom_voice")
        wavs, sr = tts.generate_custom_voice(
            text=req.text.strip(),
            language=language,
            speaker=req.speaker,
            instruct=(req.instruct or "").strip() or None,
            **kwargs,
        )
    elif req.mode == "voice_design":
        if not req.instruct or not req.instruct.strip():
            raise HTTPException(status_code=400, detail="instruct is required for voice_design")
        wavs, sr = tts.generate_voice_design(
            text=req.text.strip(),
            language=language,
            instruct=req.instruct.strip(),
            **kwargs,
        )
    else:
        if not req.ref_audio_b64:
            raise HTTPException(status_code=400, detail="ref_audio_b64 is required for voice_clone")
        if (not req.x_vector_only_mode) and (not req.ref_text or not req.ref_text.strip()):
            raise HTTPException(
                status_code=400,
                detail="ref_text is required for voice_clone unless x_vector_only_mode is true",
            )
        ref_audio = _decode_audio_b64(req.ref_audio_b64)
        wavs, sr = tts.generate_voice_clone(
            text=req.text.strip(),
            language=language,
            ref_audio=ref_audio,
            ref_text=(req.ref_text.strip() if req.ref_text else None),
            x_vector_only_mode=bool(req.x_vector_only_mode),
            **kwargs,
        )

    payload = _encode_audio_wav_bytes(wavs[0], sr)
    headers = {
        "X-Mode": req.mode,
        "X-Model-Id": model_id,
        "X-Sample-Rate": str(int(sr)),
    }
    return Response(content=payload, media_type="audio/wav", headers=headers)


@app.post("/generate/upload")
async def generate_upload(
    mode: str = Form(...),
    text: str = Form(...),
    language: str = Form("Auto"),
    speaker: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    x_vector_only_mode: bool = Form(False),
    model: Optional[str] = Form(None),
    max_new_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    top_p: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    subtalker_top_k: Optional[int] = Form(None),
    subtalker_top_p: Optional[float] = Form(None),
    subtalker_temperature: Optional[float] = Form(None),
) -> Response:
    mode = (mode or "").strip()
    if mode not in ("custom_voice", "voice_design", "voice_clone"):
        raise HTTPException(status_code=400, detail="mode must be one of custom_voice, voice_design, voice_clone")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    model_id, tts = _manager.get(mode, model)
    language = language or "Auto"
    kwargs = _collect_gen_kwargs_from_values(
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        subtalker_top_k,
        subtalker_top_p,
        subtalker_temperature,
    )

    if mode == "custom_voice":
        if not speaker:
            raise HTTPException(status_code=400, detail="speaker is required for custom_voice")
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker,
            instruct=(instruct or "").strip() or None,
            **kwargs,
        )
    elif mode == "voice_design":
        if not instruct or not instruct.strip():
            raise HTTPException(status_code=400, detail="instruct is required for voice_design")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=instruct.strip(),
            **kwargs,
        )
    else:
        if ref_audio is None:
            raise HTTPException(status_code=400, detail="ref_audio file is required for voice_clone")
        if (not x_vector_only_mode) and (not ref_text or not ref_text.strip()):
            raise HTTPException(
                status_code=400,
                detail="ref_text is required for voice_clone unless x_vector_only_mode is true",
            )
        raw = await ref_audio.read()
        if not raw:
            raise HTTPException(status_code=400, detail="ref_audio file is empty")
        ref_audio_tuple = _decode_audio_bytes(raw)
        wavs, sr = tts.generate_voice_clone(
            text=text.strip(),
            language=language,
            ref_audio=ref_audio_tuple,
            ref_text=(ref_text.strip() if ref_text else None),
            x_vector_only_mode=bool(x_vector_only_mode),
            **kwargs,
        )

    payload = _encode_audio_wav_bytes(wavs[0], sr)
    headers = {
        "X-Mode": mode,
        "X-Model-Id": model_id,
        "X-Sample-Rate": str(int(sr)),
    }
    return Response(content=payload, media_type="audio/wav", headers=headers)


@app.post("/generate/youtube")
def generate_youtube(req: YouTubeCloneRequest) -> Response:
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="url is required")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    ref_audio, ref_text = _download_youtube_ref(req.url.strip(), req.sub_lang)
    if (not req.x_vector_only_mode) and (not ref_text):
        raise HTTPException(status_code=400, detail="Transcript is required for voice_clone unless x_vector_only_mode is true")

    model_id, tts = _manager.get("voice_clone", req.model)
    language = req.language or "Auto"
    kwargs = _collect_gen_kwargs_from_values(
        req.max_new_tokens,
        req.temperature,
        req.top_k,
        req.top_p,
        req.repetition_penalty,
        req.subtalker_top_k,
        req.subtalker_top_p,
        req.subtalker_temperature,
    )
    wavs, sr = tts.generate_voice_clone(
        text=req.text.strip(),
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=bool(req.x_vector_only_mode),
        **kwargs,
    )

    payload = _encode_audio_wav_bytes(wavs[0], sr)
    headers = {
        "X-Mode": "voice_clone",
        "X-Model-Id": model_id,
        "X-Sample-Rate": str(int(sr)),
    }
    return Response(content=payload, media_type="audio/wav", headers=headers)


@app.get("/audio/models")
def audio_models() -> Dict[str, str]:
    return {
        "custom_voice": _manager.default_custom,
        "voice_design": _manager.default_design,
        "voice_clone": _manager.default_clone,
    }


@app.get("/audio/voices")
def audio_voices() -> Dict[str, Any]:
    voices = list(_OPENAI_VOICE_MAP.keys()) if _OPENAI_VOICE_MAP else []
    return {
        "voices": voices,
        "default_speakers": ["Male-en", "Female-en", "Male-zh", "Female-zh"],
    }


@app.post("/audio/speech")
def audio_speech(req: OpenAISpeechRequest) -> Response:
    return openai_speech(req)


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest) -> Response:
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="input is required")
    if not req.voice or not req.voice.strip():
        raise HTTPException(status_code=400, detail="voice is required")

    fmt = _normalize_openai_response_format(req.response_format)
    voice_cfg = _resolve_openai_voice(req.voice.strip())
    mode = voice_cfg.get("mode", "custom_voice")
    model_override = voice_cfg.get("model") or req.model
    language = req.language or "Auto"

    if mode == "custom_voice":
        speaker = voice_cfg.get("speaker") or req.voice.strip()
        instruct = (voice_cfg.get("instruct") or "").strip() or None
        model_id, tts = _manager.get("custom_voice", model_override)
        wavs, sr = tts.generate_custom_voice(
            text=req.input.strip(),
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
    elif mode == "voice_design":
        instruct = (voice_cfg.get("instruct") or req.voice.strip()).strip()
        if not instruct:
            raise HTTPException(status_code=400, detail="instruct is required for voice_design")
        model_id, tts = _manager.get("voice_design", model_override)
        wavs, sr = tts.generate_voice_design(
            text=req.input.strip(),
            language=language,
            instruct=instruct,
        )
    elif mode == "youtube_clone":
        youtube_url = voice_cfg.get("youtube_url")
        if not youtube_url:
            raise HTTPException(status_code=400, detail="youtube_url is required for youtube_clone voice")
        sub_lang = voice_cfg.get("sub_lang")
        ref_audio, ref_text = _download_youtube_ref(str(youtube_url), sub_lang)
        model_id, tts = _manager.get("voice_clone", model_override)
        wavs, sr = tts.generate_voice_clone(
            text=req.input.strip(),
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=bool(voice_cfg.get("x_vector_only_mode", False)),
        )
    elif mode == "voice_clone":
        ref_audio_b64 = voice_cfg.get("ref_audio_b64")
        ref_text = voice_cfg.get("ref_text")
        if not ref_audio_b64:
            raise HTTPException(status_code=400, detail="ref_audio_b64 is required for voice_clone voice")
        ref_audio = _decode_audio_b64(str(ref_audio_b64))
        model_id, tts = _manager.get("voice_clone", model_override)
        wavs, sr = tts.generate_voice_clone(
            text=req.input.strip(),
            language=language,
            ref_audio=ref_audio,
            ref_text=(str(ref_text).strip() if ref_text else None),
            x_vector_only_mode=bool(voice_cfg.get("x_vector_only_mode", False)),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported voice mode: {mode}")

    if fmt == "mp3":
        payload = _encode_audio_mp3_bytes(wavs[0], sr)
        media_type = "audio/mpeg"
    else:
        payload = _encode_audio_wav_bytes(wavs[0], sr)
        media_type = "audio/wav"

    headers = {
        "X-Mode": mode,
        "X-Model-Id": model_id,
        "X-Sample-Rate": str(int(sr)),
    }
    return Response(content=payload, media_type=media_type, headers=headers)


def main() -> None:
    import uvicorn

    host = os.getenv("QWEN_TTS_API_HOST", "0.0.0.0")
    port = int(os.getenv("QWEN_TTS_API_PORT", "8000"))
    uvicorn.run("qwen_tts.api.server:app", host=host, port=port)


if __name__ == "__main__":
    main()
