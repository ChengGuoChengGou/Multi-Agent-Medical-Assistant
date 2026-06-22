"""Edge TTS Service - Free Microsoft Edge TTS for Chinese medical chatbot.

Provides text-to-speech using Microsoft Edge's neural TTS voices.
No API key required. Supports 300+ voices across 70+ languages.

Usage:
    from edge_tts_service import edge_tts_generate, list_chinese_voices

    # Generate speech
    audio_bytes = await edge_tts_generate("你好，请问有什么症状？")

    # Generate with specific voice
    audio_bytes = await edge_tts_generate("你好", voice="zh-CN-XiaoxiaoNeural")
"""

import io
import logging

import edge_tts

logger = logging.getLogger(__name__)

# Recommended Chinese voices for medical chatbot
CHINESE_VOICES = {
    # Female voices (more natural for healthcare)
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # 晓晓 - warm, general
    "xiaoyi": "zh-CN-XiaoyiNeural",  # 晓伊 - gentle, medical
    "xiaomeng": "zh-CN-XiaomengNeural",  # 晓梦 - sweet
    "xiaomo": "zh-CN-XiaomoNeural",  # 晓墨 - gentle
    "xiaorui": "zh-CN-XiaoruiNeural",  # 晓睿 - warm
    # Male voices
    "yunxi": "zh-CN-YunxiNeural",  # 云希 - bright
    "yunjian": "zh-CN-YunjianNeural",  # 云健 - sports
    "yunxia": "zh-CN-YunxiaNeural",  # 云夏 - narration
    "yunyang": "zh-CN-YunyangNeural",  # 云扬 - news
}

# Default voice for medical chatbot (warm, clear female voice)
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

# Medical-context rate/pitch adjustments
MEDICAL_RATE = "+0%"  # Normal speed for clarity
MEDICAL_PITCH = "+0Hz"  # Normal pitch


async def edge_tts_generate(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = MEDICAL_RATE,
    pitch: str = MEDICAL_PITCH,
) -> bytes:
    """Generate speech audio from text using Edge TTS.

    Args:
        text: Text to convert to speech (Chinese or English)
        voice: Edge TTS voice name (e.g., 'zh-CN-XiaoxiaoNeural')
        rate: Speech rate adjustment (e.g., '+10%', '-5%')
        pitch: Pitch adjustment (e.g., '+2Hz', '-1Hz')

    Returns:
        MP3 audio bytes

    Raises:
        ValueError: If text is empty
        RuntimeError: If TTS generation fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Edge TTS has a 1000-char limit per request; chunk if needed
    chunks = _split_text(text, max_chars=900)
    all_audio = bytearray()

    for chunk in chunks:
        try:
            communicate = edge_tts.Communicate(
                text=chunk,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            buffer = io.BytesIO()
            async for chunk_data in communicate.stream():
                if chunk_data["type"] == "audio":
                    buffer.write(chunk_data["data"])

            audio_bytes = buffer.getvalue()
            if audio_bytes:
                all_audio.extend(audio_bytes)
            else:
                logger.warning(f"[EdgeTTS] Empty audio for chunk: {chunk[:50]}...")
        except Exception as e:
            logger.error(f"[EdgeTTS] Generation error: {e}")
            raise RuntimeError(f"Edge TTS generation failed: {e}") from e

    if not all_audio:
        raise RuntimeError("Edge TTS produced no audio output")

    return bytes(all_audio)


async def list_chinese_voices() -> list[dict]:
    """List all available Chinese voices from Edge TTS.

    Returns:
        List of voice dicts with name, gender, locale
    """
    try:
        voices = await edge_tts.list_voices()
        chinese = [
            {
                "name": v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
            }
            for v in voices
            if v["Locale"].startswith("zh-")
        ]
        return sorted(chinese, key=lambda x: x["name"])
    except Exception as e:
        logger.error(f"[EdgeTTS] Failed to list voices: {e}")
        # Return curated fallback
        return [
            {"name": name, "gender": "Female" if i < 5 else "Male", "locale": "zh-CN"}
            for i, name in enumerate(CHINESE_VOICES.values())
        ]


def _split_text(text: str, max_chars: int = 900) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Handles Chinese/English mixed text. Preserves sentence integrity.
    """
    if len(text) <= max_chars:
        return [text]

    # Sentence delimiters (Chinese + English)
    delimiters = {"。", "！", "？", "；", "\n", ".", "!", "?", ";"}
    chunks = []
    current = ""

    for char in text:
        current += char
        if char in delimiters and len(current) >= max_chars * 0.5:
            chunks.append(current.strip())
            current = ""

    if current.strip():
        if chunks and len(current.strip()) < 50:
            # Merge tiny trailing chunk
            chunks[-1] += current
        else:
            chunks.append(current.strip())

    return chunks if chunks else [text[:max_chars]]


def get_voice_id(voice_name: str) -> str:
    """Resolve voice name/alias to Edge TTS voice ID.

    Supports short aliases like 'xiaoxiao' → 'zh-CN-XiaoxiaoNeural'
    or full names passed through directly.
    """
    # Check alias map first
    if voice_name.lower() in CHINESE_VOICES:
        return CHINESE_VOICES[voice_name.lower()]
    # If it looks like a full voice name, use as-is
    if "-" in voice_name and voice_name.endswith("Neural"):
        return voice_name
    # Default
    return DEFAULT_VOICE
