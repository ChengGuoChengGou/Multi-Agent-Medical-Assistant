"""
Unit tests for edge_tts_service.py — TTS text splitting, voice resolution, constants.
Run: python -m pytest tests/test_edge_tts_service.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_tts_service import (
    CHINESE_VOICES,
    DEFAULT_VOICE,
    MEDICAL_PITCH,
    MEDICAL_RATE,
    _split_text,
    edge_tts_generate,
    get_voice_id,
    list_chinese_voices,
)


# ── Constants ──
class TestConstants:
    def test_chinese_voices_is_dict(self):
        assert isinstance(CHINESE_VOICES, dict)
        assert len(CHINESE_VOICES) >= 5

    def test_chinese_voices_have_xiaoxiao(self):
        assert "xiaoxiao" in CHINESE_VOICES
        assert CHINESE_VOICES["xiaoxiao"] == "zh-CN-XiaoxiaoNeural"

    def test_default_voice(self):
        assert DEFAULT_VOICE == "zh-CN-XiaoxiaoNeural"

    def test_medical_rate_and_pitch(self):
        assert isinstance(MEDICAL_RATE, str)
        assert isinstance(MEDICAL_PITCH, str)


# ── get_voice_id ──
class TestGetVoiceId:
    def test_alias_xiaoxiao(self):
        assert get_voice_id("xiaoxiao") == "zh-CN-XiaoxiaoNeural"

    def test_alias_case_insensitive(self):
        assert get_voice_id("Xiaoxiao") == "zh-CN-XiaoxiaoNeural"
        assert get_voice_id("YUNXI") == "zh-CN-YunxiNeural"

    def test_full_name_pass_through(self):
        full = "zh-CN-YunyangNeural"
        assert get_voice_id(full) == full

    def test_unknown_name_returns_default(self):
        assert get_voice_id("unknown_voice") == DEFAULT_VOICE

    def test_all_aliases_resolve(self):
        for alias, expected in CHINESE_VOICES.items():
            assert get_voice_id(alias) == expected


# ── _split_text ──
class TestSplitText:
    def test_short_text_unchanged(self):
        text = "你好"
        assert _split_text(text) == [text]

    def test_empty_list_returns_original(self):
        # Edge case: exactly at limit
        text = "a" * 900
        result = _split_text(text, max_chars=900)
        assert result == [text]

    def test_long_text_splits(self):
        # Create text with sentence boundaries
        text = "这是一个测试句子。" * 200  # ~1800 chars
        result = _split_text(text, max_chars=900)
        assert len(result) >= 2
        # All text preserved
        assert "".join(result) == text

    def test_splits_at_sentence_boundary(self):
        text = "A" * 500 + "。" + "B" * 500 + "。"
        result = _split_text(text, max_chars=900)
        assert len(result) >= 2

    def test_chinese_delimiters(self):
        text = "你好！" * 100 + "谢谢。" * 100
        result = _split_text(text, max_chars=900)
        for chunk in result:
            assert len(chunk) <= 900 or len(result) == 1

    def test_english_delimiters(self):
        text = "Hello! " * 200
        result = _split_text(text, max_chars=900)
        assert len(result) >= 2

    def test_newline_delimiter(self):
        text = "Line one.\n" * 150
        result = _split_text(text, max_chars=900)
        assert len(result) >= 2

    def test_tiny_trailing_merged(self):
        # If trailing chunk is < 50 chars, it merges with previous
        text = "A" * 800 + "。" + "end"
        result = _split_text(text, max_chars=900)
        # "end" is < 50 chars, should be merged
        assert len(result) == 1


# ── edge_tts_generate (async, needs mock) ──
class TestEdgeTtsGenerate:
    @pytest.mark.asyncio
    async def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await edge_tts_generate("")

    @pytest.mark.asyncio
    async def test_whitespace_text_raises(self):
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await edge_tts_generate("   ")

    @pytest.mark.asyncio
    async def test_generate_calls_edge_tts(self, monkeypatch):
        """Mock edge_tts.Communicate to avoid network calls."""

        class FakeCommunicate:
            def __init__(self, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"\x00\x01\x02"}

        monkeypatch.setattr("edge_tts_service.edge_tts.Communicate", FakeCommunicate)
        result = await edge_tts_generate("测试")
        assert result == b"\x00\x01\x02"

    @pytest.mark.asyncio
    async def test_generate_empty_audio_raises(self, monkeypatch):
        class FakeCommunicate:
            def __init__(self, **kwargs):
                pass

            async def stream(self):
                yield {"type": "metadata", "data": b""}

        monkeypatch.setattr("edge_tts_service.edge_tts.Communicate", FakeCommunicate)
        with pytest.raises(RuntimeError, match="no audio output"):
            await edge_tts_generate("测试")


# ── list_chinese_voices (async) ──
class TestListChineseVoices:
    @pytest.mark.asyncio
    async def test_lists_voices(self, monkeypatch):
        fake_voices = [
            {"ShortName": "zh-CN-XiaoxiaoNeural", "Gender": "Female", "Locale": "zh-CN"},
            {"ShortName": "en-US-JennyNeural", "Gender": "Female", "Locale": "en-US"},
        ]

        async def fake_list():
            return fake_voices

        monkeypatch.setattr("edge_tts_service.edge_tts.list_voices", fake_list)
        result = await list_chinese_voices()
        assert len(result) == 1
        assert result[0]["name"] == "zh-CN-XiaoxiaoNeural"

    @pytest.mark.asyncio
    async def test_fallback_on_error(self, monkeypatch):
        async def fake_list():
            raise RuntimeError("network error")

        monkeypatch.setattr("edge_tts_service.edge_tts.list_voices", fake_list)
        result = await list_chinese_voices()
        # Should return curated fallback
        assert len(result) > 0
        assert all("name" in v for v in result)
