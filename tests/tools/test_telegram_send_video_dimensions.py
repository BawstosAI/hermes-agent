"""Outgoing Telegram videos declare their displayed size.

sendVideo treats ``width``/``height``/``duration`` as optional, but without
them Telegram stores 0x0 and clients draw a square bubble with the frame
stretched into it (portrait 9:16 phone videos are the usual casualty). Both
send paths -- ``hermes send`` (tools.send_message_tool) and the live adapter
-- must probe the file and pass the attributes. ``telegram`` is stubbed for
the standalone path; ffprobe is stubbed at the subprocess boundary.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.telegram import adapter as tg_adapter


# ---------------------------------------------------------------------------
# ffprobe stub
# ---------------------------------------------------------------------------

def _ffprobe_json(width, height, rotation=None, duration="60.501000") -> str:
    stream = {"width": width, "height": height}
    if rotation is not None:
        stream["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": rotation}]
    return json.dumps({"streams": [stream], "format": {"duration": duration}})


def _stub_ffprobe(monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0) -> list:
    monkeypatch.setattr(tg_adapter, "_find_ffprobe", lambda: "/usr/bin/ffprobe")
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    return calls


def test_probe_reads_displayed_size_and_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_ffprobe(monkeypatch, _ffprobe_json(1080, 1920))
    assert tg_adapter._probe_video_attrs("/tmp/x.mp4") == {"width": 1080, "height": 1920, "duration": 61}
    assert calls and calls[0][-1] == "/tmp/x.mp4"


def test_probe_swaps_rotated_phone_footage(monkeypatch: pytest.MonkeyPatch) -> None:
    # A phone records 1920x1080 and tags the stream -90: it is *shown* 1080x1920.
    _stub_ffprobe(monkeypatch, _ffprobe_json(1920, 1080, rotation=-90))
    attrs = tg_adapter._probe_video_attrs("/tmp/x.mov")
    assert (attrs["width"], attrs["height"]) == (1080, 1920)


def test_probe_keeps_size_when_rotated_180(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ffprobe(monkeypatch, _ffprobe_json(1920, 1080, rotation=180))
    attrs = tg_adapter._probe_video_attrs("/tmp/x.mov")
    assert (attrs["width"], attrs["height"]) == (1920, 1080)


def test_probe_without_ffprobe_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tg_adapter, "_find_ffprobe", lambda: None)
    assert tg_adapter._probe_video_attrs("/tmp/x.mp4") == {}


def test_probe_failure_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ffprobe(monkeypatch, "", returncode=1)
    assert tg_adapter._probe_video_attrs("/tmp/x.mp4") == {}


# ---------------------------------------------------------------------------
# Send path 1: `hermes send --to telegram "MEDIA:/x.mp4"` (standalone Bot)
# ---------------------------------------------------------------------------

_ATTRS = {"width": 1080, "height": 1920, "duration": 61}


def _install_telegram_mock(monkeypatch: pytest.MonkeyPatch, bot_factory: MagicMock) -> None:
    parse_mode = SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML")
    constants_mod = SimpleNamespace(ParseMode=parse_mode)
    _MessageEntity = lambda **_kw: SimpleNamespace(**_kw)
    telegram_mod = SimpleNamespace(
        Bot=bot_factory,
        MessageEntity=_MessageEntity,
        constants=constants_mod,
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram_mod)
    monkeypatch.setitem(sys.modules, "telegram.constants", constants_mod)


def _make_bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    bot.send_video = AsyncMock(return_value=SimpleNamespace(message_id=3))
    bot.send_document = AsyncMock(return_value=SimpleNamespace(message_id=4))
    return bot


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "TELEGRAM_PROXY", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY",
        "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: None, raising=False)
    monkeypatch.setattr(
        "gateway.platforms.base._detect_macos_system_proxy", lambda: None
    )


def _tmpfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"x")
    f.close()
    return f.name


def test_hermes_send_declares_video_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_senders import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    monkeypatch.setattr(tg_adapter, "_probe_video_attrs", lambda path: dict(_ATTRS))
    video = _tmpfile(".mp4")
    try:
        res = asyncio.run(
            _send_telegram("tok", "123", "60 s, hook on the market", media_files=[(video, False)])
        )
        assert res["success"] is True
        kwargs = bot.send_video.await_args.kwargs
        assert (kwargs["width"], kwargs["height"], kwargs["duration"]) == (1080, 1920, 61)
        assert kwargs["supports_streaming"] is True
        bot.send_document.assert_not_awaited()
    finally:
        os.unlink(video)


def test_hermes_send_still_delivers_when_probe_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.send_message_senders import _send_telegram

    _no_proxy(monkeypatch)
    bot = _make_bot()
    _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
    monkeypatch.setattr(tg_adapter, "_probe_video_attrs", lambda path: {})
    video = _tmpfile(".mp4")
    try:
        res = asyncio.run(_send_telegram("tok", "123", "", media_files=[(video, False)]))
        assert res["success"] is True
        kwargs = bot.send_video.await_args.kwargs
        assert "width" not in kwargs and "height" not in kwargs
    finally:
        os.unlink(video)


@pytest.mark.parametrize("send_path,retry_error", [
    ("standalone", None), ("adapter", None),
    ("standalone", "Message thread not found"), ("adapter", "Message thread not found"),
    ("standalone", "Can't parse caption"),
])
def test_real_portrait_probe_reaches_both_send_paths(tmp_path, monkeypatch, send_path, retry_error):
    """Real MP4 + ffprobe + send path; only the Telegram network is replaced."""
    import shutil
    import subprocess

    from gateway.config import PlatformConfig
    from tools.send_message_senders import _send_telegram

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not tg_adapter._find_ffprobe():
        pytest.skip("ffmpeg and ffprobe required for real media integration")
    video = tmp_path / "portrait.mp4"
    subprocess.run(
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=72x128:r=10",
         "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True, capture_output=True, timeout=30,
    )
    _no_proxy(monkeypatch)
    bot = _make_bot()
    if retry_error:
        from telegram.error import BadRequest
        bot.send_video.side_effect = [BadRequest(retry_error), SimpleNamespace(message_id=3)]
    if send_path == "standalone":
        _install_telegram_mock(monkeypatch, MagicMock(return_value=bot))
        result = asyncio.run(_send_telegram(
            "tok", "123", "Portrait", media_files=[(str(video), False)], thread_id="42"))
        assert result["success"] is True
    else:
        adapter = tg_adapter.TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
        adapter._bot = bot
        result = asyncio.run(adapter.send_video(
            "123", str(video), caption="Portrait", metadata={"thread_id": "42", "direct_messages_topic_id": "42",
                                                    "telegram_dm_topic_reply_fallback": True}))
        assert result.success is True
    for call in bot.send_video.await_args_list:
        kwargs = call.kwargs
        assert (kwargs["width"], kwargs["height"], kwargs["duration"]) == (72, 128, 1)
        assert kwargs["supports_streaming"] is True
    routing_key = "message_thread_id"
    assert bot.send_video.await_args_list[0].kwargs[routing_key] == 42
    if retry_error == "Message thread not found":
        assert bot.send_video.await_count == 2
        assert routing_key not in bot.send_video.await_args.kwargs
    bot.send_document.assert_not_awaited()


# ---------------------------------------------------------------------------
# Send path 2: the live adapter's send_video (agent replying in a chat)
# ---------------------------------------------------------------------------

def test_adapter_send_video_declares_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.config import PlatformConfig

    adapter = tg_adapter.TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._bot.send_video = AsyncMock(return_value=SimpleNamespace(message_id=7))
    monkeypatch.setattr(tg_adapter, "_probe_video_attrs", lambda path: dict(_ATTRS))
    video = _tmpfile(".mp4")
    try:
        res = asyncio.run(adapter.send_video("123", video, caption="60 s"))
        assert res.success is True
        kwargs = adapter._bot.send_video.await_args.kwargs
        assert (kwargs["width"], kwargs["height"], kwargs["duration"]) == (1080, 1920, 61)
        assert kwargs["supports_streaming"] is True
    finally:
        os.unlink(video)
