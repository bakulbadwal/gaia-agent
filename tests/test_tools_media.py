"""Tests for gaia_agent.tools_media.

Offline tests generate their own files in a tmp dir. Network tests are marked
@pytest.mark.network. Vision tests skip unless a model endpoint is configured.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gaia_agent import tools_media as tm  # noqa: E402
from gaia_agent.tools_media import (  # noqa: E402
    TOOLS_MEDIA,
    image_question,
    read_file,
    run_python_file,
    transcribe_audio,
    youtube_frames_question,
    youtube_transcript,
)

# A short public talk with English captions, unrelated to the course questions.
CAPTIONED_VIDEO = os.environ.get("GAIA_TEST_YT_URL", "https://www.youtube.com/watch?v=jNQXAC9IVRw")


def _vision_available() -> bool:
    """True only if gaia_agent.config is importable and its provider looks usable."""
    try:
        from gaia_agent import config
    except Exception:
        return False
    try:
        if hasattr(config, "load_env"):
            config.load_env()
        info = config.provider_info() if hasattr(config, "provider_info") else {}
    except Exception:
        return False
    provider = str(info.get("provider") or os.environ.get("GAIA_PROVIDER", "gemini")).lower()
    for k in ("has_key", "key_present", "api_key_set", "key_set"):
        if k in info:
            return bool(info[k])
    if provider == "lmstudio":
        try:
            import requests

            requests.get("http://localhost:1234/v1/models", timeout=2).raise_for_status()
            return True
        except Exception:
            return False
    key_env = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}.get(provider)
    return bool(key_env and os.environ.get(key_env))


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point caches (transcripts, youtube) at a tmp dir unless a test opts out."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(tm, "_data_dir", lambda: data)
    yield data


def test_exports():
    names = [t.name for t in TOOLS_MEDIA]
    assert names == [
        "read_file",
        "run_python_file",
        "transcribe_audio",
        "image_question",
        "youtube_transcript",
        "youtube_audio_transcript",
        "youtube_frames_question",
    ]
    for t in TOOLS_MEDIA:
        assert t.output_type == "string"
        assert t.description


def test_read_file_xlsx(tmp_path):
    import openpyxl
    from openpyxl.styles import PatternFill

    path = tmp_path / "sales.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Item", "Type", "Price"])
    rows = [("Burger", "Food", 5.5), ("Soda", "Drink", 1.25), ("Fries", "Food", 2.0), ("Tea", "Drink", 1.0)]
    for r in rows:
        ws.append(r)
    ws["A2"].fill = PatternFill(start_color="FF00FF00", end_color="FF00FF00", fill_type="solid")
    ws2 = wb.create_sheet("Other")
    ws2.append(["x", "y"])
    ws2.append([1, 2])
    wb.save(path)

    out = read_file(str(path))
    assert not out.startswith("ERROR"), out
    assert "2 sheet(s)" in out
    assert "sheet 'Sales'" in out and "sheet 'Other'" in out
    assert "rows: 4" in out
    assert "Burger" in out and "Price" in out
    assert "Type" in out and "'Drink'" in out and "'Food'" in out  # category-like column
    assert "Price=9.75" in out  # numeric sum
    assert "FF00FF00" in out and "A2='Burger'" in out  # cell fill colour


def test_read_file_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,score\nann,3\nbob,4\n")
    out = read_file(str(path))
    assert "rows: 2" in out and "ann" in out and "score=7" in out


def test_read_file_text_py_json(tmp_path):
    py = tmp_path / "code.py"
    py.write_text("print(6 * 7)\n")
    assert "print(6 * 7)" in read_file(str(py))
    js = tmp_path / "d.json"
    js.write_text('{"a": [1, 2]}')
    assert '"a"' in read_file(str(js))


def test_read_file_media_hints(tmp_path):
    from PIL import Image

    img = tmp_path / "pic.png"
    Image.new("RGB", (10, 20)).save(img)
    out = read_file(str(img))
    assert "image_question" in out and "10x20" in out
    aud = tmp_path / "a.mp3"
    aud.write_bytes(b"\x00" * 10)
    assert "transcribe_audio" in read_file(str(aud))


def test_read_file_pdf(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    path = tmp_path / "blank.pdf"
    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        w.write(f)
    out = read_file(str(path))
    assert "1 page(s)" in out


def test_read_file_missing_and_data_files_lookup(tmp_path, _isolated_data_dir):
    assert read_file(str(tmp_path / "nope.xlsx")).startswith("ERROR")
    files = _isolated_data_dir / "files"
    files.mkdir()
    (files / "abc123_notes.txt").write_text("hello from data/files")
    # Bare file name resolves to data/files/<task_id>_<name>.
    assert "hello from data/files" in read_file("notes.txt")


def test_run_python_file(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import sys\nprint(sum(range(10)))\nprint('warn', file=sys.stderr)\n")
    out = run_python_file(str(script))
    assert "exit code: 0" in out and "45" in out and "warn" in out


def test_run_python_file_timeout(tmp_path):
    script = tmp_path / "loop.py"
    script.write_text("print('start', flush=True)\nimport time\ntime.sleep(10)\n")
    out = run_python_file(str(script), timeout_s=1)
    assert out.startswith("ERROR: timed out")


def test_tools_never_raise():
    assert read_file("").startswith("ERROR")
    assert run_python_file("/definitely/missing.py").startswith("ERROR")
    assert transcribe_audio("/definitely/missing.mp3").startswith("ERROR")
    assert image_question("/definitely/missing.png", "what?").startswith("ERROR")


def test_vtt_dedupe():
    raw = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
hello there

00:00:02.000 --> 00:00:04.000
hello there
<00:00:02.500><c>general</c><00:00:03.000><c> kenobi</c>

00:00:04.000 --> 00:00:06.000
general kenobi
"""
    lines = tm._dedupe(tm._parse_vtt(raw))
    assert [t for _, t in lines] == ["hello there", "general kenobi"]
    assert lines[1][0] == 2.0


def test_transcribe_audio_synthetic(tmp_path):
    if not shutil.which("say"):
        pytest.skip("macOS say not available")
    aiff = tmp_path / "out.aiff"
    wav = tmp_path / "out.wav"
    subprocess.run(["say", "-o", str(aiff), "hello world, this is a test"], check=True)
    subprocess.run([tm._ffmpeg(), "-loglevel", "error", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(wav)],
                   check=True)
    out = transcribe_audio(str(wav))
    assert "hello" in out.lower(), out
    # Second call is served from the transcript cache, with timestamps.
    out2 = transcribe_audio(str(wav), timestamps=True)
    assert "hello" in out2.lower() and "[00:00-" in out2


@pytest.mark.network
def test_youtube_transcript_network():
    out = youtube_transcript(CAPTIONED_VIDEO)
    assert not out.startswith("ERROR"), out
    assert "Captions:" in out and "[00:" in out, out[:500]


@pytest.mark.network
@pytest.mark.skipif(not _vision_available(), reason="no vision model key/endpoint configured")
def test_image_question_vision(tmp_path):
    from PIL import Image, ImageDraw

    img = tmp_path / "red.png"
    im = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(im).rectangle([50, 50, 150, 150], fill="red")
    im.save(img)
    out = image_question(str(img), "What colour is the square? Answer with one word.")
    assert "red" in out.lower(), out


@pytest.mark.network
@pytest.mark.skipif(not _vision_available(), reason="no vision model key/endpoint configured")
def test_youtube_frames_question_vision():
    out = youtube_frames_question(CAPTIONED_VIDEO, "What animals are visible?", num_frames=4)
    assert not out.startswith("ERROR"), out
    assert "batch 1" in out
