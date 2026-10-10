"""Build the practice set: 12 generated-file tasks + 4 live-Wikipedia multi-hop tasks.

    .venv/bin/python -m evals.practice.build_practice            # (re)write files + tasks.jsonl
    .venv/bin/python -m evals.practice.build_practice --audio    # also re-synthesize the audio

Every task is original (written for this repo, not taken or paraphrased from GAIA or the
course's 20 questions) and GAIA-Level-1 shaped: one attached file or a 2-hop lookup, a short
exact-match answer, and a trap that punishes the usual agent failure (strict vs non-strict
bounds, missing values treated as zero, per-sheet vs combined totals, a near-miss name, ...).

Gold answers for generated tasks are never typed by hand: each is DERIVED FROM THE FILE ON DISK
after writing it (``derive_gold``), and the generator asserts that the derivation matches the
value it intended. tests/test_evals.py re-derives every one again with independent code.

Wikipedia golds were checked against the live pages on 2026-10-10; each task stores the URL, the
revision id read, and the supporting sentence (see WIKI_TASKS).

Deterministic: fixed seeds, fixed document timestamps. Audio is synthesized with macOS `say`
(voice Samantha) + ffmpeg, so it is only rebuilt with --audio on a Mac; the committed mp3s and
their TTS scripts (stored in each task's ``source``) are the reference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

PRACTICE_DIR = Path(__file__).resolve().parent
FILES_DIR = PRACTICE_DIR / "files"
TASKS_PATH = PRACTICE_DIR / "tasks.jsonl"
MANIFEST_PATH = PRACTICE_DIR / "manifest.json"
REL_ROOT = PRACTICE_DIR.parent.parent  # project root; task "file" paths are relative to it
GENERATOR = "evals/practice/build_practice.py"
RETRIEVED = "2026-10-10"
FIXED_TS = dt.datetime(2026, 10, 10, 12, 0, 0)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
TEXT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def rel(p: Path) -> str:
    return p.resolve().relative_to(REL_ROOT).as_posix()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _font(size: int, candidates=TEXT_FONT_CANDIDATES):
    from PIL import ImageFont

    for c in candidates:
        if Path(c).is_file():
            return ImageFont.truetype(c, size)
    raise RuntimeError(f"No usable TrueType font among {candidates}")


def _fmt_num(x: float) -> str:
    """Gold-number formatting: integers without a decimal point, otherwise minimal digits."""
    if float(x).is_integer():
        return str(int(x))
    return f"{x}".rstrip("0").rstrip(".")


# =========================================================================== P01 xlsx: loans

LOAN_MEMBERS = {
    "Student": ["Ava Lindqvist", "Ben Osei", "Chloe Marsh", "Dev Patel", "Elif Kaya", "Femi Adeyemi"],
    "Staff": ["Grace Holt", "Hugo Brandt", "Ines Duarte"],
    "Faculty": ["Jonah Weiss", "Keiko Mori"],
    "Community": ["Liam Doyle", "Maya Rios", "Noor Haddad"],
}


def _loans_rows() -> list[dict]:
    rng = random.Random(4101)
    rows = []
    start = dt.date(2026, 2, 15)
    for i in range(58):
        tier = rng.choice(["Student"] * 5 + ["Staff"] * 2 + ["Faculty"] + ["Community"] * 2)
        member = rng.choice(LOAN_MEMBERS[tier])
        out = start + dt.timedelta(days=rng.randint(0, 60))
        days = rng.choice([None] + list(range(4, 36)))
        ret = out + dt.timedelta(days=days) if days is not None else None
        rows.append({"Member": member, "Tier": tier, "Checked Out": out, "Returned": ret})
    # Boundary cases that a careless filter gets wrong.
    rows += [
        {"Member": "Ava Lindqvist", "Tier": "Student", "Checked Out": dt.date(2026, 3, 2), "Returned": dt.date(2026, 3, 23)},  # exactly 21 -> no
        {"Member": "Dev Patel", "Tier": "Student", "Checked Out": dt.date(2026, 3, 9), "Returned": dt.date(2026, 3, 31)},  # 22 -> yes
        {"Member": "Ben Osei", "Tier": "Student", "Checked Out": dt.date(2026, 2, 28), "Returned": dt.date(2026, 4, 2)},  # Feb -> no
        {"Member": "Chloe Marsh", "Tier": "Student", "Checked Out": dt.date(2026, 3, 31), "Returned": dt.date(2026, 4, 30)},  # 30, Mar 31 -> yes
        {"Member": "Elif Kaya", "Tier": "Student", "Checked Out": dt.date(2026, 3, 18), "Returned": None},  # not returned -> no
        {"Member": "Grace Holt", "Tier": "Staff", "Checked Out": dt.date(2026, 3, 5), "Returned": dt.date(2026, 4, 9)},  # staff -> no
    ]
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["Loan ID"] = f"L{1000 + i}"
    return rows


def build_p01(path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Loans"
    cols = ["Loan ID", "Member", "Tier", "Checked Out", "Returned"]
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
    for r in _loans_rows():
        ws.append([r[c] for c in cols])
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=5):
        for c in row:
            c.number_format = "yyyy-mm-dd"
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["D"].width = 13
    ws.column_dimensions["E"].width = 13
    _save_wb(wb, path)


def derive_p01(path: Path) -> str:
    import pandas as pd

    df = pd.read_excel(path)
    out = pd.to_datetime(df["Checked Out"])
    ret = pd.to_datetime(df["Returned"])
    m = (df["Tier"] == "Student") & (out.dt.year == 2026) & (out.dt.month == 3) & ret.notna()
    return str(int(((ret - out).dt.days[m] > 21).sum()))


def _save_wb(wb, path: Path) -> None:
    wb.properties.creator = "gaia-agent practice set"
    wb.properties.created = FIXED_TS
    wb.properties.modified = FIXED_TS
    wb.save(path)
    _freeze_zip(path)


def _freeze_zip(path: Path) -> None:
    """Make an .xlsx byte-reproducible: fixed zip entry times and a fixed core.xml modified stamp
    (openpyxl stamps both with the wall clock on save)."""
    import re
    import zipfile

    with zipfile.ZipFile(path) as zin:
        entries = [(info, zin.read(info.filename)) for info in zin.infolist()]
    stamp = FIXED_TS.strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info, data in entries:
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:)",
                              lambda m: m.group(1) + stamp.encode() + m.group(2), data)
            zi = zipfile.ZipInfo(info.filename, date_time=FIXED_TS.timetuple()[:6])
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            zout.writestr(zi, data)
    tmp.replace(path)


# =========================================================================== P02 xlsx: warehouses

WAREHOUSES = {
    "Leeds": [("KX-4471", "Hex bolt M8", 420), ("KX-2210", "Washer 8mm", 310), ("LT-0912", "Cable tie 200mm", 150),
              ("RM-3307", "Rivet 4mm", 95), ("PB-1180", "Pipe bracket", 60), ("KX-5032", "Lock nut M8", 240)],
    "Bristol": [("KX-2210", "Washer 8mm", 280), ("RM-3307", "Rivet 4mm", 390), ("PB-1180", "Pipe bracket", 515),
                ("LT-0912", "Cable tie 200mm", 205), ("SC-7754", "Wood screw 40mm", 330)],
    "Glasgow": [("RM-3307", "Rivet 4mm", 260), ("KX-2210", "Washer 8mm", 190), ("SC-7754", "Wood screw 40mm", 120),
                ("KX-4471", "Hex bolt M8", 45), ("LT-0912", "Cable tie 200mm", 300), ("PB-1180", "Pipe bracket", 30)],
}


def build_p02(path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    wb.remove(wb.active)
    bins = random.Random(4102)
    for name, items in WAREHOUSES.items():
        ws = wb.create_sheet(name)
        ws.append(["SKU", "Description", "Bin", "Qty On Hand"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for sku, desc, qty in items:
            ws.append([sku, desc, f"{bins.choice('ABCDE')}{bins.randint(1, 40):02d}", qty])
        ws.column_dimensions["B"].width = 18
    _save_wb(wb, path)


def derive_p02(path: Path) -> str:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    allrows = pd.concat(sheets.values(), ignore_index=True)
    totals = allrows.groupby("SKU")["Qty On Hand"].sum().sort_values(ascending=False)
    assert totals.iloc[0] > totals.iloc[1], "combined maximum must be unique"
    return str(totals.index[0])


# =========================================================================== P03 csv: trips


def _trips_rows() -> list[dict]:
    rng = random.Random(4103)
    routes = ["Coastal", "Valley", "Airport", "Uni Loop", "Harbour"]
    base = dt.datetime(2026, 5, 14, 5, 30)
    rows = []
    for _ in range(36):
        dep = base + dt.timedelta(minutes=rng.randrange(0, 18 * 60, 5))
        dur = rng.choice(list(range(35, 150, 5)))
        rows.append((rng.choice(routes), dep, dep + dt.timedelta(minutes=dur)))
    edge = [
        ("Airport", dt.datetime(2026, 5, 14, 18, 0), 95),  # at 18:00, 95 min -> yes
        ("Coastal", dt.datetime(2026, 5, 14, 17, 59), 120),  # 17:59 -> no
        ("Valley", dt.datetime(2026, 5, 14, 18, 30), 90),  # exactly 90 -> no
        ("Harbour", dt.datetime(2026, 5, 14, 23, 10), 105),  # crosses midnight, 105 -> yes
        ("Uni Loop", dt.datetime(2026, 5, 14, 22, 45), 85),  # crosses midnight, 85 -> no
    ]
    rows += [(r, d, d + dt.timedelta(minutes=m)) for r, d, m in edge]
    rows.sort(key=lambda r: r[1])
    return [
        {"trip_id": f"T{301 + i}", "route": r, "departure": d.strftime("%Y-%m-%d %H:%M"), "arrival": a.strftime("%Y-%m-%d %H:%M")}
        for i, (r, d, a) in enumerate(rows)
    ]


def build_p03(path: Path) -> None:
    import csv

    rows = _trips_rows()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["trip_id", "route", "departure", "arrival"])
        w.writeheader()
        w.writerows(rows)


def derive_p03(path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["departure", "arrival"])
    mins = (df["arrival"] - df["departure"]).dt.total_seconds() / 60
    late = df["departure"].dt.hour >= 18
    return str(int((late & (mins > 90)).sum()))


# =========================================================================== P04 csv: air quality

STATIONS = ["Harbour Road", "Mill Lane", "Kingsway", "Riverside Park", "Station Square", "Northgate"]
STATION_LEVEL = {"Harbour Road": 31.0, "Mill Lane": 18.0, "Kingsway": 24.5, "Riverside Park": 12.0,
                 "Station Square": 28.0, "Northgate": 21.0}
STATION_MISSING = {"Harbour Road": 9, "Northgate": 2, "Kingsway": 1}  # Harbour Road: highest true mean, most gaps


def build_p04(path: Path) -> None:
    import csv

    rng = random.Random(4104)
    rows = []
    for st in STATIONS:
        hours = list(range(24))
        missing = set(rng.sample(hours, STATION_MISSING.get(st, 0)))
        for h in hours:
            val = "" if h in missing else f"{max(1.0, rng.gauss(STATION_LEVEL[st], 3.0)):.1f}"
            rows.append({"timestamp": f"2026-01-20T{h:02d}:00", "station": st, "pm25_ugm3": val})
    rows.sort(key=lambda r: (r["timestamp"], r["station"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp", "station", "pm25_ugm3"])
        w.writeheader()
        w.writerows(rows)


def derive_p04(path: Path) -> str:
    import pandas as pd

    df = pd.read_csv(path)
    means = df.groupby("station")["pm25_ugm3"].mean().sort_values(ascending=False)  # NaN skipped
    zero = df.fillna({"pm25_ugm3": 0}).groupby("station")["pm25_ugm3"].mean().sort_values(ascending=False)
    assert list(means.index[:3]) != list(zero.index[:3]), "the missing-value trap must change the answer"
    assert all(means.iloc[i] - means.iloc[i + 1] > 0.5 for i in range(3)), "ranking margins too thin"
    return ", ".join(means.index[:3])


# =========================================================================== P05 py: queue simulation

QUEUE_SIM = '''"""Ticket-counter simulation used for weekend capacity planning."""


def arrivals(minute):
    """Customers arriving in a given minute: a rush at the top and bottom of each hour."""
    if minute % 30 < 5:
        return 3
    return 1 if minute % 3 == 0 else 0


def simulate(minutes, counters=2, service_time=4):
    """Return how many customers are still queueing (not being served) after `minutes`."""
    queue = 0
    busy = [0] * counters  # minutes left on the customer at each counter
    for t in range(minutes):
        queue += arrivals(t)
        for i in range(counters):
            if busy[i] > 0:
                busy[i] -= 1
            if busy[i] == 0 and queue > 0:
                queue -= 1
                busy[i] = service_time
    return queue


if __name__ == "__main__":
    print("customers waiting after 60 minutes:", simulate(60))
'''


def build_p05(path: Path) -> None:
    path.write_text(QUEUE_SIM, encoding="utf-8")


def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def derive_p05(path: Path) -> str:
    mod = _load_module(path, "_practice_queue_sim")
    default = mod.simulate(60)
    asked = mod.simulate(240, counters=3)
    assert asked != default and asked > 0
    return str(asked)


# =========================================================================== P06 py: cipher

CIPHER = '''"""Toy obfuscation used to tag warehouse labels. Not cryptography."""

KEY = "orbit"


def encode(word):
    out = []
    for i, ch in enumerate(word):
        shift = ord(KEY[i % len(KEY)]) - ord("a")
        out.append(chr((ord(ch) - ord("a") + shift + i) % 26 + ord("a")))
    return "".join(reversed(out))
'''
P06_PLAIN = "lantern"


def build_p06(path: Path) -> None:
    path.write_text(CIPHER, encoding="utf-8")


def p06_target(path: Path) -> str:
    return _load_module(path, "_practice_cipher").encode(P06_PLAIN)


def derive_p06(path: Path) -> str:
    """Invert encode() independently of the generator's plaintext."""
    mod = _load_module(path, "_practice_cipher")
    target = p06_target(path)
    chars = list(reversed(target))
    plain = "".join(
        chr((ord(c) - 97 - (ord(mod.KEY[i % len(mod.KEY)]) - 97) - i) % 26 + 97) for i, c in enumerate(chars)
    )
    assert mod.encode(plain) == target
    return plain


# =========================================================================== P07/P08 audio

P07_TTS = (
    "Good evening. This is a platform alteration. "
    "The eighteen fifteen service to Carlisle will now depart from platform nine, not platform four. "
    "The eighteen forty-two service to Lancaster remains at platform six. "
    "The nineteen oh five service to Preston is cancelled. "
    "Passengers for Preston should instead take the nineteen twenty service, departing from platform eleven. "
    "We apologise for any inconvenience."
)
P07_GOLD = "11"

P08_TTS = (
    "Sensor log for bay three, recorded at the start of the shift. All readings are in degrees Celsius. "
    "Reading one: 21.4. Reading two: 19.8. Reading three: 22.6. Reading four: 20.2. Reading five: 18.9. "
    "End of log."
)


def p08_readings(tts: str = P08_TTS) -> list[float]:
    import re

    return [float(x) for x in re.findall(r"Reading \w+: (\d+\.\d)", tts)]


def p08_gold(tts: str = P08_TTS) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    vals = [Decimal(str(v)) for v in p08_readings(tts)]
    mean = sum(vals) / len(vals)
    return str(mean.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def synth_audio(text: str, path: Path, voice: str = "Samantha") -> None:
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        raise RuntimeError("audio synthesis needs macOS `say` and ffmpeg")
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "speech.aiff"
        subprocess.run(["say", "-v", voice, "-r", "170", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ac", "1", "-ar", "16000",
             "-b:a", "48k", "-map_metadata", "-1", "-fflags", "+bitexact", str(path)],
            check=True,
        )


# =========================================================================== P09 chess image

P09_FEN = "6k1/1n1Q1ppp/6q1/R7/7N/B5b1/5PPP/6K1 w - - 0 1"
GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}


def p09_mates(fen: str = P09_FEN) -> list[str]:
    import chess

    b = chess.Board(fen)
    mates = []
    for m in list(b.legal_moves):
        san = b.san(m)
        b.push(m)
        if b.is_checkmate():
            mates.append(san)
        b.pop()
    return mates


def build_p09(path: Path, fen: str = P09_FEN) -> None:
    import chess
    from PIL import Image, ImageDraw

    sq, margin = 72, 30
    size = 8 * sq + 2 * margin
    img = Image.new("RGB", (size, size), (250, 248, 240))
    d = ImageDraw.Draw(img)
    piece_font = _font(60, FONT_CANDIDATES)
    label_font = _font(18)
    board = chess.Board(fen)
    for rank in range(8):
        for file in range(8):
            x0, y0 = margin + file * sq, margin + (7 - rank) * sq
            light = (rank + file) % 2 == 1
            d.rectangle([x0, y0, x0 + sq, y0 + sq], fill=(240, 217, 181) if light else (181, 136, 99))
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                g = GLYPH[piece.symbol().lower()]
                white = piece.color == chess.WHITE
                d.text((x0 + sq / 2, y0 + sq / 2 + 2), g, font=piece_font, anchor="mm",
                       fill=(255, 255, 255) if white else (20, 20, 20),
                       stroke_width=2 if white else 0, stroke_fill=(0, 0, 0))
    for i in range(8):
        d.text((margin + i * sq + sq / 2, size - margin / 2), "abcdefgh"[i], font=label_font, anchor="mm", fill=(40, 40, 40))
        d.text((margin / 2, margin + (7 - i) * sq + sq / 2), str(i + 1), font=label_font, anchor="mm", fill=(40, 40, 40))
    d.rectangle([margin, margin, margin + 8 * sq, margin + 8 * sq], outline=(40, 40, 40), width=2)
    img.save(path, format="PNG", optimize=True)


def derive_p09(path: Path) -> str:
    mates = p09_mates()
    assert len(mates) == 1, mates
    return mates[0]


# =========================================================================== P10 bar chart

P10_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
P10_VALUES = [88, 71, 57, 55, 49, 38, 29, 41, 66, 92, 97, 85]


def build_p10(path: Path) -> None:
    from PIL import Image, ImageDraw

    W, H = 1000, 560
    left, right, top, bottom = 70, 30, 70, 70
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title, small, tiny = _font(24), _font(16), _font(15)
    d.text((W / 2, 30), "Monthly rainfall at Elmsford Vale weather station, 2025 (mm)", font=title, anchor="mm", fill=(30, 30, 30))
    plot_h = H - top - bottom
    vmax = 110
    for v in range(0, vmax + 1, 20):
        y = top + plot_h - v / vmax * plot_h
        d.line([left, y, W - right, y], fill=(225, 225, 225), width=1)
        d.text((left - 10, y), str(v), font=tiny, anchor="rm", fill=(90, 90, 90))
    slot = (W - left - right) / 12
    for i, (m, v) in enumerate(zip(P10_MONTHS, P10_VALUES)):
        x0 = left + i * slot + slot * 0.18
        x1 = left + (i + 1) * slot - slot * 0.18
        y = top + plot_h - v / vmax * plot_h
        d.rectangle([x0, y, x1, top + plot_h], fill=(52, 101, 164))
        d.text(((x0 + x1) / 2, y - 12), str(v), font=small, anchor="mm", fill=(20, 20, 20))
        d.text(((x0 + x1) / 2, top + plot_h + 18), m[:3], font=small, anchor="mm", fill=(30, 30, 30))
    d.line([left, top + plot_h, W - right, top + plot_h], fill=(60, 60, 60), width=2)
    img.save(path, format="PNG", optimize=True)


def derive_p10() -> str:
    mean = sum(P10_VALUES) / len(P10_VALUES)
    dist = sorted((abs(v - mean), m) for m, v in zip(P10_MONTHS, P10_VALUES))
    assert dist[1][0] - dist[0][0] >= 1.0, "closest month must be clearly unique"
    return dist[0][1]


# =========================================================================== P11 PDF minutes

P11_ITEMS = [
    ("AI-01", "Priya Raman", "2026-11-03", "Circulate revised vendor shortlist"),
    ("AI-02", "Tomas Ortega", "2026-11-05", "Book site visit to the Leeds depot"),
    ("AI-03", "Priya Nair", "2026-11-06", "Collect insurance quotes"),
    ("AI-04", "Priya Raman", "2026-11-14", "Draft RFP section on data retention"),
    ("AI-05", "Helen Okafor", "2026-11-10", "Confirm budget envelope with finance"),
    ("AI-06", "Priya Raman", "2026-11-15", "Schedule security review"),
    ("AI-07", "Priya Nair", "2026-11-02", "Update risk register"),
    ("AI-08", "Priya Raman", "2026-12-01", "Prepare board summary"),
    ("AI-09", "Tomas Ortega", "2026-11-12", "Reconcile Q3 invoices"),
    ("AI-10", "Priya Raman", "2026-10-28", "Send meeting notes to the steering group"),
    ("AI-11", "Helen Okafor", "2026-11-20", "Review draft contract terms"),
]

P11_LINES = [
    "Northfield Logistics - Procurement Working Group",
    "Minutes of meeting held 21 October 2026, 14:00-15:10, Room 3B",
    "",
    "Present: Helen Okafor (chair), Priya Raman, Priya Nair, Tomas Ortega",
    "Apologies: Sam Whitfield",
    "",
    "1. Vendor shortlist. The group agreed to keep four vendors on the shortlist.",
    "2. Budget. Finance has not yet confirmed the envelope; Helen to follow up.",
    "3. Security. A review must happen before any contract is signed.",
    "4. AOB. Next meeting on 4 November 2026.",
    "",
    "Action items",
]


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_text_pdf(path: Path, lines: list[str], title: str = "Minutes") -> None:
    """A minimal, valid single-page PDF 1.4 with Helvetica text (no third-party library)."""
    ops = ["BT", "/F1 11 Tf", "14 TL", "50 790 Td"]
    for ln in lines:
        ops.append(f"({_pdf_escape(ln)}) Tj T*")
    ops.append("ET")
    stream = zlib.compress("\n".join(ops).encode("latin-1"))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ("<< /Title (%s) /Producer (gaia-agent practice set) /CreationDate (D:20261010120000Z) >>"
         % _pdf_escape(title)).encode("latin-1"),
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 6 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))


def build_p11(path: Path) -> None:
    lines = list(P11_LINES)
    for aid, owner, due, what in P11_ITEMS:
        lines.append(f"{aid} | {what} | Owner: {owner} | Due: {due}")
    write_text_pdf(path, lines, title="Procurement Working Group minutes")


def derive_p11(path: Path) -> str:
    import re

    from pypdf import PdfReader

    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    items = re.findall(r"(AI-\d+) \| .*? \| Owner: (.+?) \| Due: (\d{4}-\d{2}-\d{2})", text)
    assert len(items) == len(P11_ITEMS), (len(items), text[:500])
    n = sum(1 for _, owner, due in items if owner.strip() == "Priya Raman" and due < "2026-11-15")
    return str(n)


# =========================================================================== P12 JSON orders

P12_CUSTOMERS = [("Marta", "Silva"), ("Owen", "Pryce"), ("Lena", "Vogel"), ("Raj", "Iyer"), ("Sofia", "Brun"),
                 ("Theo", "Nakamura")]


def build_p12(path: Path) -> None:
    rng = random.Random(4112)
    catalog = {"DESK-01": 249.0, "CHAIR-07": 189.5, "LAMP-02": 39.99, "SHELF-11": 74.25, "MAT-03": 22.0}
    orders = []
    statuses = ["shipped"] * 6 + ["cancelled", "refunded", "pending"]
    for i in range(26):
        first, last = rng.choice(P12_CUSTOMERS)
        items = []
        for sku in rng.sample(sorted(catalog), rng.randint(1, 3)):
            items.append({"sku": sku, "qty": rng.randint(1, 3), "unit_price": catalog[sku]})
        orders.append({"order_id": f"SO-{5100 + i}", "customer": {"first": first, "last": last,
                       "email": f"{first.lower()}.{last.lower()}@example.org"},
                       "status": rng.choice(statuses), "items": items})
    # A big order that is NOT shipped, so "top overall" differs from "top shipped".
    orders.append({"order_id": "SO-5190", "customer": {"first": "Owen", "last": "Pryce", "email": "owen.pryce@example.org"},
                   "status": "cancelled", "items": [{"sku": "DESK-01", "qty": 16, "unit_price": 249.0},
                                                                {"sku": "CHAIR-07", "qty": 2, "unit_price": 189.5}]})
    doc = {"exported_at": "2026-09-30T23:59:00Z", "currency": "GBP", "orders": orders}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def derive_p12(path: Path) -> str:
    doc = json.loads(path.read_text(encoding="utf-8"))
    shipped: dict[str, float] = {}
    overall: dict[str, float] = {}
    for o in doc["orders"]:
        name = f'{o["customer"]["first"]} {o["customer"]["last"]}'
        total = sum(it["qty"] * it["unit_price"] for it in o["items"])
        overall[name] = overall.get(name, 0) + total
        if o["status"] == "shipped":
            shipped[name] = shipped.get(name, 0) + total
    top = sorted(shipped.items(), key=lambda kv: -kv[1])
    assert top[0][1] - top[1][1] > 1, "top shipped customer must be clearly unique"
    assert max(overall, key=overall.get) != top[0][0], "the status trap must change the answer"
    return top[0][0]


# =========================================================================== registry

FILE_TASKS = [
    {
        "id": "p01-xlsx-late-loans", "file": "library_loans.xlsx", "build": build_p01, "derive": derive_p01,
        "format": "number", "skills": ["file:xlsx", "filter", "date-arithmetic", "count", "strict-bound"],
        "question": "The attached spreadsheet is a library's loan log. Consider only loans by members in the "
                    "Student tier that were checked out during March 2026. How many of those loans were returned "
                    "more than 21 days after the checkout date? Loans with no return date do not count.",
        "derivation": "Tier == Student, checkout in 2026-03, Returned not blank, (Returned - Checked Out).days > 21",
    },
    {
        "id": "p02-xlsx-multisheet-sku", "file": "warehouse_stock.xlsx", "build": build_p02, "derive": derive_p02,
        "format": "string", "skills": ["file:xlsx", "multi-sheet", "group-by", "argmax"],
        "question": "The attached workbook has one sheet per warehouse. Summed across all warehouses, which SKU has "
                    "the largest total quantity on hand? Answer with the SKU code only.",
        "derivation": "concat all sheets, group by SKU, sum Qty On Hand, argmax (unique)",
    },
    {
        "id": "p03-csv-evening-trips", "file": "shuttle_trips.csv", "build": build_p03, "derive": derive_p03,
        "format": "number", "skills": ["file:csv", "datetime", "filter", "count", "strict-bound"],
        "question": "The attached CSV lists one day's shuttle trips. How many trips departed at or after 18:00 and "
                    "took longer than 90 minutes from departure to arrival?",
        "derivation": "departure.hour >= 18 and (arrival - departure) > 90 min (overnight arrivals included)",
    },
    {
        "id": "p04-csv-pm25-ranking", "file": "air_quality_hourly.csv", "build": build_p04, "derive": derive_p04,
        "format": "ordered_list", "skills": ["file:csv", "missing-values", "group-by", "ranking", "ordered-list"],
        "question": "The attached file has hourly PM2.5 readings from several monitoring stations; blank cells are "
                    "missed readings and should be ignored, not treated as zero. Which three stations have the "
                    "highest mean PM2.5? List the station names, highest mean first, separated by commas.",
        "derivation": "groupby(station).mean() with NaN skipped, top 3 descending",
    },
    {
        "id": "p05-py-queue-sim", "file": "queue_sim.py", "build": build_p05, "derive": derive_p05,
        "format": "number", "skills": ["file:py", "code-execution", "modify-parameters"],
        "question": "The attached Python script simulates a ticket counter and prints how many customers are "
                    "waiting after 60 minutes. If the same simulation were run for 240 minutes with three counters "
                    "open instead of the default (everything else unchanged), how many customers would still be "
                    "waiting at the end?",
        "derivation": "simulate(240, counters=3) from the attached module",
    },
    {
        "id": "p06-py-invert-cipher", "file": "label_cipher.py", "build": build_p06, "derive": derive_p06,
        "format": "string", "skills": ["file:py", "code-reasoning", "inversion"],
        "question": None,  # filled in from the generated file (needs the ciphertext)
        "derivation": "invert encode(): reverse, then subtract key shift and position index mod 26",
    },
    {
        "id": "p07-audio-platform", "file": "station_announcement.mp3", "build": None, "derive": None,
        "format": "number", "skills": ["file:audio", "transcription", "distractors"],
        "question": "Listen to the attached station announcement. From which platform should a passenger travelling "
                    "to Preston now depart? Answer with the platform number only.",
        "derivation": "TTS script (source.tts_text): the 19:05 to Preston is cancelled; take the 19:20 from platform eleven",
        "tts": P07_TTS,
    },
    {
        "id": "p08-audio-mean-reading", "file": "sensor_readings.mp3", "build": None, "derive": None,
        "format": "number", "skills": ["file:audio", "transcription", "arithmetic", "rounding"],
        "question": "The attached recording is a technician reading out temperature measurements in degrees "
                    "Celsius. What is the mean of the readings, rounded to one decimal place?",
        "derivation": "mean of the five readings in the TTS script, ROUND_HALF_UP to 0.1",
        "tts": P08_TTS,
    },
    {
        "id": "p09-image-chess-mate", "file": "chess_diagram.png", "build": build_p09, "derive": derive_p09,
        "format": "string", "skills": ["file:image", "vision", "chess", "verification"],
        "question": "The attached image shows a chess position with White to move (White pieces are drawn in white, "
                    "Black pieces in black; White plays up the board from rank 1). White has exactly one move that "
                    "delivers checkmate immediately. Give that move in standard algebraic notation (SAN), "
                    "including the mate symbol.",
        "derivation": "python-chess: legal moves from source.fen whose result is checkmate (exactly one)",
    },
    {
        "id": "p10-image-chart-mean", "file": "rainfall_chart.png", "build": build_p10, "derive": None,
        "format": "string", "skills": ["file:image", "vision", "chart-reading", "arithmetic"],
        "question": "The attached bar chart shows a weather station's monthly rainfall for one year. Which month's "
                    "rainfall was closest to that year's monthly average? Give the full month name.",
        "derivation": "argmin |value - mean(values)| over the 12 plotted values (source.values)",
    },
    {
        "id": "p11-pdf-action-items", "file": "procurement_minutes.pdf", "build": build_p11, "derive": derive_p11,
        "format": "number", "skills": ["file:pdf", "filter", "date-comparison", "near-miss-names", "count"],
        "question": "How many action items in the attached meeting minutes are owned by Priya Raman and are due "
                    "before 15 November 2026?",
        "derivation": "parse 'AIxx | ... | Owner: X | Due: D' lines; Owner == 'Priya Raman' and D < 2026-11-15",
    },
    {
        "id": "p12-json-top-customer", "file": "orders_export.json", "build": build_p12, "derive": derive_p12,
        "format": "string", "skills": ["file:json", "nested-data", "filter", "group-by", "argmax"],
        "question": "The attached JSON is an order export. Counting only orders whose status is \"shipped\", which "
                    "customer spent the most in total (sum of quantity times unit price over their items)? Give the "
                    "customer's full name as First Last.",
        "derivation": "status == shipped; per customer sum(qty * unit_price); argmax (unique)",
    },
]


def _wiki(title: str, revid: int, evidence: str) -> dict:
    return {
        "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
        "revid": revid,
        "retrieved": RETRIEVED,
        "evidence": evidence,
    }


WIKI_TASKS = [
    {
        "id": "w01-wiki-element-city", "format": "number", "gold": "67",
        "skills": ["web", "wikipedia", "multi-hop", "number"],
        "question": "A chemical element is named after the Latin name of the capital city of Sweden. What is that "
                    "element's atomic number?",
        "hops": [
            _wiki("Holmium", 1377857933, "The element's name comes from Holmia, the Latin name for the city of Stockholm."),
            _wiki("Holmium", 1377857933, "Holmium is a chemical element; it has symbol Ho and atomic number 67."),
        ],
    },
    {
        "id": "w02-wiki-architect-prize", "format": "number", "gold": "2003",
        "skills": ["web", "wikipedia", "multi-hop", "year"],
        "question": "In what year was the architect who designed the Sydney Opera House awarded the Pritzker "
                    "Architecture Prize?",
        "hops": [
            _wiki("Sydney Opera House", 1378768237,
                  "Designed by Danish architect Jørn Utzon and completed by an Australian architectural team headed "
                  "by Peter Hall, the building was formally opened by Queen Elizabeth II on 20 October 1973, ..."),
            _wiki("Jørn Utzon", 1373787728, "In 2003, he was awarded the Pritzker Prize, architecture's highest honor."),
        ],
    },
    {
        "id": "w03-wiki-bolivia-borders", "format": "ordered_list", "gold": "Argentina, Brazil, Chile, Paraguay, Peru",
        "skills": ["web", "wikipedia", "list", "ordered-list", "alphabetical"],
        "question": "Which countries share a land border with Bolivia? List them in alphabetical order, separated by "
                    "commas.",
        "hops": [
            _wiki("Bolivia", 1378338095,
                  "Bolivia is bordered by Brazil to the north and east, Paraguay to the southeast, Argentina to the "
                  "south, Chile to the southwest, and Peru to the west."),
        ],
    },
    {
        "id": "w04-wiki-mozart-river", "format": "string", "gold": "Salzach",
        "skills": ["web", "wikipedia", "multi-hop", "entity"],
        "question": "Which river flows through the city where Wolfgang Amadeus Mozart was born? Give only the "
                    "river's name, without the word \"River\".",
        "hops": [
            _wiki("Wolfgang Amadeus Mozart", 1379458284,
                  "Born in Salzburg, Mozart quickly emerged as a child prodigy under the training of his father "
                  "Leopold, a skilled pedagogue."),
            _wiki("Salzburg", 1378187032,
                  "The city lies on the Salzach River, near the border with Germany and at the foot of the Alps mountains."),
        ],
    },
]


# =========================================================================== build


def build(audio: bool = False, quiet: bool = False) -> list[dict]:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    tasks: list[dict] = []
    manifest: dict = {"generator": GENERATOR, "built_for": RETRIEVED, "files": {}}

    for spec in FILE_TASKS:
        path = FILES_DIR / spec["file"]
        source: dict = {"type": "generated", "generator": GENERATOR, "derivation": spec["derivation"]}
        if spec.get("tts"):
            if audio or not path.exists():
                synth_audio(spec["tts"], path)
            source.update({"tts_text": spec["tts"], "tts_voice": "Samantha (macOS say, rate 170)"})
            gold = P07_GOLD if spec["id"].startswith("p07") else p08_gold()
        else:
            spec["build"](path)
            if spec["id"].startswith("p10"):
                gold = derive_p10()
                source["values"] = dict(zip(P10_MONTHS, P10_VALUES))
            else:
                gold = spec["derive"](path)
        if spec["id"].startswith("p09"):
            source["fen"] = P09_FEN
        question = spec["question"]
        if spec["id"].startswith("p06"):
            target = p06_target(path)
            question = (f"The attached Python module defines a function encode(). Which lowercase English word w "
                        f"satisfies encode(w) == '{target}'?")
            source["ciphertext"] = target
            assert gold == P06_PLAIN, gold
        tasks.append({
            "id": spec["id"], "question": question, "file": rel(path), "gold": gold,
            "format": spec["format"], "skills": spec["skills"], "source": source,
        })
        manifest["files"][rel(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    for spec in WIKI_TASKS:
        tasks.append({
            "id": spec["id"], "question": spec["question"], "file": None, "gold": spec["gold"],
            "format": spec["format"], "skills": spec["skills"],
            "source": {"type": "wikipedia", "retrieved": RETRIEVED, "hops": spec["hops"]},
        })

    # Generator self-checks against intended values (the test suite re-derives independently).
    intended = {"p05-py-queue-sim": None, "p07-audio-platform": "11", "p09-image-chess-mate": "Qe8#",
                "p10-image-chart-mean": "September", "p06-py-invert-cipher": P06_PLAIN}
    for t in tasks:
        want = intended.get(t["id"])
        if want is not None:
            assert t["gold"] == want, (t["id"], t["gold"], want)

    with open(TASKS_PATH, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not quiet:
        for t in tasks:
            print(f"{t['id']:<28} {t['format']:<13} gold={t['gold']}")
        print(f"wrote {len(tasks)} tasks -> {rel(TASKS_PATH)}")
    return tasks


def load_tasks(path: Path = TASKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio", action="store_true", help="re-synthesize the audio files (macOS only)")
    args = ap.parse_args(argv)
    build(audio=args.audio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
