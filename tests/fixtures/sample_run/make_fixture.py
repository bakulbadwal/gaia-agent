"""Regenerate the FAKE sample run used to develop and test report/build_report.py.

Everything here is invented: fictional questions about fictional places, made-up
answers, made-up scores. It exists only so the results site can be built and
tested before (and independently of) a real run. It is never presented as results:
the FIXTURE marker file makes the site show a banner, and build_report.py refuses
to write fixture data into docs/.

    python tests/fixtures/sample_run/make_fixture.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = "fixture-run-002"
MODEL = "fixture/fake-model-8b"
PROVIDER = "fixture"
rng = random.Random(7)


def code_out(thought: str, code: str) -> str:
    return f"Thought: {thought}\n<code>\n{code}\n</code>"


def logs(printed: str, last: str = "None") -> str:
    return f"Execution logs:\n{printed}\nLast output from code snippet:\n{last}"


def step(i, typ, thought=None, code=None, obs=None, err=None, tin=0, tout=0, dur=0.0, calls=None, plan=None):
    s = {
        "index": i,
        "type": typ,
        "model_output": plan if plan is not None else (code_out(thought, code) if code else thought),
        "code": code,
        "tool_calls": calls
        if calls is not None
        else ([{"name": "python_interpreter", "arguments": code}] if code else []),
        "observation": obs,
        "error": err,
        "input_tokens": tin,
        "output_tokens": tout,
        "duration_s": dur,
    }
    return s


PLAN = """## 1. Facts survey
### 1.1. Facts given in the task
- {given}
### 1.2. Facts to look up
- {lookup}
### 1.3. Facts to derive
- {derive}

## 2. Plan
1. {p1}
2. {p2}
3. Format the answer exactly as asked and call final_answer.
<end_plan>"""

SUB_Q01 = {
    "agent": "web_agent",
    "task": "Open https://library.example-falls.test/authors/placeholder-ada and return the full Year | Title table "
            "verbatim.",
    "final_answer": "| Year | Title |\n| 1971 | The Quiet Mill |\n| 1978 | Rivers of Example |\n"
                    "| 1984 | A Map of Nowhere |\n| 1989 | Late Light |\n| 1990 | The Second Bridge |\n"
                    "| 2004 | Collected Placeholders |",
    "seconds": 5.4,
    "partial": False,
    "steps": [
        {"index": 0, "type": "task", "agent": "web_agent", "model_output": "Open the catalogue page and return the table.",
         "code": None, "tool_calls": [], "observation": None, "error": None, "input_tokens": None,
         "output_tokens": None, "duration_s": None},
        {"index": 1, "type": "action", "agent": "web_agent", "step_number": 1,
         "model_output": "Thought: Read the page and print it.\n<code>\n"
                         "page = visit_webpage(\"https://library.example-falls.test/authors/placeholder-ada\")\n"
                         "print(page)\n</code>",
         "code": "page = visit_webpage(\"https://library.example-falls.test/authors/placeholder-ada\")\nprint(page)",
         "tool_calls": [{"name": "visit_webpage", "arguments": "(fixture)"}],
         "observation": "Execution logs:\n# Placeholder, Ada\n| Year | Title |\n| 1971 | The Quiet Mill |\n...\n"
                        "Last output from code snippet:\nNone",
         "error": None, "input_tokens": 1650, "output_tokens": 61, "duration_s": 3.6},
        {"index": 2, "type": "action", "agent": "web_agent", "step_number": 2, "is_final_answer": True,
         "model_output": "Thought: Return the table section verbatim.\n<code>\nfinal_answer(page.split('\\n\\n')[2])\n</code>",
         "code": "final_answer(page.split('\\n\\n')[2])",
         "tool_calls": [{"name": "final_answer", "arguments": "(fixture)"}],
         "observation": None, "error": None, "input_tokens": 2210, "output_tokens": 40, "duration_s": 1.5},
    ],
}

LONG_PAGE = "\n".join(
    f"| {1950 + i} | Placeholder Annual Report vol. {i} | Example Falls Press | {rng.randint(80, 400)} pp |"
    for i in range(140)
)

QUESTIONS = [
    {
        "id": "fixture-q01-library",
        "q": "The public library of the fictional town of Example Falls keeps an online catalogue. How many books "
        "by the author Ada Placeholder in that catalogue were published before 1990?",
        "answer": "4",
        "steps": [
            step(0, "planning", plan=PLAN.format(
                given="Library: Example Falls public library; author: Ada Placeholder; cutoff: before 1990 (exclusive).",
                lookup="The catalogue listing for Ada Placeholder, with publication years.",
                derive="Count of titles with year < 1990.",
                p1="Search the web for the catalogue page.",
                p2="Open the page, print every title with its year, count in code.",
            ), tin=1840, tout=212, dur=3.1),
            step(1, "action", "I will search for the catalogue page first.",
                 'results = web_search("Example Falls public library catalogue Ada Placeholder")\nprint(results)',
                 logs("1. Example Falls Library — Author: Placeholder, Ada\n   https://library.example-falls.test/authors/placeholder-ada\n"
                      "   Catalogue entries for Ada Placeholder (1971–2004)...\n2. Ada Placeholder — fictional author (fixture)\n"
                      "   https://wiki.example.test/Ada_Placeholder"),
                 tin=2410, tout=96, dur=4.8),
            dict(step(2, "action", "The first result looks like the catalogue. I will hand the page reading to the "
                      "web agent and ask for the raw table, not a summary.",
                      'table = web_agent(task="Open https://library.example-falls.test/authors/placeholder-ada and '
                      'return the full Year | Title table verbatim.")\nprint(table)',
                 logs("# Placeholder, Ada\n\n<script>alert('this is page content, not code to run')</script>\n"
                      "<img src=x onerror=alert(1)>\n\n| Year | Title |\n|---|---|\n| 1971 | The Quiet Mill |\n"
                      "| 1978 | Rivers of Example |\n| 1984 | A Map of Nowhere |\n| 1989 | Late Light |\n"
                      "| 1990 | The Second Bridge |\n| 2004 | Collected Placeholders |"),
                 tin=3920, tout=88, dur=6.2), sub_agents=[SUB_Q01]),
            step(3, "action", "Six titles. 'Before 1990' is exclusive, so 1990 itself does not count. I will count in code.",
                 "years = [1971, 1978, 1984, 1989, 1990, 2004]\nbefore = [y for y in years if y < 1990]\nprint(before, len(before))",
                 logs("[1971, 1978, 1984, 1989] 4"), tin=4410, tout=74, dur=2.0),
            step(4, "final", "The count is 4.", 'final_answer("4")', logs("", "4"), tin=4590, tout=31, dur=1.4),
        ],
        "seconds": 19.6,
    },
    {
        "id": "fixture-q02-spreadsheet",
        "q": "The attached spreadsheet lists sales for a fictional food-truck chain by region and item. What were the "
        "total sales of the East region in USD? Express your answer with two decimal places.",
        "file": "fixture_sales.xlsx",
        "answer": "1284.50",
        "steps": [
            step(1, "action", "There is an attached file, so I inspect it before anything else.",
                 'print(read_file("data/files/fixture-q02-spreadsheet.xlsx"))',
                 logs("Sheet 'Sales' (24 rows x 4 cols)\n| Region | Item | Units | Sales |\n|---|---|---|---|\n"
                      "| East | Tacos | 41 | 287.00 |\n| East | Burritos | 33 | 330.00 |\n| West | Tacos | 28 | 196.00 |\n"
                      "| East | Drinks | 115 | 287.50 |\n| East | Churros | 76 | 380.00 |\n| North | Drinks | 90 | 225.00 |\n…"),
                 tin=2050, tout=71, dur=2.9),
            step(2, "action", "I sum the East rows with pandas rather than by hand.",
                 'import pandas as pd\ndf = pd.read_excel("data/files/fixture-q02-spreadsheet.xlsx")\n'
                 'east = df[df["Region"] == "East"]["Sales"].sum()\nprint(f"{east:.2f}")',
                 logs("1284.50"), tin=2690, tout=82, dur=1.8),
            step(3, "final", "Total is 1284.50, two decimals as asked.", 'final_answer("1284.50")', logs("", "1284.50"),
                 tin=2810, tout=29, dur=1.1),
        ],
        "seconds": 8.4,
    },
    {
        "id": "fixture-q03-voice-memo",
        "q": "A friend left the attached voice memo with a recipe for a fictional pesto. List the ingredients they "
        "mention, alphabetized, as a comma-separated list.",
        "file": "fixture_memo.mp3",
        "answer": "basil, garlic, olive oil, parmesan, pine nuts",
        "steps": [
            step(1, "action", "Audio file: transcribe it locally first.",
                 'text = transcribe_audio("data/files/fixture-q03-voice-memo.mp3")\nprint(text)',
                 logs("Hey, so for the pesto you want a big bunch of basil, two cloves of garlic, a handful of pine nuts, "
                      "some parmesan, and good olive oil. Salt to taste but that's not really an ingredient, ha."),
                 tin=1980, tout=64, dur=11.7),
            step(2, "action", "Salt is explicitly dismissed by the speaker; I keep the five named ingredients and sort them.",
                 'items = ["basil", "garlic", "pine nuts", "parmesan", "olive oil"]\nprint(", ".join(sorted(items)))',
                 logs("basil, garlic, olive oil, parmesan, pine nuts"), tin=2240, tout=70, dur=1.3),
            step(3, "final", "Done.", 'final_answer("basil, garlic, olive oil, parmesan, pine nuts")',
                 logs("", "basil, garlic, olive oil, parmesan, pine nuts"), tin=2330, tout=36, dur=0.9),
        ],
        "seconds": 16.2,
    },
    {
        "id": "fixture-q04-video-birds",
        "q": "In the fixture video https://www.youtube.com/watch?v=FIXTURE0001, what is the highest number of "
        "birds visible on screen at the same time?",
        "answer": "3",
        "steps": [
            step(1, "action", "Try captions first; they are cheap.",
                 'print(youtube_transcript("https://www.youtube.com/watch?v=FIXTURE0001"))',
                 logs("No captions available for this video (it has no subtitles or automatic captions)."),
                 tin=1890, tout=58, dur=3.4),
            step(2, "action", "No captions, and the question is visual anyway. Sample frames and ask the vision model.",
                 'ans = youtube_frames_question("https://www.youtube.com/watch?v=FIXTURE0001",\n'
                 '    "How many birds are visible in this frame? Answer with a number.", num_frames=16)\nprint(ans)',
                 logs("frame 00:04 → 1\nframe 00:12 → 2\nframe 00:19 → 3\nframe 00:27 → 2\nframe 00:35 → 3\n"
                      "frame 00:41 → 1\n(12 more frames, all ≤ 3)"),
                 tin=2520, tout=101, dur=48.6),
            step(3, "final", "Maximum over sampled frames is 3.", 'final_answer("3")', logs("", "3"),
                 tin=2700, tout=27, dur=1.0),
        ],
        "seconds": 55.9,
    },
    {
        "id": "fixture-q05-wiki-revision",
        "q": "As of 1 January 2020, how many top-level sections did the (fictional) Wikipedia article "
        "'List of Example Falls landmarks' have?",
        "answer": "7",
        "steps": [
            step(1, "action", "This is an 'as of' question, so I need the revision live on that date, not today's page.",
                 'print(wikipedia_revision("List of Example Falls landmarks", as_of="2020-01-01")[:2500])',
                 logs("Revision 931000001 (2019-12-18T09:12:44Z), live on 2020-01-01.\nSections:\n"
                      "1 History\n2 Bridges\n3 Mills\n3.1 Upper mill\n4 Churches\n5 Parks\n6 See also\n7 References"),
                 tin=2110, tout=77, dur=5.5),
            step(2, "action", "Top-level means I exclude 3.1. Count programmatically.",
                 'secs = ["1","2","3","3.1","4","5","6","7"]\nprint(len([s for s in secs if "." not in s]))',
                 logs("7"), tin=2480, tout=62, dur=1.2),
            step(3, "final", "Seven top-level sections.", 'final_answer("7")', logs("", "7"), tin=2560, tout=24, dur=0.8),
        ],
        "seconds": 9.9,
    },
    {
        "id": "fixture-q06-rate-limit",
        "q": "Which fictional composer wrote the 'Example Falls Overture', according to the town's orchestra website?",
        "answer": "",
        "error": "RateLimitError: 429 Too Many Requests from provider 'fixture' after 3 retries",
        "steps": [
            step(1, "action", "Search for the orchestra website.",
                 'print(web_search("Example Falls Overture composer orchestra"))',
                 logs("1. Example Falls Civic Orchestra — Repertoire\n   https://orchestra.example-falls.test/repertoire"),
                 tin=1930, tout=60, dur=4.1),
            step(2, "action", None, None, None,
                 err="Error while generating output: RateLimitError: 429 Too Many Requests", dur=31.0),
        ],
        "seconds": 37.3,
    },
    {
        "id": "fixture-q07-timeout",
        "q": "How many distinct bridge designs appear across all editions of the fictional 'Placeholder Annual "
        "Report' series?",
        "answer": "",
        "error": "TimeoutError: question exceeded the 300 s per-question budget",
        "steps": [
            step(i, "action", "Keep paging through the report index." if i > 1 else "Find the series index.",
                 f'print(visit_webpage("https://press.example-falls.test/annual-reports", start={(i - 1) * 15000}))',
                 logs(LONG_PAGE if i == 2 else f"(page {i} of the index, 15000 chars)"),
                 tin=2000 + 1900 * i, tout=70, dur=24.0 + i)
            for i in range(1, 11)
        ],
        "seconds": 300.0,
    },
    {
        "id": "fixture-q08-chess",
        "q": "The attached image shows a (made-up) chess position with White to move. Give White's winning move "
        "in algebraic notation.",
        "file": "fixture_board.png",
        "answer": "Qh5",
        "steps": [
            step(1, "action", "Ask the vision model to read the board as FEN so I can verify with python-chess.",
                 'fen = image_question("data/files/fixture-q08-chess.png", "Give the position as FEN, White to move.")\nprint(fen)',
                 logs("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
                 tin=2150, tout=88, dur=7.9),
            step(2, "action", "Check candidate moves with python-chess instead of trusting the vision model's judgement.",
                 'import chess\nb = chess.Board(fen)\nprint([b.san(m) for m in b.legal_moves][:12])',
                 None, err="Code execution failed at line 'b = chess.Board(fen)': ValueError: expected 'w' or 'b' "
                 "for turn part of fen: 'w\\n'",
                 tin=2560, tout=95, dur=1.6),
            step(3, "action", "The FEN string has a trailing newline; strip it and retry.",
                 'b = chess.Board(fen.strip())\nprint([b.san(m) for m in b.legal_moves][:12])',
                 logs("['Nxe5', 'Bc4', 'Bb5', 'Qh5'..."), tin=3010, tout=84, dur=1.4),
            step(4, "final", "The fixture position's intended move is Qh5.", 'final_answer("Qh5")', logs("", "Qh5"),
                 tin=3150, tout=26, dur=0.9),
        ],
        "seconds": 14.8,
    },
    {
        "id": "fixture-q09-python",
        "q": "What is the final numeric output of the attached Python program?",
        "file": "fixture_program.py",
        "answer": "42",
        "steps": [
            step(1, "action", "Run the attached program in the sandboxed runner.",
                 'print(run_python_file("data/files/fixture-q09-python.py"))',
                 logs("exit code 0\nstdout:\nwarming up...\n13\n42"), tin=1880, tout=57, dur=2.2),
            step(2, "final", "The last printed number is 42.", 'final_answer("42")', logs("", "42"),
                 tin=1990, tout=25, dur=0.7),
        ],
        "seconds": 4.1,
    },
    {
        "id": "fixture-q10-no-answer",
        "q": "What colour is the door of the fictional Example Falls town hall in the earliest archived photo?",
        "answer": "",
        "steps": [
            step(1, "action", "Look for archived photos.", 'print(web_search("Example Falls town hall archived photo door"))',
                 logs("No results found."), tin=1900, tout=55, dur=3.0),
            step(2, "action", "Nothing indexed. Try the encyclopedia.", 'print(wikipedia_search("Example Falls town hall"))',
                 logs("No Wikipedia articles matched."), tin=2200, tout=58, dur=2.6),
        ],
        "seconds": 6.0,
    },
]


TOOLS = ["web_search", "visit_webpage", "wikipedia_search", "wikipedia_page", "wikipedia_revision",
         "wikipedia_page_info", "read_file", "run_python_file", "transcribe_audio", "image_question",
         "youtube_transcript", "youtube_audio_transcript", "youtube_frames_question", "web_agent", "final_answer"]
CALL_RE = re.compile(r"(?<![\w.])(" + "|".join(TOOLS) + r")\s*\(")


def parsed_calls(code):
    """What gaia_agent.trace records for a CodeAgent step: one entry per tool call in the code."""
    return [{"name": m.group(1), "arguments": "(fixture)"} for m in CALL_RE.finditer(code or "")]


def blank(typ, agent="manager"):
    return {"index": 0, "type": typ, "agent": agent, "model_output": None, "code": None, "tool_calls": [],
            "observation": None, "error": None, "error_type": None, "input_tokens": None,
            "output_tokens": None, "duration_s": None}


def to_trace_steps(q, agent="manager"):
    """Reshape the authored steps into gaia_agent.trace's schema: a task step first, action
    steps with step_number (the final_answer one flagged), then a synthetic final step."""
    task = blank("task", agent)
    task["model_output"] = q["q"] + (f"\n\nAttached file: data/files/{q['file']}" if q.get("file") else "")
    out = [task]
    for s in q["steps"]:
        d = dict(blank(s["type"], agent), **{k: v for k, v in s.items() if k != "index"})
        if s["type"] in ("action", "final"):
            d["type"] = "action"
            d["step_number"] = s["index"]
            d["tool_calls"] = parsed_calls(s.get("code"))
            d["is_final_answer"] = s["type"] == "final"
            if d["is_final_answer"]:
                d["action_output"] = q["answer"]
            d["sub_agents"] = s.get("sub_agents", [])
            if d["error"]:
                d["error_type"] = "AgentExecutionError" if "Code execution" in d["error"] else "AgentGenerationError"
        out.append(d)
    final = blank("final", agent)
    final["model_output"] = q["answer"] or None
    final["error"] = q.get("error")
    final["error_type"] = (q.get("error") or "").split(":")[0] or None
    out.append(final)
    for i, s in enumerate(out):
        s["index"] = i
    return out


def walk(steps):
    for s in steps:
        yield s
        for run in s.get("sub_agents") or []:
            yield from walk(run["steps"])


def totals(steps):
    by = {}
    all_steps = list(walk(steps))
    for s in all_steps:
        for c in s["tool_calls"]:
            by[c["name"]] = by.get(c["name"], 0) + 1
    calls = [s for s in all_steps if s["type"] in ("action", "planning")]
    return {
        "steps": sum(1 for s in steps if s["type"] == "action"),
        "planning_steps": sum(1 for s in steps if s["type"] == "planning"),
        "input_tokens": sum(s["input_tokens"] or 0 for s in calls),
        "output_tokens": sum(s["output_tokens"] or 0 for s in calls),
        "total_tokens": sum((s["input_tokens"] or 0) + (s["output_tokens"] or 0) for s in calls),
        "model_calls": len(calls),
        "tool_calls": sum(by.values()),
        "tool_calls_by_name": dict(sorted(by.items(), key=lambda kv: -kv[1])),
        "step_errors": sum(1 for s in all_steps if s["error"]),
        "sub_agent_calls": sum(len(s.get("sub_agents") or []) for s in all_steps),
        "token_usage_complete": True,
    }


def clip(s, n=4000):
    if s is None or len(s) <= n:
        return s
    return s[:n] + f"…[truncated {len(s) - n} chars]"


# --------------------------------------------------------------------------- practice set (FAKE)
# The real practice set lives in evals/practice/; these are invented stand-ins with the same
# shape, so the report's practice section and full trajectory viewer can be built and tested.

PRACTICE = [
    {"q": "A fictional bakery in Example Falls sells 37 loaves a day for 12 days. How many loaves does it sell?",
     "gold": "444", "wrong": "434", "format": "number", "file": None,
     "script": [("Multiply in code rather than in my head.", "print(37 * 12)", "444")]},
    {"q": "The attached CSV lists invented parcel weights. Which parcel id is the heaviest?",
     "gold": "PX-17", "wrong": "PX-71", "format": "string", "file": "parcels.csv",
     "script": [("Read the file first.", 'print(read_file("parcels.csv"))',
                 "| id | kg |\n| PX-03 | 2.1 |\n| PX-17 | 9.8 |\n| PX-71 | 9.6 |"),
                ("Take the argmax in pandas.", 'import pandas as pd\ndf = pd.read_csv("parcels.csv")\n'
                 'print(df.loc[df["kg"].idxmax(), "id"])', "PX-17")]},
    {"q": "On the invented docs site docs.example-practice.test, which port does the demo server listen on?",
     "gold": "8642", "wrong": "8080", "format": "number", "file": None,
     "script": [("Search for the docs page.", 'print(web_search("example-practice demo server port"))',
                 "1. Demo server — docs.example-practice.test/demo"),
                ("Open it and find the port.", 'print(visit_webpage("https://docs.example-practice.test/demo"))',
                 "The demo server listens on port 8642 by default.")]},
    {"q": "According to the invented encyclopedia entry 'Lake Placeholder', how deep is the lake in metres?",
     "gold": "31", "wrong": "13", "format": "number", "file": None,
     "script": [("Look the article up.", 'print(wikipedia_page("Lake Placeholder"))',
                 "Lake Placeholder is a fictional lake. Max depth: 31 m.")]},
    {"q": "How many talks are listed on the invented PracticeConf 2026 schedule page?",
     "gold": "14", "wrong": "12", "format": "number", "file": None,
     "script": [("Find the schedule.", 'print(web_search("PracticeConf 2026 schedule"))',
                 "1. PracticeConf 2026 — Schedule"),
                ("Count the talk rows in code.", 'page = visit_webpage("https://practiceconf.example.test/2026")\n'
                 'print(sum(1 for l in page.splitlines() if l.startswith("| Talk")))', "14")]},
    {"q": "The attached spreadsheet of invented invoices: what is the total of the unpaid invoices in USD?",
     "gold": "905.25", "wrong": "1205.25", "format": "number", "file": "invoices.xlsx",
     "script": [("Inspect the workbook.", 'print(read_file("invoices.xlsx"))',
                 "Sheet 'Invoices' (9 rows)\n| no | total | paid |\n| 1 | 300.00 | yes |\n| 2 | 410.25 | no |"),
                ("Sum the unpaid totals.", 'df = pd.read_excel("invoices.xlsx")\nprint(df[df.paid == "no"].total.sum())',
                 "905.25")]},
    {"q": "As of 2021, how many members did the invented 'Example Choir' list on its encyclopedia page?",
     "gold": "23", "wrong": "27", "format": "number", "file": None,
     "script": [("An 'as of' question needs the old revision.",
                 'print(wikipedia_revision("Example Choir", as_of="2021-12-31")[:1500])',
                 "Revision live on 2021-12-31: the choir lists 23 members.")]},
    {"q": "The attached invented voicemail names a meeting room. Which room?",
     "gold": "Room Cobalt", "wrong": "Room Copper", "format": "string", "file": "voicemail.mp3",
     "script": [("Transcribe locally.", 'print(transcribe_audio("voicemail.mp3"))',
                 "Hi, we moved the review to Room Cobalt, not Copper, see you at three.")]},
    {"q": "In which year did the invented 'Practice Bridge' open?",
     "gold": "1957", "wrong": "1975", "format": "number", "file": None,
     "script": [("Search for the bridge.", 'print(web_search("Practice Bridge opened year"))',
                 "Practice Bridge (fictional), opened 1957.")]},
    {"q": "The attached invented bar chart shows visits per weekday. Which day has the tallest bar?",
     "gold": "Tuesday", "wrong": "Thursday", "format": "string", "file": "visits.png",
     "script": [("Ask the vision model for every bar's value, then pick the max in code.",
                 'print(image_question("visits.png", "List each weekday and its bar value."))',
                 "Mon 41, Tue 58, Wed 39, Thu 55, Fri 47")]},
    {"q": "How many top-level sections does the invented encyclopedia article 'Example Ridge' have?",
     "gold": "9", "wrong": "11", "format": "number", "file": None,
     "script": [("Read the section list.", 'print(wikipedia_page_info("Example Ridge"))',
                 "Sections: 1 Geography 2 History 2.1 Early 3 Trails 4 Flora 5 Fauna 6 Access 7 Weather 8 See also "
                 "9 References")]},
    {"q": "In the invented practice video, how many red balloons are visible at once at most?",
     "gold": "5", "wrong": "4", "format": "number", "file": None,
     "script": [("Sample frames and count.",
                 'print(youtube_frames_question("https://www.youtube.com/watch?v=PRACTICE01", "How many red balloons?"))',
                 "frame 00:03 → 2\nframe 00:09 → 5\nframe 00:15 → 4")]},
]


def practice_trace(rid, cfg, it, task, arch):
    """A fake trace consistent with the eval item: same step count, tokens, seconds, answer."""
    n = max(2, int(it.get("steps") or 2))
    tin, tout, secs = it.get("input_tokens") or 0, it.get("output_tokens") or 0, it.get("seconds") or 0.0
    w = list(range(1, n + 1))
    W = sum(w)
    script = list(task["script"])
    while len(script) < n - 1:
        script.append(("Double-check the intermediate result before answering.", "print(result)",
                       "(re-checked; unchanged)"))
    steps = [dict(blank("task", "agent"),
                  model_output=task["q"] + (f"\n\nAttached file: runs/practice_files/{it['id']}/{task['file']}"
                                            if task["file"] else ""))]
    wrong = not it.get("correct")
    for i in range(1, n):
        thought, code, out = script[i - 1]
        st = dict(blank("action", "agent"), step_number=i, model_output=code_out(thought, code), code=code,
                  tool_calls=parsed_calls(code), observation=logs(out), sub_agents=[], is_final_answer=False,
                  input_tokens=int(tin * w[i - 1] / W), output_tokens=int(tout / n),
                  duration_s=round(secs * w[i - 1] / W, 1))
        if wrong and i == 2:
            st["observation"] = None
            st["error"] = "Code execution failed at line 'print(result)': NameError: name 'result' is not defined"
            st["error_type"] = "AgentExecutionError"
        steps.append(st)
    ans = it["answer"] or ""
    fa = f'final_answer("{ans}")'
    steps.append(dict(blank("action", "agent"), step_number=n, model_output=code_out("Answer as asked.", fa), code=fa,
                      tool_calls=parsed_calls(fa), observation=logs("", ans), sub_agents=[], is_final_answer=True,
                      action_output=ans, input_tokens=int(tin * w[-1] / W), output_tokens=int(tout / n),
                      duration_s=round(secs * w[-1] / W, 1)))
    final = blank("final", "agent")
    final["model_output"] = ans or None
    steps.append(final)
    for i, s in enumerate(steps):
        s["index"] = i
    return {"schema_version": 1, "task_id": it["id"], "question": task["q"], "run_id": rid,
            "model": cfg["model"], "provider": PROVIDER, "arch": arch, "started_at": "2026-10-10T16:00:00Z",
            "seconds": secs, "final_answer": ans or None, "error": None, "steps": steps, "totals": totals(steps)}


def write_practice(all_results, runs):
    """Practice tasks.jsonl, practice traces for two runs, and a fake runs/usage.jsonl."""
    pdir = ROOT / "evals" / "practice"
    pdir.mkdir(parents=True, exist_ok=True)
    skills = [["arithmetic"], ["file"], ["web"], ["wikipedia"], ["web", "counting"], ["file", "arithmetic"],
              ["wikipedia", "as_of"], ["audio"], ["web"], ["vision"], ["wikipedia", "counting"], ["video"]]
    with open(pdir / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for j, t in enumerate(PRACTICE):
            fh.write(json.dumps({"id": f"practice-{j + 1:02d}", "question": t["q"],
                                 "file": f"evals/practice/files/{t['file']}" if t["file"] else None,
                                 "gold": t["gold"], "format": t["format"], "skills": skills[j],
                                 "source": {"type": "fixture", "derivation": "FIXTURE: invented"}}) + "\n")
    traced = {"fixture-eval-baseline-1": 12, "fixture-eval-multi-1": 3}
    for rid, k in traced.items():
        res = all_results[rid]
        tdir = runs / "traces" / rid
        tdir.mkdir(parents=True, exist_ok=True)
        for old_file in tdir.glob("*.json"):
            old_file.unlink()
        for j, it in enumerate(res["items"][:k]):
            if it.get("skipped"):
                continue
            tr = practice_trace(rid, res["config"], it, PRACTICE[j], res["config"]["arch"])
            (tdir / f"{it['id']}.json").write_text(json.dumps(tr, indent=2, ensure_ascii=False), encoding="utf-8")

    # One usage row per model call: the GAIA fixture run and the traced practice runs.
    rows = []
    t = 0
    for rid in [RUN] + list(traced):
        tdir = runs / "traces" / rid
        for f in sorted(tdir.glob("*.json")):
            tr = json.loads(f.read_text(encoding="utf-8"))
            for st in walk(tr["steps"]):
                if st["type"] not in ("action", "planning"):
                    continue
                t += 1
                vision = any(c["name"] in ("image_question", "youtube_frames_question") for c in st["tool_calls"])
                fb = 1 if t % 9 == 0 else 0
                errors = [{"model": "fixture/fake-model-8b:free", "kind": "rate_limit", "status": 429}] * (1 if t % 4 == 0 else 0)
                if fb:
                    errors = errors + [{"model": "fixture/fake-model-8b:free", "kind": "upstream_rate_limit", "status": 429}]
                failed = bool(st.get("error")) and "RateLimit" in str(st.get("error"))
                model = ("fixture/fake-vision-3b:free" if vision else
                         "fixture/fake-fallback-4b:free" if fb else "fixture/fake-model-8b:free")
                rows.append({"ts": f"2026-10-10T{14 + t // 60:02d}:{t % 60:02d}:00+00:00", "run_id": rid,
                             "kind": "vision" if vision else "text", "preset": "fixture",
                             "requested_model": "fixture/fake-model-8b:free", "latency_s": st.get("duration_s") or 1.0,
                             "waited_limiter_s": 0.5 if t % 3 == 0 else 0.0, "waited_backoff_s": 2.0 if errors else 0.0,
                             "attempts": 1 + len(errors), "retries": len(errors) - fb, "errors": errors,
                             "status": "error" if failed else "ok", "model": None if failed else model,
                             "fallback_index": fb, "input_tokens": st.get("input_tokens") or 0,
                             "output_tokens": st.get("output_tokens") or 0})
    (runs / "usage.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def main():
    (ROOT / "FIXTURE").write_text(
        "This directory is FAKE sample data for developing and testing report/build_report.py.\n"
        "It is not a GAIA run and must never be presented as results.\n",
        encoding="utf-8",
    )
    runs = ROOT / "runs"
    tdir = runs / "traces" / RUN
    tdir.mkdir(parents=True, exist_ok=True)
    lines = []
    t0 = 1791900000  # fixed epoch so output is deterministic
    for k, q in enumerate(QUESTIONS):
        for s in q["steps"]:
            s["observation"] = clip(s["observation"])
        steps = to_trace_steps(q)
        tr = {
            "schema_version": 1,
            "task_id": q["id"],
            "question": q["q"],
            "run_id": RUN,
            "model": MODEL,
            "provider": PROVIDER,
            "arch": "multi",
            "started_at": f"2026-10-10T14:{k * 3:02d}:00Z",
            "seconds": q["seconds"],
            "final_answer": q["answer"] or None,
            "error": q.get("error"),
            "steps": steps,
            "totals": totals(steps),
        }
        (tdir / f"{q['id']}.json").write_text(json.dumps(tr, indent=2, ensure_ascii=False), encoding="utf-8")
        lines.append(
            {
                "task_id": q["id"],
                "question": q["q"],
                "answer": q["answer"],
                "raw": q["answer"] or None,
                "steps": tr["totals"]["steps"],
                "error": q.get("error"),
                "seconds": q["seconds"],
                "provider_info": {"provider": PROVIDER, "model": MODEL, "arch": "multi"},
                "run_id": RUN,
                **({"file_name": q["file"]} if q.get("file") else {}),
            }
        )
    # An older run's line for q01 precedes the current run's: the report must pick the latest run.
    old = dict(lines[0], answer="5", run_id="fixture-run-001", seconds=22.0)
    (runs / "answers.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in [old] + lines) + "\n", encoding="utf-8"
    )
    subs = [
        {"submitted_at": "2026-10-09T21:04:00Z", "run_id": "fixture-run-001",
         "response": {"score": 20.0, "correct_count": 4, "total_attempted": 10,
                      "message": "FIXTURE: fake submission response (4/20 correct)."}},
        {"submitted_at": "2026-10-10T15:02:00Z", "run_id": RUN,
         "response": {"score": 30.0, "correct_count": 6, "total_attempted": 10,
                      "message": "FIXTURE: fake submission response (6/20 correct)."}},
    ]
    (runs / "submissions.jsonl").write_text("\n".join(json.dumps(s) for s in subs) + "\n", encoding="utf-8")
    ea = {
        "run_id": RUN,
        "generated_at": "2026-10-10T15:00:00+00:00",
        "n_tasks": len(QUESTIONS),
        "max_steps": 12,
        "items": [
            {"task_id": "fixture-q06-rate-limit", "tags": ["rate_limited"],
             "evidence": "rate_limited: step 2: RateLimitError: 429 Too Many Requests"},
            {"task_id": "fixture-q07-timeout", "tags": ["timeout", "paging_loop"],
             "evidence": "timeout: TimeoutError: question exceeded the 300 s budget; paging_loop: visit_webpage x10 on "
                         "press.example-falls.test/annual-reports"},
            {"task_id": "fixture-q10-no-answer", "tags": ["empty_or_refusal_answer"],
             "evidence": "empty_or_refusal_answer: two searches returned nothing; no final answer"},
            {"task_id": "fixture-q08-chess", "tags": ["code_execution_error", "vision_used"],
             "evidence": "code_execution_error: step 2: ValueError: expected 'w' or 'b' for turn part of fen; "
                         "vision_used: image_question"},
            {"task_id": "fixture-q04-video-birds", "tags": ["vision_used", "tool_error:youtube_transcript"],
             "evidence": "vision_used: youtube_frames_question"},
            {"task_id": "fixture-q01-library", "tags": [], "evidence": ""},
        ],
        "tag_descriptions": {
            "rate_limited": "A model call hit a 429 / quota",
            "timeout": "Wall-clock or model timeout",
            "paging_loop": "Same tool on the same target 4+ times (paging/re-reading)",
            "empty_or_refusal_answer": "Empty, refusal or 'unknown' answer",
            "code_execution_error": "The agent's own Python code failed",
            "vision_used": "Used the vision model",
        },
        "clean_tasks": ["fixture-q01-library", "fixture-q02-spreadsheet", "fixture-q03-voice-memo",
                        "fixture-q05-wiki-revision", "fixture-q09-python"],
        "note": "FIXTURE: heuristic behaviour tags from traces; no gold answers are used.",
    }
    ea["counts"] = {}
    for it in ea["items"]:
        for t in it["tags"]:
            ea["counts"][t] = ea["counts"].get(t, 0) + 1
    (runs / "error_analysis.json").write_text(json.dumps(ea, indent=2), encoding="utf-8")

    # Evals: a 12-item practice set (fake), repeated runs per config for pass@k / pass^k.
    edir = ROOT / "evals" / "results"
    edir.mkdir(parents=True, exist_ok=True)
    for old_file in edir.glob("*.json"):
        old_file.unlink()
    difficulty = [0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1]
    skills = [["arithmetic"], ["file"], ["web"], ["wikipedia"], ["web", "counting"], ["file", "arithmetic"],
              ["wikipedia", "as_of"], ["audio"], ["web"], ["vision"], ["wikipedia", "counting"], ["video"]]
    ablation_rows = []
    all_results = {}
    configs = [
        ("baseline", {"disabled_tools": [], "arch": "single", "verify": False}, 0.0, 3),
        ("no-web", {"disabled_tools": ["web_search", "visit_webpage"], "arch": "single", "verify": False}, -0.3, 1),
        ("no-wiki", {"disabled_tools": ["wikipedia_search", "wikipedia_page", "wikipedia_revision"],
                     "arch": "single", "verify": False}, -0.12, 1),
        ("multi", {"disabled_tools": [], "arch": "multi", "verify": False}, 0.05, 2),
        ("verify", {"disabled_tools": [], "arch": "single", "verify": True}, 0.08, 2),
    ]
    for name, cfg, shift, reps in configs:
        for r in range(reps):
            items = []
            for j, p in enumerate(difficulty):
                ok = rng.random() < max(0.02, min(0.98, p + shift))
                tin, tout = rng.randint(6000, 30000), rng.randint(300, 1500)
                if cfg["arch"] == "multi":
                    tin = int(tin * 1.6)
                if cfg["verify"]:
                    tin = int(tin * 1.2)
                strict = ok and rng.random() > 0.15  # GAIA-normalized match is a little more lenient
                gold = PRACTICE[j]["gold"]
                items.append({
                    "id": f"practice-{j + 1:02d}",
                    "answer": (gold if strict else gold + ".") if ok else PRACTICE[j]["wrong"], "gold": gold,
                    "format": "number" if j % 2 else "string", "skills": skills[j],
                    "correct": ok, "correct_strict": strict, "seconds": round(rng.uniform(5, 70), 1),
                    "steps": rng.randint(2, 9), "input_tokens": tin, "output_tokens": tout, "error": None,
                })
            if name == "no-wiki":  # a budget stop: the harness marks the rest skipped
                items[-1] = {"id": "practice-12", "gold": PRACTICE[11]["gold"], "skills": skills[11], "answer": None,
                             "correct": None, "correct_strict": None, "skipped": True,
                             "error": "request budget spent (FIXTURE)"}
            done = [i for i in items if not i.get("skipped")]
            n = len(done)
            c = sum(bool(i["correct"]) for i in done)
            rid = f"fixture-eval-{name}-{r + 1}"
            res = {
                "run_id": rid,
                "config": {"model": MODEL, **cfg},
                "set": "practice",
                "items": items,
                "summary": {"n": n, "correct": c, "accuracy": c / n,
                            "mean_seconds": sum(i["seconds"] for i in done) / n,
                            "mean_steps": sum(i["steps"] for i in done) / n,
                            "total_tokens": sum(i["input_tokens"] + i["output_tokens"] for i in done)},
            }
            (edir / f"{rid}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            all_results[rid] = res
            ablation_rows.append((name.replace("-", "_"), rid))
    # The ablation report evals/ablate.py writes next to the runs (not a run itself).
    names = {}
    for nm, rid in ablation_rows:
        names.setdefault(nm, []).append(rid)
    fake_p = {"no_web": 0.008, "no_wiki": 0.07, "multi": 1.0, "verify": 0.29}
    report = {"tag": "fixture", "created": "2026-10-10T16:00:00+00:00", "set": "practice",
              "configs": [{"name": nm, "run_ids": rids, "status": "ok",
                           **({"vs_baseline": {"shared_tasks": 12, "mcnemar_p": fake_p[nm]}} if nm in fake_p else {})}
                          for nm, rids in names.items()]}
    (edir / "ablation_fixture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (edir / "passk_fixture-eval-baseline-1_gaia.json").write_text(
        json.dumps({"runs": names["baseline"], "metric": "gaia", "note": "FIXTURE: not a run doc"}, indent=2),
        encoding="utf-8")
    write_practice(all_results, runs)
    print("fixture written to", ROOT)


if __name__ == "__main__":
    main()
