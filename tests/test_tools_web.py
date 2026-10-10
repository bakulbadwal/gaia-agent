"""Tests for gaia_agent.tools_web.

Offline tests cover the answer-leak guard, HTML/table conversion and paging.
Network tests (marked `network`) hit DuckDuckGo/Wikipedia with generic queries only.
"""

import re

import pytest
from bs4 import BeautifulSoup

from gaia_agent import tools_web as tw
from gaia_agent.tools_web import (
    BLOCK_NOTE,
    TOOLS_WEB,
    html_to_markdown,
    is_blocked,
    table_to_markdown,
    visit_webpage,
    web_search,
    wikipedia_page,
    wikipedia_page_info,
    wikipedia_revision,
    wikipedia_search,
)

# ---------------------------------------------------------------- guard (offline)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "https://huggingface.co/datasets/someone/some-dump",
        "https://www.huggingface.co/datasets/x/y/resolve/main/data.json",
        "https://huggingface.co/spaces/agents-course/Final_Assignment_Template",
        "https://huggingface.co/spaces/random-user/my-unit4-agent",
        "https://agents-course-unit4-scoring.hf.space/questions",
        "https://github.com/someone/GAIA-solutions",
        "https://example.com/page?q=Gaia%20answers",
        "https://example.com/%67%61%69%61",  # url-encoded "gaia"
        "https://hf.co/datasets/foo/bar",
    ],
)
def test_guard_blocks_answer_sources(url):
    assert is_blocked(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://en.wikipedia.org/wiki/Marie_Curie",
        "https://www.python.org/downloads/",
        "https://huggingface.co/docs/transformers/index",
        "https://arxiv.org/abs/1706.03762",
        "",
        None,
    ],
)
def test_guard_allows_normal_sources(url):
    assert not is_blocked(url)


def test_block_note_explains_reason():
    assert "gaia" in BLOCK_NOTE.lower()
    assert "huggingface.co/datasets" in BLOCK_NOTE


def test_web_search_refuses_gaia_query_offline():
    out = web_search("GAIA benchmark validation answers")
    assert out.startswith("ERROR")
    assert "BLOCKED" in out


def test_visit_webpage_refuses_blocked_urls_offline():
    for url in [
        "https://huggingface.co/datasets/gaia-benchmark/GAIA/tree/main",
        "huggingface.co/spaces/agents-course/Unit4",
        "https://example.com/gaia-leak.json",
    ]:
        out = visit_webpage(url)
        assert out.startswith("ERROR") and "BLOCKED" in out, url


def test_web_search_filters_results_mentioning_gaia(monkeypatch):
    fake = [
        {"title": "Good result", "href": "https://example.com/a", "body": "useful facts"},
        {"title": "GAIA validation answers", "href": "https://example.com/b", "body": "x"},
        {"title": "Dump", "href": "https://huggingface.co/datasets/x/y", "body": "x"},
        {"title": "Another", "href": "https://example.org/c", "body": "the Gaia benchmark says"},
        {"title": "Second good", "href": "https://example.net/d", "body": "more facts"},
    ]
    monkeypatch.setattr(tw, "_ddgs_search", lambda q, n: fake)
    out = web_search("some neutral query", max_results=5)
    assert "https://example.com/a" in out and "https://example.net/d" in out
    assert "example.com/b" not in out and "huggingface.co" not in out and "example.org/c" not in out
    assert "3 result(s) removed" in out
    assert "gaia" not in out.lower().replace("answer-leak", "")  # no leaked text


def test_web_search_never_raises(monkeypatch):
    def boom(q, n):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(tw, "_ddgs_search", boom)
    out = web_search("anything")
    assert out.startswith("ERROR") and "rate limited" in out


# ---------------------------------------------------------------- conversion (offline)


def test_table_to_markdown_expands_spans():
    html = """
    <table class="wikitable"><caption>Results</caption>
      <tr><th>Year</th><th>Team</th><th>Score</th></tr>
      <tr><td rowspan="2">2001</td><td>A|B</td><td>3</td></tr>
      <tr><td>C</td><td>4</td></tr>
      <tr><td colspan="2">Total</td><td>7</td></tr>
    </table>"""
    table = BeautifulSoup(html, "lxml").find("table")
    md = table_to_markdown(table)
    lines = md.splitlines()
    assert lines[0] == "Table: Results"
    assert lines[1] == "| Year | Team | Score |"
    assert lines[2] == "|---|---|---|"
    assert "| 2001 | A\\|B | 3 |" in lines
    assert "| 2001 | C | 4 |" in lines  # rowspan repeated
    assert "| Total | Total | 7 |" in lines  # partial colspan repeated


def test_full_width_span_not_repeated():
    html = "<table><tr><th colspan='2'>Infobox title</th></tr><tr><th>Born</th><td>1900</td></tr></table>"
    md = table_to_markdown(BeautifulSoup(html, "lxml").find("table"))
    assert md.splitlines()[0] == "| Infobox title |  |"


def test_html_to_markdown_keeps_tables_and_drops_wiki_noise():
    html = """<div class="mw-parser-output">
      <h2>History<span class="mw-editsection">[edit]</span></h2>
      <p>Some text<sup class="reference">[1]</sup> here with <a href="/wiki/X">a link</a>.</p>
      <table class="wikitable"><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
      <div class="navbox">huge nav box</div>
    </div>"""
    md = html_to_markdown(html, base_url="https://en.wikipedia.org/wiki/T", keep_links=False, wiki=True)
    assert "## History" in md
    assert "[edit]" not in md and "[1]" not in md and "navbox" not in md and "huge nav" not in md
    assert "| A | B |" in md and "| 1 | 2 |" in md
    assert "a link" in md and "/wiki/X" not in md


def test_html_to_markdown_absolutises_links():
    md = html_to_markdown('<p><a href="/about">About</a></p>', base_url="https://example.com/x/", keep_links=True)
    assert "https://example.com/about" in md


def test_page_text_paging_note():
    text = "abcdefghij" * 100
    out = tw._page_text(text, 0, 100, "visit_webpage(url='u', start={start})")
    assert "TRUNCATED" in out and "start=100" in out
    out2 = tw._page_text(text, 950, 100, "f(start={start})")
    assert "TRUNCATED" not in out2 and out2.endswith("abcdefghij")
    assert tw._page_text(text, 5000, 100, "f(start={start})").startswith("ERROR")


def test_as_of_timestamp():
    assert tw._as_of_timestamp("2022-03-05") == "2022-03-05T23:59:59Z"
    assert tw._as_of_timestamp("2023-02") == "2023-02-28T23:59:59Z"
    assert tw._as_of_timestamp("2020") == "2020-12-31T23:59:59Z"
    with pytest.raises(ValueError):
        tw._as_of_timestamp("last year")


def test_tools_exported_with_schemas():
    names = [t.name for t in TOOLS_WEB]
    assert names == [
        "web_search",
        "visit_webpage",
        "wikipedia_search",
        "wikipedia_page",
        "wikipedia_revision",
        "wikipedia_page_info",
    ]
    for t in TOOLS_WEB:
        assert t.output_type == "string"
        assert t.description


def test_bad_inputs_return_error_strings():
    assert visit_webpage("").startswith("ERROR")
    assert web_search("").startswith("ERROR")
    assert wikipedia_page("").startswith("ERROR")
    assert wikipedia_revision("Marie Curie", "not a date").startswith("ERROR")


# ---------------------------------------------------------------- network


@pytest.mark.network
def test_network_web_search_returns_results():
    out = web_search("Python programming language official website", max_results=5)
    assert not out.startswith("ERROR"), out
    assert out.count("URL: http") >= 2


@pytest.mark.network
def test_network_wikipedia_search():
    out = wikipedia_search("Marie Curie")
    assert "Marie Curie" in out and "en.wikipedia.org/wiki/" in out


@pytest.mark.network
def test_network_wikipedia_page_renders_markdown_table():
    out = wikipedia_page("List of countries and dependencies by population (United Nations)", max_chars=20000)
    assert not out.startswith("ERROR"), out[:300]
    assert "Sections:" in out
    # a markdown table: header row, separator row, data rows
    assert re.search(r"^\|.+\|\n\|(---\|)+\n\|.+\|$", out, re.M), out[:2000]
    assert "| India |" in out or "| China |" in out


@pytest.mark.network
def test_network_wikipedia_page_section_and_paging():
    full = wikipedia_page("Marie Curie", max_chars=2000)
    assert "TRUNCATED" in full and "start=2000" in full
    sec = wikipedia_page("Marie Curie", section="Legacy", max_chars=3000)
    assert not sec.startswith("ERROR"), sec[:300]
    assert "Section:" in sec
    bad = wikipedia_page("Marie Curie", section="Zzzz no such section")
    assert bad.startswith("ERROR") and "Available sections" in bad


@pytest.mark.network
def test_network_wikipedia_revision_is_on_or_before_date():
    out = wikipedia_revision("Marie Curie", "2019-06-15", max_chars=3000)
    assert not out.startswith("ERROR"), out[:300]
    m = re.search(r"saved (\d{4}-\d{2}-\d{2})T", out)
    assert m, out[:500]
    assert m.group(1) <= "2019-06-15"
    assert "oldid=" in out and "Born" in out


@pytest.mark.network
def test_network_wikipedia_revision_before_creation():
    out = wikipedia_revision("ChatGPT", "2015-01-01")
    assert out.startswith("ERROR") and "did not exist" in out


@pytest.mark.network
def test_network_wikipedia_page_info():
    out = wikipedia_page_info("Marie Curie")
    assert "Talk page: https://en.wikipedia.org/wiki/Talk:Marie_Curie" in out
    assert re.search(r"First revision \(page created\): \d{4}-\d{2}-\d{2}", out)
    assert "Quality status:" in out


@pytest.mark.network
def test_network_wikipedia_missing_page():
    out = wikipedia_page("Qwxzv nonexistent article title 98765")
    assert out.startswith("ERROR") and "not found" in out


@pytest.mark.network
def test_network_visit_webpage_html_and_wikipedia_url():
    out = visit_webpage("https://www.python.org/", max_chars=3000)
    assert not out.startswith("ERROR"), out[:300]
    assert "Python" in out
    wiki = visit_webpage("https://en.wikipedia.org/wiki/Marie_Curie", max_chars=2000)
    assert "Wikipedia via API" in wiki and "| Born |" in wiki


@pytest.mark.network
def test_network_visit_webpage_pdf():
    out = visit_webpage("https://arxiv.org/pdf/1706.03762", max_chars=2000)
    assert out.startswith("[PDF,"), out[:300]
    assert "Attention Is All You Need" in out
