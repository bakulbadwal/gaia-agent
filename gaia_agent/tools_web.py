"""Web tools for the GAIA agent: web search, page visiting and Wikipedia access.

Every tool returns a string and never raises: failures come back as text that
starts with "ERROR:". Long output is truncated with a note that says how to page.

An answer-leak guard (``is_blocked``) keeps the agent away from the GAIA dataset,
Hugging Face datasets/Spaces (where answer dumps and other students' hard-coded
answers live) and anything that mentions GAIA by name.
"""

import re
import time
from collections import OrderedDict
from io import BytesIO
from urllib.parse import quote, unquote, urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify
from smolagents import tool

USER_AGENT = (
    "gaia-agent/0.1 (HF Agents Course Unit 4 student project; "
    "https://huggingface.co/learn/agents-course) python-requests"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
TIMEOUT = 25
DEFAULT_MAX_CHARS = 15000
HARD_MAX_CHARS = 60000

# ---------------------------------------------------------------------------
# Answer-leak guard
# ---------------------------------------------------------------------------

BLOCK_NOTE = (
    "BLOCKED by the answer-leak guard: this agent must solve each question itself, so it may not "
    "read the GAIA benchmark, its answer keys, or copies of them. Any URL, query or search result "
    "that mentions 'gaia', and any Hugging Face dataset or Space page (huggingface.co/datasets/, "
    "huggingface.co/spaces/, *.hf.space, including agents-course), is refused. Find the facts "
    "from primary sources instead (Wikipedia, official sites, papers, news)."
)

_BLOCKED_SUBSTRINGS = (
    "gaia",
    "huggingface.co/datasets/",
    "huggingface.co/spaces/",
    "hf.co/datasets/",
    "hf.co/spaces/",
    ".hf.space",
    "agents-course",
)


def is_blocked(url_or_text) -> bool:
    """Return True if a URL or a piece of text must be refused by the answer-leak guard.

    Blocks anything mentioning "gaia" (case-insensitive), any Hugging Face dataset or Space
    URL (including agents-course), and *.hf.space hosts. URL-encoded text is decoded first.
    """
    if not url_or_text:
        return False
    text = str(url_or_text)
    candidates = {text.lower()}
    try:
        candidates.add(unquote(unquote(text)).lower())
    except Exception:
        pass
    for cand in candidates:
        squashed = cand.replace("www.", "")
        for bad in _BLOCKED_SUBSTRINGS:
            if bad in cand or bad in squashed:
                return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session = requests.Session()
_session.headers.update(HEADERS)


def _http_get(url: str, params: dict | None = None, retries: int = 3, **kwargs) -> requests.Response:
    """GET with retry/backoff on 429 and 5xx. Raises on final failure."""
    last_exc = None
    timeout = kwargs.pop("timeout", TIMEOUT)
    for attempt in range(retries):
        try:
            resp = _session.get(url, params=params, timeout=timeout, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = requests.HTTPError(f"HTTP {resp.status_code} for {resp.url}")
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 * (2**attempt)
                time.sleep(min(wait, 10))
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            msg = str(exc)
            if "NameResolution" in msg or "Failed to resolve" in msg or "nodename nor servname" in msg:
                raise  # DNS failure: retrying will not help
            time.sleep(1.5 * (2**attempt))
    raise last_exc if last_exc else RuntimeError(f"GET failed: {url}")


class _LRU(OrderedDict):
    def __init__(self, maxsize: int = 64):
        super().__init__()
        self.maxsize = maxsize

    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


_cache = _LRU(64)


def _clamp_paging(start, max_chars) -> tuple[int, int]:
    try:
        start = int(start or 0)
    except (TypeError, ValueError):
        start = 0
    try:
        max_chars = int(max_chars or DEFAULT_MAX_CHARS)
    except (TypeError, ValueError):
        max_chars = DEFAULT_MAX_CHARS
    return max(0, start), max(500, min(max_chars, HARD_MAX_CHARS))


def _page_text(text: str, start: int, max_chars: int, how: str) -> str:
    """Slice text to [start, start+max_chars) and add a paging note.

    `how` is a format string with a `{start}` placeholder describing the follow-up call.
    """
    total = len(text)
    if start >= total and total > 0:
        return f"ERROR: start={start} is past the end of the content ({total} chars). Use a smaller start."
    end = min(total, start + max_chars)
    chunk = text[start:end]
    notes = []
    if start > 0:
        notes.append(f"[Showing chars {start}-{end} of {total}.]")
    if end < total:
        notes.append(
            f"[TRUNCATED: showing chars {start}-{end} of {total}. "
            f"To read more call {how.format(start=end)}]"
        )
    if notes:
        if start > 0:
            chunk = notes[0] + "\n" + chunk
        if end < total:
            chunk = chunk + "\n\n" + notes[-1]
    return chunk


# ---------------------------------------------------------------------------
# HTML -> markdown (with real markdown tables)
# ---------------------------------------------------------------------------

def _cell_text(cell: Tag) -> str:
    for br in cell.find_all("br"):
        br.replace_with(" / ")
    text = cell.get_text(" ", strip=True)
    if not text:
        alts = [img.get("alt", "").strip() for img in cell.find_all("img") if img.get("alt")]
        text = " ".join(a for a in alts if a)
    text = re.sub(r"\s+", " ", text).replace("|", "\\|")
    return text.strip()


def _span(cell: Tag, attr: str) -> int:
    try:
        return max(1, min(int(re.sub(r"[^0-9]", "", str(cell.get(attr, "1"))) or 1), 200))
    except ValueError:
        return 1


def table_to_markdown(table: Tag) -> str:
    """Convert an HTML table to a markdown table, expanding rowspan/colspan into a grid."""
    rows = []
    for tr in table.find_all("tr"):
        # only rows that belong to this table, not to nested tables
        if tr.find_parent("table") is not table:
            continue
        rows.append(tr)
    grid: dict[tuple[int, int], str] = {}
    header_rows = set()
    spans: list[tuple[int, int, int, int]] = []
    n_cols = 0
    for r, tr in enumerate(rows):
        c = 0
        cells = [cell for cell in tr.find_all(["td", "th"]) if cell.find_parent("tr") is tr]
        if cells and all(cell.name == "th" for cell in cells):
            header_rows.add(r)
        for cell in cells:
            while (r, c) in grid:
                c += 1
            text = _cell_text(cell)
            rs, cs = _span(cell, "rowspan"), _span(cell, "colspan")
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = text
            spans.append((r, c, rs, cs))
            c += cs
        n_cols = max(n_cols, c)
    if not grid or n_cols == 0:
        return ""
    # a cell spanning the full table width (e.g. an infobox title/image row) is shown once, not repeated
    for r0, c0, rs, cs in spans:
        if c0 == 0 and cs >= n_cols > 1:
            for dr in range(rs):
                for dc in range(1, cs):
                    grid[(r0 + dr, dc)] = ""
    n_rows = max(r for r, _ in grid) + 1
    matrix = [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]
    matrix = [row for row in matrix if any(cell for cell in row)]
    if not matrix:
        return ""
    caption = table.find("caption")
    lines = []
    if caption and caption.find_parent("table") is table:
        lines.append(f"Table: {caption.get_text(' ', strip=True)}")
    header = matrix[0]
    body = matrix[1:]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * n_cols) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_WIKI_NOISE_SELECTORS = [
    ".mw-editsection",
    "sup.reference",
    ".reference",
    ".mw-cite-backlink",
    ".navbox",
    ".navbox-styles",
    ".vertical-navbox",
    ".sistersitebox",
    ".noprint",
    ".mw-empty-elt",
    ".shortdescription",
    ".hatnote",
    "#toc",
    ".toc",
    ".mw-jump-link",
    ".catlinks",
    "style",
    "link",
    "script",
    "noscript",
    "meta",
]

_GENERIC_NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe", "template", "nav", "footer", "form", "button", "link", "meta"]


def html_to_markdown(html, base_url: str = "", keep_links: bool = True, wiki: bool = False) -> str:
    """Convert HTML to markdown; tables become markdown tables (rowspan/colspan expanded)."""
    soup = BeautifulSoup(html, "lxml")
    if wiki:
        for sel in _WIKI_NOISE_SELECTORS:
            for el in soup.select(sel):
                el.decompose()
        # keep hatnotes out but keep their info? (dropped above as noise)
    else:
        for name in _GENERIC_NOISE_TAGS:
            for el in soup.find_all(name):
                el.decompose()
        main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article")
        if main is not None and len(main.get_text(" ", strip=True)) > 800:
            soup = BeautifulSoup(str(main), "lxml")
    # absolutise links
    if keep_links and base_url:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("javascript:", "#")):
                del a["href"]
            else:
                a["href"] = urljoin(base_url, href)
    # tables -> placeholders (outermost tables only; nested content is flattened into cells)
    tables = {}
    for i, table in enumerate([t for t in soup.find_all("table") if t.find_parent("table") is None]):
        md = table_to_markdown(table)
        key = f"XXTABLEPLACEHOLDER{i}XX"
        tables[key] = md
        p = soup.new_tag("p")
        p.string = key
        table.replace_with(p)
    strip = ["a", "img"] if not keep_links else ["img"]
    md = markdownify(str(soup), heading_style="ATX", strip=strip, escape_underscores=False, escape_asterisks=False)
    for key, table_md in tables.items():
        md = md.replace(key, "\n" + table_md + "\n" if table_md else "")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ---------------------------------------------------------------------------
# Wikipedia (MediaWiki API) helpers
# ---------------------------------------------------------------------------

def _api_url(lang: str = "en") -> str:
    lang = re.sub(r"[^a-z\-]", "", (lang or "en").lower()) or "en"
    return f"https://{lang}.wikipedia.org/w/api.php"


def _wiki_api(params: dict, lang: str = "en") -> dict:
    p = {"format": "json", "formatversion": "2"}
    p.update(params)
    resp = _http_get(_api_url(lang), params=p)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"MediaWiki API error {err.get('code')}: {err.get('info')}")
    return data


def _wiki_url(title: str, lang: str = "en") -> str:
    return f"https://{lang}.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="/:()_,'-")


def _clean_title(title: str) -> str:
    title = (title or "").strip()
    m = re.match(r"https?://([a-z\-]+)\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)", title)
    if m:
        title = unquote(m.group(2))
    return title.replace("_", " ").strip()


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", BeautifulSoup(s or "", "lxml").get_text(" ")).strip()


def _search_titles(query: str, limit: int = 10, lang: str = "en") -> tuple[list[dict], dict]:
    data = _wiki_api(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "srprop": "snippet|timestamp|wordcount",
            "srinfo": "suggestion|totalhits",
        },
        lang,
    )
    return data.get("query", {}).get("search", []), data.get("query", {}).get("searchinfo", {})


def _parse_page(title: str = "", oldid: int | None = None, section_index: str | None = None, lang: str = "en") -> dict:
    key = ("parse", lang, title, oldid, section_index)
    hit = _cache.get_or_none(key)
    if hit is not None:
        return hit
    params = {
        "action": "parse",
        "prop": "text|sections|revid|displaytitle",
        "disableeditsection": "1",
        "disablelimitreport": "1",
        "disabletoc": "1",
    }
    if oldid:
        params["oldid"] = str(oldid)
    else:
        params["page"] = title
        params["redirects"] = "1"
    if section_index is not None:
        params["section"] = str(section_index)
    data = _wiki_api(params, lang)["parse"]
    _cache.put(key, data)
    return data


def _format_sections(sections: list[dict]) -> str:
    if not sections:
        return "(no sections)"
    parts = []
    for s in sections:
        num = s.get("number", "")
        line = _strip_html(s.get("line", ""))
        indent = "  " * max(0, int(s.get("toclevel", 1)) - 1)
        parts.append(f"{indent}{num} {line}")
    return "\n".join(parts)


def _find_section(sections: list[dict], wanted: str) -> dict | None:
    w = wanted.strip().lower().lstrip("#")
    if not w:
        return None
    for s in sections:  # exact number, title or anchor
        if w in (str(s.get("number", "")).lower(), _strip_html(s.get("line", "")).lower(), str(s.get("anchor", "")).lower().replace("_", " "), str(s.get("anchor", "")).lower()):
            return s
    for s in sections:  # substring fallback
        if w in _strip_html(s.get("line", "")).lower():
            return s
    import difflib

    names = {_strip_html(s.get("line", "")).lower(): s for s in sections}
    close = difflib.get_close_matches(w, list(names), n=1, cutoff=0.8)
    return names[close[0]] if close else None


def _render_wiki(title: str, section: str, start: int, max_chars: int, oldid: int | None, lang: str, header_extra: str, how: str) -> str:
    full = _parse_page(title=title, oldid=oldid, lang=lang)
    real_title = full.get("title", title)
    sections = full.get("sections", [])
    section = (section or "").strip()
    sec_label = ""
    if section:
        found = _find_section(sections, section)
        if found is None:
            return (
                f"ERROR: section '{section}' not found in '{real_title}'. Available sections:\n"
                + _format_sections(sections)
            )
        idx = str(found.get("index", ""))
        if not idx.isdigit():
            return f"ERROR: section '{section}' is transcluded from another page and cannot be fetched alone; read the whole page instead."
        data = _parse_page(title=real_title, oldid=oldid, section_index=idx, lang=lang)
        html = data["text"]
        sec_label = f"Section: {found.get('number')} {_strip_html(found.get('line', ''))}\n"
    else:
        html = full["text"]
    key = ("md", lang, real_title, oldid, section)
    md = _cache.get_or_none(key)
    if md is None:
        md = html_to_markdown(html, base_url=_wiki_url(real_title, lang), keep_links=False, wiki=True)
        _cache.put(key, md)
    header = (
        f"Title: {real_title}\nURL: {_wiki_url(real_title, lang)}\n{header_extra}{sec_label}"
        f"Sections:\n{_format_sections(sections)}\n\n---\n"
    )
    if start > 0:
        header = f"Title: {real_title} {sec_label.strip()}\n---\n"
    return header + _page_text(md, start, max_chars, how.replace("{title}", repr(real_title)).replace("{section}", repr(section)))


def _missing_page_message(title: str, lang: str = "en") -> str:
    msg = f"ERROR: Wikipedia page '{title}' not found"
    try:
        results, info = _search_titles(title, limit=5, lang=lang)
        if results:
            msg += ". Closest titles: " + "; ".join(r["title"] for r in results)
        if info.get("suggestion"):
            msg += f". Did you mean: {info['suggestion']}"
    except Exception:
        pass
    return msg + ". Use wikipedia_search to find the exact title."


def _resolve_revision(title: str, as_of: str, lang: str = "en") -> dict | None:
    """Return the latest revision (dict with revid, timestamp, user, comment, title) at/before as_of."""
    ts = _as_of_timestamp(as_of)
    data = _wiki_api(
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "redirects": "1",
            "rvlimit": "1",
            "rvdir": "older",
            "rvstart": ts,
            "rvprop": "ids|timestamp|user|comment|size",
        },
        lang,
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return {"missing": True}
    page = pages[0]
    revs = page.get("revisions") or []
    if not revs:
        return {"missing": False, "title": page.get("title"), "none_before": True}
    rev = dict(revs[0])
    rev["title"] = page.get("title")
    return rev


def _as_of_timestamp(as_of: str) -> str:
    s = (as_of or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?Z?)?", s)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if m.group(4):
            return f"{y}-{mo}-{d}T{m.group(4)}:{m.group(5)}:{m.group(6) or '00'}Z"
        return f"{y}-{mo}-{d}T23:59:59Z"
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        import calendar

        y, mo = int(m.group(1)), int(m.group(2))
        last = calendar.monthrange(y, mo)[1]
        return f"{y:04d}-{mo:02d}-{last:02d}T23:59:59Z"
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return f"{m.group(1)}-12-31T23:59:59Z"
    raise ValueError(f"as_of must look like YYYY-MM-DD (or YYYY-MM / YYYY); got {as_of!r}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_SEARCH_BACKENDS = ["duckduckgo,brave,google,startpage", "auto", "mojeek,yahoo,wikipedia"]


def _ddgs_search(query: str, n: int) -> list[dict]:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    last_err = None
    for attempt, backend in enumerate(_SEARCH_BACKENDS):
        try:
            results = DDGS(timeout=12).text(query, max_results=n, backend=backend)
            if results:
                return results
            last_err = DDGSException("No results found.")
        except Exception as exc:  # RatelimitException, TimeoutException, DDGSException...
            last_err = exc
        time.sleep(min(1.0 * (2**attempt), 6))
    raise last_err if last_err else RuntimeError("search failed")


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """Searches the web (DuckDuckGo plus fallback engines) and returns result titles, URLs and snippets.
    Use it to find pages, then read promising ones with visit_webpage. Snippets can be stale or wrong;
    confirm facts on the page itself. Results about the GAIA benchmark are removed by design.

    Args:
        query: The search query, e.g. 'Marie Curie Nobel Prize years'. Use quotes for exact phrases and site:example.com to restrict a domain.
        max_results: Number of results to return (1-20, default 8).
    """
    try:
        query = (query or "").strip()
        if not query:
            return "ERROR: empty query."
        if is_blocked(query):
            return "ERROR: query refused. " + BLOCK_NOTE
        try:
            n = max(1, min(int(max_results or 8), 20))
        except (TypeError, ValueError):
            n = 8
        try:
            raw = _ddgs_search(query, n + 6)
        except Exception as exc:
            return f"ERROR: web search failed after retries ({type(exc).__name__}: {exc}). Try rephrasing, or use wikipedia_search."
        kept, dropped = [], 0
        seen = set()
        for r in raw:
            url = r.get("href") or r.get("url") or ""
            title = r.get("title", "") or ""
            body = r.get("body", "") or ""
            if is_blocked(url) or is_blocked(title) or is_blocked(body):
                dropped += 1
                continue
            if url in seen:
                continue
            seen.add(url)
            kept.append((title, url, body))
            if len(kept) >= n:
                break
        if not kept:
            note = f" ({dropped} results were removed by the answer-leak guard)" if dropped else ""
            return f"No results found for '{query}'{note}. Try different keywords."
        lines = [f"Search results for: {query}"]
        for i, (title, url, body) in enumerate(kept, 1):
            body = re.sub(r"\s+", " ", body).strip()
            if len(body) > 400:
                body = body[:400] + "..."
            lines.append(f"{i}. {title}\n   URL: {url}\n   {body}")
        if dropped:
            lines.append(f"[{dropped} result(s) removed by the answer-leak guard.]")
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: web_search failed: {type(exc).__name__}: {exc}"


def _pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "ERROR: this URL is a PDF and no PDF library (pypdf) is installed, so its text cannot be extracted."
    reader = PdfReader(BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:  # broken page
            txt = f"[page could not be extracted: {exc}]"
        parts.append(f"--- page {i} ---\n{txt.strip()}")
    text = "\n\n".join(parts).strip()
    if not re.search(r"\w", re.sub(r"--- page \d+ ---", "", text)):
        return f"[PDF with {len(reader.pages)} page(s) but no extractable text; it is probably scanned images.]"
    return f"[PDF, {len(reader.pages)} page(s)]\n\n" + text


_WIKI_URL_RE = re.compile(r"^https?://([a-z\-]+)\.(?:m\.)?wikipedia\.org/(?:wiki/([^?#]+)|w/index\.php\?.*)", re.I)


def _wikipedia_from_url(url: str):
    """If url is a Wikipedia article (or oldid) URL, return (lang, title, oldid); else None."""
    m = _WIKI_URL_RE.match(url)
    if not m:
        return None
    lang = m.group(1).lower()
    qs = parse_qs(urlparse(url).query)
    oldid = None
    if "oldid" in qs and qs["oldid"][0].isdigit():
        oldid = int(qs["oldid"][0])
    title = unquote(m.group(2)).replace("_", " ") if m.group(2) else (qs.get("title", [""])[0].replace("_", " "))
    if not title and not oldid:
        return None
    if title.split(":", 1)[0] in ("Special", "File", "Media") or ("action" in qs and qs["action"][0] != "view"):
        return None
    return lang, title, oldid


def _fetch_url_as_text(url: str) -> str:
    hit = _cache.get_or_none(("url", url))
    if hit is not None:
        return hit
    wiki = _wikipedia_from_url(url)
    if wiki:
        lang, title, oldid = wiki
        data = _parse_page(title=title, oldid=oldid, lang=lang)
        real = data.get("title", title)
        md = html_to_markdown(data["text"], base_url=_wiki_url(real, lang), keep_links=False, wiki=True)
        text = f"Title: {real}\nURL: {url}\n(Wikipedia via API{', revision ' + str(oldid) if oldid else ''})\n\n{md}"
        _cache.put(("url", url), text)
        return text
    resp = _http_get(url, headers={"Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"}, allow_redirects=True)
    final_url = resp.url
    if is_blocked(final_url):
        return "ERROR: the page redirected to a blocked URL. " + BLOCK_NOTE
    if resp.status_code >= 400:
        return f"ERROR: HTTP {resp.status_code} fetching {url}. The page may not exist or may block bots; try another source (e.g. web_search, or the Wayback Machine at https://web.archive.org/web/2023/{url})."
    ctype = (resp.headers.get("Content-Type") or "").lower()
    content = resp.content
    if "pdf" in ctype or content[:5] == b"%PDF-" or final_url.lower().split("?")[0].endswith(".pdf"):
        text = _pdf_to_text(content)
    elif "html" in ctype or "xml" in ctype or content.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
        soup_title = BeautifulSoup(content, "lxml").title
        title = soup_title.get_text(strip=True) if soup_title else ""
        md = html_to_markdown(content, base_url=final_url, keep_links=True, wiki=False)
        text = f"Title: {title}\nURL: {final_url}\n\n{md}"
    elif ctype.startswith("text/") or "json" in ctype or "javascript" in ctype or "csv" in ctype:
        text = resp.text
    else:
        text = (
            f"ERROR: {url} returned non-text content (Content-Type: {ctype or 'unknown'}, {len(content)} bytes). "
            "Use a file/media tool for images, audio, video or spreadsheets."
        )
    _cache.put(("url", url), text)
    return text


@tool
def visit_webpage(url: str, start: int = 0, max_chars: int = 15000) -> str:
    """Fetches a web page (HTML or PDF) and returns its content as markdown, including tables.
    Long pages are paged: the output ends with a note giving the `start` value for the next chunk.
    Wikipedia article URLs (including ?oldid= revision links) are rendered through the Wikipedia API.

    Args:
        url: Full URL to read, starting with http:// or https://.
        start: Character offset to start reading from (use the value given in the truncation note to page forward).
        max_chars: Maximum number of characters to return (default 15000, max 60000).
    """
    try:
        url = (url or "").strip()
        if not url:
            return "ERROR: empty url."
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url
        if is_blocked(url):
            return "ERROR: URL refused. " + BLOCK_NOTE
        start, max_chars = _clamp_paging(start, max_chars)
        text = _fetch_url_as_text(url)
        if text.startswith("ERROR:"):
            return text
        if not text.strip():
            return f"The page at {url} returned no readable text (it may need JavaScript)."
        return _page_text(text, start, max_chars, f"visit_webpage(url={url!r}, start={{start}})")
    except Exception as exc:
        return f"ERROR: could not fetch {url}: {type(exc).__name__}: {exc}"


@tool
def wikipedia_search(query: str) -> str:
    """Searches English Wikipedia and returns the top matching article titles with snippets.
    Use the exact title it returns with wikipedia_page, wikipedia_revision or wikipedia_page_info.

    Args:
        query: Search terms, e.g. 'Marie Curie' or 'tallest buildings in Europe'.
    """
    try:
        query = (query or "").strip()
        if not query:
            return "ERROR: empty query."
        results, info = _search_titles(query, limit=10)
        if not results:
            msg = f"No Wikipedia results for '{query}'."
            if info.get("suggestion"):
                msg += f" Did you mean: {info['suggestion']}?"
            return msg
        lines = [f"Wikipedia search results for: {query} (total hits: {info.get('totalhits', '?')})"]
        for i, r in enumerate(results, 1):
            snippet = _strip_html(r.get("snippet", ""))
            lines.append(
                f"{i}. {r['title']} ({r.get('wordcount', '?')} words, last edited {r.get('timestamp', '?')[:10]})\n"
                f"   {_wiki_url(r['title'])}\n   {snippet}"
            )
        if info.get("suggestion"):
            lines.append(f"Did you mean: {info['suggestion']}?")
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: wikipedia_search failed: {type(exc).__name__}: {exc}"


@tool
def wikipedia_page(title: str, section: str = "", start: int = 0, max_chars: int = 15000) -> str:
    """Returns the CURRENT version of an English Wikipedia article as markdown, with tables rendered as
    markdown tables (rowspan/colspan expanded). The header lists the article's numbered sections; pass
    one as `section` (by name like 'History' or number like '3.1') to read only that part.
    Long output is paged with `start`. For how a page looked on a past date, use wikipedia_revision.

    Args:
        title: Article title (e.g. 'Marie Curie') or a full en.wikipedia.org URL. Redirects are followed.
        section: Optional section name or number to fetch only that section (empty = whole article).
        start: Character offset to start from (use the value given in the truncation note to page forward).
        max_chars: Maximum characters to return (default 15000, max 60000).
    """
    try:
        t = _clean_title(title)
        if not t:
            return "ERROR: empty title."
        start, max_chars = _clamp_paging(start, max_chars)
        try:
            _parse_page(title=t)
        except RuntimeError as exc:
            if "missingtitle" in str(exc) or "invalidtitle" in str(exc):
                return _missing_page_message(t)
            raise
        how = "wikipedia_page(title={title}, section={section}, start={start})"
        return _render_wiki(t, section or "", start, max_chars, None, "en", "(current version)\n", how)
    except Exception as exc:
        return f"ERROR: wikipedia_page failed for '{title}': {type(exc).__name__}: {exc}"


@tool
def wikipedia_revision(title: str, as_of: str, section: str = "", start: int = 0, max_chars: int = 15000) -> str:
    """Returns an English Wikipedia article as it was on a past date: the latest revision saved at or
    before the end of `as_of` (UTC). Same markdown/table rendering and section/paging options as
    wikipedia_page. Use it for questions like 'according to Wikipedia as of July 2023' or 'the 2022 version'.

    Args:
        title: Article title (e.g. 'Marie Curie') or a full en.wikipedia.org URL.
        as_of: Date in ISO format YYYY-MM-DD. 'YYYY-MM' means the end of that month and 'YYYY' the end of that year.
        section: Optional section name or number to fetch only that section (empty = whole article).
        start: Character offset to start from (use the value given in the truncation note to page forward).
        max_chars: Maximum characters to return (default 15000, max 60000).
    """
    try:
        t = _clean_title(title)
        if not t:
            return "ERROR: empty title."
        start, max_chars = _clamp_paging(start, max_chars)
        try:
            rev = _resolve_revision(t, as_of)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if rev.get("missing"):
            return _missing_page_message(t)
        if rev.get("none_before"):
            first = _first_revision(rev["title"])
            when = f" Its first revision was saved {first['timestamp']}." if first else ""
            return f"ERROR: '{rev['title']}' had no revisions on or before {as_of} (the page did not exist yet).{when}"
        extra = (
            f"Revision: oldid={rev['revid']} saved {rev['timestamp']} by {rev.get('user', '?')} "
            f"(latest revision on or before {as_of}; edit summary: {rev.get('comment', '')[:150]!r})\n"
            f"Permalink: https://en.wikipedia.org/w/index.php?oldid={rev['revid']}\n"
        )
        how = "wikipedia_revision(title={title}, as_of=" + repr(as_of) + ", section={section}, start={start})"
        return _render_wiki(rev["title"], section or "", start, max_chars, int(rev["revid"]), "en", extra, how)
    except Exception as exc:
        return f"ERROR: wikipedia_revision failed for '{title}' as of {as_of}: {type(exc).__name__}: {exc}"


def _first_revision(title: str, lang: str = "en") -> dict | None:
    data = _wiki_api(
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "redirects": "1",
            "rvlimit": "1",
            "rvdir": "newer",
            "rvprop": "ids|timestamp|user|comment",
        },
        lang,
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    revs = pages[0].get("revisions") or []
    return revs[0] if revs else None


def _extract_templates(wikitext: str, names: tuple[str, ...]) -> list[str]:
    """Return raw {{...}} template blocks whose name starts with any of `names` (balanced braces)."""
    out = []
    lower = wikitext.lower()
    i = 0
    while True:
        i = lower.find("{{", i)
        if i < 0:
            break
        head = lower[i + 2 : i + 60].lstrip()
        if any(head.startswith(n) for n in names):
            depth, j = 0, i
            while j < len(wikitext) - 1:
                if wikitext[j : j + 2] == "{{":
                    depth += 1
                    j += 2
                    continue
                if wikitext[j : j + 2] == "}}":
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                    continue
                j += 1
            out.append(wikitext[i:j])
            i = j
        else:
            i += 2
    return out


@tool
def wikipedia_page_info(title: str) -> str:
    """Returns metadata about an English Wikipedia article: canonical title and URL, talk page link,
    page creation (first revision) date and creator, latest edit, size, Wikidata id, quality status
    (Featured / Good article / Featured list categories), the talk page's article-history record
    (FAC/GAN nominations, dates and results, when present) and any existing Featured-article-candidate
    or Good-article review pages with their creation date and creator (usually the nominator / reviewer).

    Args:
        title: Article title (e.g. 'Marie Curie') or a full en.wikipedia.org URL.
    """
    try:
        t = _clean_title(title)
        if not t:
            return "ERROR: empty title."
        data = _wiki_api(
            {
                "action": "query",
                "titles": t,
                "redirects": "1",
                "prop": "info|pageprops|categories|revisions",
                "inprop": "talkid|url|displaytitle",
                "clcategories": "Category:Featured articles|Category:Good articles|Category:Featured lists|Category:Former featured articles",
                "rvprop": "ids|timestamp|user|size",
                "rvlimit": "1",
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return _missing_page_message(t)
        page = pages[0]
        real = page["title"]
        lines = [f"Title: {real}", f"URL: {page.get('fullurl', _wiki_url(real))}"]
        redirects = data.get("query", {}).get("redirects") or []
        if redirects:
            lines.append(f"(redirected from '{redirects[0].get('from')}')")
        lines.append(f"Page id: {page.get('pageid')}, current size: {page.get('length')} bytes")
        talk_title = "Talk:" + real
        lines.append(f"Talk page: {_wiki_url(talk_title)}" + ("" if page.get("talkid") else " (does not exist)"))
        latest = (page.get("revisions") or [{}])[0]
        if latest:
            lines.append(f"Latest revision: oldid={latest.get('revid')} at {latest.get('timestamp')} by {latest.get('user')}")
        first = _first_revision(real)
        if first:
            lines.append(
                f"First revision (page created): {first.get('timestamp')} by {first.get('user')} (oldid={first.get('revid')})"
            )
        lines.append(f"History: https://en.wikipedia.org/w/index.php?title={quote(real.replace(' ', '_'))}&action=history")
        props = page.get("pageprops") or {}
        if props.get("wikibase_item"):
            lines.append(f"Wikidata: https://www.wikidata.org/wiki/{props['wikibase_item']}")
        if "disambiguation" in props:
            lines.append("This is a disambiguation page.")
        cats = [c["title"].replace("Category:", "") for c in page.get("categories") or []]
        lines.append("Quality status: " + (", ".join(cats) if cats else "not in Featured/Good article/Featured list categories"))

        # Talk page article history (FAC/GAN/peer review records)
        if page.get("talkid"):
            try:
                tw = _wiki_api({"action": "parse", "page": talk_title, "prop": "wikitext", "section": "0"})
                wikitext = tw.get("parse", {}).get("wikitext", "")
                blocks = _extract_templates(
                    wikitext, ("article history", "articlehistory", "article milestones", "featured article", "fa passed", "ga|", "ga |", "good article", "failedga", "dyk talk", "on this day")
                )
                if blocks:
                    joined = "\n".join(blocks)
                    if len(joined) > 4000:
                        joined = joined[:4000] + "\n[...truncated; read the talk page with visit_webpage for the rest]"
                    lines.append("Talk-page history templates (raw wikitext):\n" + joined)
            except Exception as exc:
                lines.append(f"(talk page templates unavailable: {exc})")

        # Nomination / review pages
        candidates = [f"Wikipedia:Featured article candidates/{real}"] + [
            f"Wikipedia:Featured article candidates/{real}/archive{i}" for i in range(1, 5)
        ] + [f"Wikipedia:Featured list candidates/{real}", f"Wikipedia:Featured list candidates/{real}/archive1"] + [
            f"Talk:{real}/GA{i}" for i in range(1, 4)
        ] + [f"Wikipedia:Peer review/{real}/archive1"]
        exists = _wiki_api({"action": "query", "titles": "|".join(candidates), "prop": "info"})
        found = [p["title"] for p in exists.get("query", {}).get("pages", []) if not p.get("missing") and not p.get("invalid")]
        order = {c: i for i, c in enumerate(candidates)}
        found.sort(key=lambda x: order.get(x, 99))
        if found:
            lines.append("Nomination / review pages:")
            for f_title in found:
                fr = _first_revision(f_title)
                created = f" created {fr.get('timestamp')} by {fr.get('user')}" if fr else ""
                lines.append(f"- {f_title}: {_wiki_url(f_title)}{created}")
        else:
            lines.append("Nomination / review pages: none found under the standard FAC/FLC/GA/peer-review titles.")
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: wikipedia_page_info failed for '{title}': {type(exc).__name__}: {exc}"


TOOLS_WEB = [web_search, visit_webpage, wikipedia_search, wikipedia_page, wikipedia_revision, wikipedia_page_info]

__all__ = [
    "TOOLS_WEB",
    "BLOCK_NOTE",
    "is_blocked",
    "html_to_markdown",
    "table_to_markdown",
    "web_search",
    "visit_webpage",
    "wikipedia_search",
    "wikipedia_page",
    "wikipedia_revision",
    "wikipedia_page_info",
]
