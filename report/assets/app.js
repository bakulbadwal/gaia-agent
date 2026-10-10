/* GAIA Agent results site. Vanilla JS, no dependencies.
 * All data comes from the JSON block that report/build_report.py inlines into index.html.
 * Every string from a run (questions, model output, web pages the agent read) is untrusted:
 * it is only ever inserted with textContent / createTextNode, never as HTML. */
(function () {
  "use strict";

  var DATA;
  try {
    DATA = JSON.parse(document.getElementById("report-data").textContent);
  } catch (e) {
    document.getElementById("main").prepend(document.createTextNode("Could not read the report data: " + e));
    return;
  }

  var META = DATA.meta || {};
  /* Public build (the default): GAIA items carry no question, answer or step text at all; the
   * builder strips it before this page is written. The page only adapts its layout to that. */
  var PUBLIC = META.public !== false;
  var QUESTIONS = DATA.questions || [];
  var TRACES = DATA.traces || {};
  var AN = DATA.analytics || {};
  var QBYID = {};
  QUESTIONS.forEach(function (q) { QBYID[q.task_id] = q; });
  var PR = DATA.practice || null;
  var PRUNS = (PR && PR.runs) || [];
  var PRUN = {};
  PRUNS.forEach(function (r) { PRUN[r.run_id] = r; });
  var PTASKS = (PR && PR.tasks) || {};
  var USAGE = DATA.usage || null;
  var prac = { run: (PR && PR.default_run) || (PRUNS[0] || {}).run_id || null, select: null };


  var WEB = ["web_search", "visit_webpage", "wikipedia_search", "wikipedia_page", "wikipedia_revision", "wikipedia_page_info"];
  var MEDIA = ["read_file", "run_python_file", "transcribe_audio", "image_question", "youtube_transcript", "youtube_audio_transcript", "youtube_frames_question"];
  var KNOWN = {};
  (META.known_tools || WEB.concat(MEDIA)).forEach(function (t) { KNOWN[t] = 1; });
  function learnTools(tr) {
    var by = ((tr || {}).totals || {}).tool_calls_by_name || {};
    Object.keys(by).forEach(function (t) { KNOWN[t] = 1; });
  }
  Object.keys(TRACES).forEach(function (id) { learnTools(TRACES[id]); });
  Object.keys((PR && PR.traces) || {}).forEach(function (rid) {
    Object.keys(PR.traces[rid]).forEach(function (tid) { learnTools(PR.traces[rid][tid]); });
  });
  var STATUS = { answered: "Answered", error: "Error", timeout: "Timeout", empty: "No answer" };
  var TRUNC_RE = /…\[truncated (\d+) chars\]/;

  /* ------------------------------------------------------------------ DOM helpers */

  function h(tag, attrs) {
    var e = document.createElement(tag);
    setAttrs(e, attrs);
    for (var i = 2; i < arguments.length; i++) add(e, arguments[i]);
    return e;
  }
  var SVGNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs) {
    var e = document.createElementNS(SVGNS, tag);
    setAttrs(e, attrs);
    for (var i = 2; i < arguments.length; i++) add(e, arguments[i]);
    return e;
  }
  function setAttrs(e, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "class") e.setAttribute("class", v);
      else if (k === "text") e.textContent = v;
      else if (k.slice(0, 2) === "on") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    });
  }
  function add(parent, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { add(parent, x); }); return; }
    parent.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  function $(sel, root) { return (root || document).querySelector(sel); }
  function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }

  function icon(path, size) {
    return s("svg", { width: size || 14, height: size || 14, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" }, s("path", { d: path }));
  }
  var I = {
    chev: "M9 6l6 6-6 6",
    search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
    info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 16v-4M12 8h.01",
    prev: "M15 18l-6-6 6-6",
    next: "M9 18l6-6-6-6",
    expand: "M4 9l8 8 8-8",
    collapse: "M4 15l8-8 8 8",
    sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
    moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
    lock: "M6 11h12v10H6zM8 11V7a4 4 0 0 1 8 0v4",
    check: "M5 12l5 5 9-11",
    cross: "M6 6l12 12M18 6L6 18"
  };

  /* ------------------------------------------------------------------ formatting */

  function isNum(x) { return typeof x === "number" && isFinite(x); }
  function fmtInt(n) { return isNum(n) ? Math.round(n).toLocaleString("en-US") : "—"; }
  function fmtTok(n) {
    if (!isNum(n) || n === 0) return "—";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e4) return Math.round(n / 1e3) + "k";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(n);
  }
  function fmtSec(x) {
    if (!isNum(x)) return "—";
    if (x < 10) return x.toFixed(1) + " s";
    if (x < 60) return Math.round(x) + " s";
    var m = Math.floor(x / 60), r = Math.round(x - m * 60);
    if (r === 60) { m += 1; r = 0; }
    return m + "m " + (r < 10 ? "0" : "") + r + "s";
  }
  function fmtPct(p, d) { return isNum(p) ? (p * 100).toFixed(d === undefined ? 1 : d) + "%" : "—"; }
  function fmtDate(iso) {
    if (!iso) return "—";
    var d = new Date(iso);
    if (isNaN(d)) return String(iso);
    return d.toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit", timeZone: "UTC" }) + " UTC";
  }
  function shortQ(q, n) {
    q = String(q || "").replace(/\s+/g, " ").trim();
    n = n || 110;
    return q.length > n ? q.slice(0, n - 1).replace(/\s+\S*$/, "") + "…" : q;
  }
  function toolClass(name) { return WEB.indexOf(name) >= 0 ? "" : MEDIA.indexOf(name) >= 0 ? "media" : "other"; }
  function toolChip(name, count) {
    return h("span", { class: "tool " + toolClass(name), title: count > 1 ? name + " × " + count : name }, name, count > 1 ? h("span", { class: "x" }, "×" + count) : null);
  }
  function pill(status) { return h("span", { class: "pill " + status }, STATUS[status] || status); }
  function plural(n, w) { return n + " " + w + (n === 1 ? "" : "s"); }
  /* What to call a GAIA item: its question in a private build, its id and file type in public. */
  function qLabel(q, n) {
    if (!q) return "";
    if (q.question) return shortQ(q.question, n);
    return "task " + q.task_id + (q.attachment ? " · ." + q.attachment + " file" : "");
  }
  function attChip(ext) {
    return ext ? h("span", { class: "att", title: "Attachment type" }, "." + ext) : null;
  }
  function verdictPill(it) {
    if (!it || it.skipped) return h("span", { class: "pill empty" }, "Skipped");
    if (it.correct === true) return h("span", { class: "pill answered", title: it.correct_strict === false ? "Passes GAIA's normalized match, fails the strict string match" : null }, it.correct_strict === false ? "Correct · format miss" : "Correct");
    if (it.correct === false) return h("span", { class: "pill error" }, "Wrong");
    return h("span", { class: "pill empty" }, "Ungraded");
  }
  function humanTag(t) {
    var parts = String(t).split(":");
    var base = parts[0].replace(/[_-]+/g, " ").trim();
    base = base.charAt(0).toUpperCase() + base.slice(1);
    return parts.length > 1 ? base + " · " + parts.slice(1).join(":") : base;
  }
  /* Tags that record what the agent did rather than something going wrong. */
  var INFO_TAGS = { vision_used: 1 };

  /* Text with the trace contract's truncation marker rendered as a badge. */
  function textNodes(str) {
    str = str == null ? "" : String(str);
    var m = TRUNC_RE.exec(str);
    if (!m) return [document.createTextNode(str)];
    return [
      document.createTextNode(str.slice(0, m.index)),
      h("span", { class: "trunc", title: "Clipped when the trace was saved" }, "+" + fmtInt(+m[1]) + " chars truncated"),
      document.createTextNode(str.slice(m.index + m[0].length))
    ];
  }

  function empty(title, body, cmd) {
    return h("div", { class: "empty" }, h("strong", null, title), h("span", null, body), cmd ? h("div", { style: "margin-top:10px" }, h("code", null, cmd)) : null);
  }

  /* ------------------------------------------------------------------ tooltip */

  var tip = $("#tip");
  function bindTip(el, text) {
    el.addEventListener("mouseenter", function () { tip.textContent = text; tip.classList.add("on"); });
    el.addEventListener("mousemove", function (ev) { placeTip(ev.clientX, ev.clientY); });
    el.addEventListener("mouseleave", function () { tip.classList.remove("on"); });
    el.addEventListener("focus", function () {
      var r = el.getBoundingClientRect();
      tip.textContent = text; tip.classList.add("on"); placeTip(r.left + r.width / 2, r.top);
    });
    el.addEventListener("blur", function () { tip.classList.remove("on"); });
  }
  function placeTip(x, y) {
    var w = tip.offsetWidth, hh = tip.offsetHeight;
    var left = Math.min(Math.max(8, x - w / 2), window.innerWidth - w - 8);
    var top = y - hh - 12;
    if (top < 8) top = y + 16;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  /* ------------------------------------------------------------------ theme + nav */

  function initTheme() {
    var btn = $("#theme-btn");
    function effective() {
      var t = document.documentElement.getAttribute("data-theme");
      if (t) return t;
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    function paint() {
      clear(btn).appendChild(icon(effective() === "dark" ? I.sun : I.moon, 16));
      btn.setAttribute("aria-label", "Switch to " + (effective() === "dark" ? "light" : "dark") + " theme");
    }
    btn.addEventListener("click", function () {
      var next = effective() === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("gaia-theme", next); } catch (e) { /* storage unavailable */ }
      paint();
    });
    paint();
  }

  function initNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".nav a"));
    if (!("IntersectionObserver" in window)) return;
    var current = null;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) current = en.target.id;
      });
      links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === "#" + current); });
    }, { rootMargin: "-45% 0px -50% 0px" });
    links.forEach(function (a) { var t = document.getElementById(a.getAttribute("href").slice(1)); if (t) io.observe(t); });
  }

  /* ------------------------------------------------------------------ hero */

  function renderHero() {
    if (META.fixture) $("#fixture-banner").hidden = false;
    if (!PUBLIC) $("#private-banner").hidden = false;

    var chips = $("#meta-chips");
    var items = [
      ["Model", META.model], ["Provider", META.provider], ["Architecture", META.arch ? (META.arch === "multi" ? "multi-agent" : "single CodeAgent") : null],
      ["Vision", META.vision_model], ["GAIA run", META.run_id], ["Build", PUBLIC ? "public · GAIA redacted" : "private · full text"]
    ];
    var any = false;
    items.forEach(function (it) {
      if (!it[1]) return;
      any = true;
      chips.appendChild(h("span", { class: "chip" }, h("b", null, it[0]), h("span", null, String(it[1]))));
    });
    if (!any) chips.appendChild(h("span", { class: "chip" }, h("b", null, "Run"), h("span", null, "none yet")));

    var card = $("#scorecard");
    var sub = (DATA.submission || {}).latest;
    var mark = (DATA.submission || {}).pass_mark || 30;
    var score = sub && isNum(sub.score) ? sub.score : null;
    var top = h("div", { class: "score-top" },
      h("div", null,
        h("div", { class: "score-label" }, META.fixture ? "Official score (fixture)" : "Official score"),
        h("div", { class: "score-big" }, score === null ? "—" : String(+score.toFixed(1)), score === null ? null : h("small", null, "%"))
      ),
      score === null ? h("span", { class: "verdict fail" }, "Not submitted")
        : score >= mark ? h("span", { class: "verdict pass" }, "Above the " + mark + "% pass mark")
        : h("span", { class: "verdict fail" }, "Below the " + mark + "% pass mark")
    );
    card.appendChild(top);
    if (sub) {
      card.appendChild(h("div", { class: "score-sub" },
        isNum(sub.correct_count) ? h("b", null, sub.correct_count + " correct") : null,
        isNum(sub.total_attempted) ? " of " + sub.total_attempted + " submitted, graded by the course API" : " graded by the course API"));
    } else {
      card.appendChild(h("div", { class: "score-sub" }, "No submission recorded yet. ", h("code", null, "python submit.py"), " does a dry run; ", h("code", null, "--yes"), " sends it."));
    }
    var pct = score === null ? 0 : Math.max(0, Math.min(100, score));
    card.appendChild(h("div", null,
      h("div", { class: "gauge", role: "img", "aria-label": score === null ? "No score yet; pass mark " + mark + "%" : "Score " + score + "% against a pass mark of " + mark + "%" },
        h("div", { class: "gauge-fill", style: "width:" + pct + "%" }),
        h("div", { class: "gauge-mark", style: "left:" + mark + "%" }, h("span", null, "pass " + mark + "%"))
      ),
      h("div", { class: "gauge-scale" }, h("span", null, "0"), h("span", null, "100%"))
    ));
    if (sub) {
      var dl = h("dl", { class: "score-meta" },
        h("div", null, h("dt", null, "Submitted"), h("dd", null, fmtDate(sub.submitted_at))),
        h("div", null, h("dt", null, "Run"), h("dd", { class: "mono" }, sub.run_id || "—"))
      );
      card.appendChild(dl);
      var hist = (DATA.submission.history || []).filter(function (x) { return isNum(x.score); });
      if (hist.length > 1) {
        var row = h("div", { class: "history" }, "History");
        hist.forEach(function (x, i) {
          if (i) row.appendChild(document.createTextNode("→"));
          row.appendChild(h("span", { class: "h-pt", title: fmtDate(x.submitted_at) + " · " + (x.run_id || "") }, +x.score.toFixed(1) + "%"));
        });
        card.appendChild(row);
      }
      if (sub.message) card.appendChild(h("p", { class: "muted", style: "font-size:12.5px" }, "API: ", h("span", { class: "mono" }, sub.message)));
    }

    var k = $("#kpis");
    var st = AN.status || {};
    var n = QUESTIONS.length;
    var EV = DATA.evals || {};
    var base = (EV.groups || []).filter(function (g) { return g.is_baseline; })[0];
    var pk = base && base.passk && base.passk.k.length > 1 ? base.passk : null;
    var U = (USAGE || {}).totals || null;
    var kp = [
      ["Practice accuracy", base && isNum(base.accuracy) ? fmtPct(base.accuracy, 0) : "—",
        base ? base.label + " · " + plural(base.runs, "run") + " × " + base.items_per_run : "practice set not run yet", "#evals"],
      pk ? ["Reliability", fmtPct(pk.pass_hat_k[pk.k.length - 1], 0), "pass^" + pk.k[pk.k.length - 1] + ": right on every run", "#evals"]
        : ["Reliability", "—", "needs 2+ repeat runs", "#evals"],
      ["GAIA questions", n ? fmtInt(n) : "—", n ? fmtInt(st.answered || 0) + " answered · " + ((st.error || 0) + (st.timeout || 0)) + " failed" : "not run yet", "#questions"],
      ["Median time", fmtSec((AN.seconds || {}).median), isNum((AN.seconds || {}).p90) ? "p90 " + fmtSec(AN.seconds.p90) + " per question" : "", "#analytics"],
      ["Median steps", isNum((AN.steps || {}).median) ? String(+AN.steps.median.toFixed(1)) : "—", isNum((AN.steps || {}).max) ? "max " + AN.steps.max : "", "#analytics"],
      ["Model requests", U ? fmtInt(U.requests) : "—", U ? fmtTok(U.input_tokens + U.output_tokens) + " tokens · " + plural(U.retries, "retry").replace("retrys", "retries") : "none logged yet", "#usage"],
      ["API cost", USAGE ? (USAGE.all_free === false ? "paid!" : "$0") : "—", USAGE ? (USAGE.all_free === false ? "a non-free model was used" : "free models only") : "", "#usage"]
    ];
    kp.forEach(function (x) {
      k.appendChild(h("a", { class: "kpi", href: x[3] }, h("div", { class: "k" }, x[0]), h("div", { class: "v" }, x[1]), h("div", { class: "s" }, x[2] || " ")));
    });
  }

  /* ------------------------------------------------------------------ scoreboard */

  var sb = { key: "n", dir: 1, filter: "all", q: "" };
  var COLS = PUBLIC ? [
    { key: "n", label: "#", cls: "c-n" },
    { key: "task_id", label: "Task", cls: "c-q" },
    { key: "tags", label: "Failure tags", cls: "c-a" },
    { key: "status", label: "Status", cls: "c-s" },
    { key: "tools", label: "Tools used", cls: "c-t" },
    { key: "steps", label: "Steps", cls: "r" },
    { key: "model_calls", label: "Calls", cls: "r" },
    { key: "tokens", label: "Tokens", cls: "r" },
    { key: "seconds", label: "Time", cls: "r" }
  ] : [
    { key: "n", label: "#", cls: "c-n" },
    { key: "question", label: "Question", cls: "c-q col-q" },
    { key: "answer", label: "Submitted answer", cls: "c-a" },
    { key: "status", label: "Status", cls: "c-s" },
    { key: "tools", label: "Tools used", cls: "c-t" },
    { key: "steps", label: "Steps", cls: "r" },
    { key: "tokens", label: "Tokens", cls: "r" },
    { key: "seconds", label: "Time", cls: "r" }
  ];
  var STATUS_ORDER = { answered: 0, empty: 1, timeout: 2, error: 3 };
  function sortVal(q, key) {
    if (key === "tools") return Object.keys(q.tools || {}).reduce(function (a, t) { return a + q.tools[t]; }, 0);
    if (key === "status") return STATUS_ORDER[q.status];
    if (key === "question" || key === "answer" || key === "task_id") return String(q[key] || "").toLowerCase();
    if (key === "tags") return (q.tags || []).length;
    return isNum(q[key]) ? q[key] : -1;
  }

  function renderScoreboard() {
    var root = $("#scoreboard");
    if (!QUESTIONS.length) {
      root.appendChild(empty("No GAIA run yet", "Run the agent on the course questions to fill this table. Each answer is cached in runs/answers.jsonl with its trace.", "python run.py"));
      return;
    }
    var counts = { all: QUESTIONS.length };
    QUESTIONS.forEach(function (q) { counts[q.status] = (counts[q.status] || 0) + 1; });

    var ph = PUBLIC ? "Search task ids, tools and tags" : "Search questions and answers";
    var input = h("input", { type: "search", placeholder: ph, "aria-label": ph });
    input.addEventListener("input", function () { sb.q = input.value.trim().toLowerCase(); body(); });
    var seg = h("div", { class: "seg", role: "group", "aria-label": "Filter by status" });
    ["all", "answered", "error", "timeout", "empty"].forEach(function (f) {
      if (f !== "all" && !counts[f]) return;
      var b = h("button", { type: "button", "aria-pressed": f === sb.filter ? "true" : "false" }, f === "all" ? "All" : STATUS[f], h("span", { class: "c" }, counts[f]));
      b.addEventListener("click", function () {
        sb.filter = f;
        Array.prototype.forEach.call(seg.children, function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
        body();
      });
      seg.appendChild(b);
    });
    root.appendChild(h("div", { class: "controls" }, h("label", { class: "search" }, icon(I.search, 15), input), seg));

    var thead = h("tr");
    COLS.forEach(function (c) {
      var th = h("th", { class: c.cls === "r" ? "r" : "", scope: "col" });
      var btn = h("button", { type: "button" }, c.label, h("span", { class: "arrow", "aria-hidden": "true" }, "▲"));
      btn.addEventListener("click", function () {
        sb.dir = sb.key === c.key ? -sb.dir : (c.key === "n" || c.key === "question" || c.key === "answer" || c.key === "task_id" ? 1 : -1);
        sb.key = c.key;
        body();
      });
      th.appendChild(btn);
      th._key = c.key;
      thead.appendChild(th);
    });
    var tbody = h("tbody");
    var table = h("table", { class: "sb" + (PUBLIC ? " pub" : "") }, h("caption", { class: "sr-only" }, "Questions in the submitted run"), h("thead", null, thead), tbody);
    root.appendChild(h("div", { class: "card table-wrap" }, table));
    root.appendChild(h("p", { class: "note" }, icon(I.info, 14),
      h("span", null, "The course API returns one aggregate score, not per-question grades, so no row here says right or wrong. " +
        (PUBLIC ? "Question text and answers are withheld under the contamination policy above. " : "") +
        "Per-item correctness is measured on the practice set.")));

    function body() {
      Array.prototype.forEach.call(thead.children, function (th) {
        if (th._key === sb.key) {
          th.setAttribute("aria-sort", sb.dir > 0 ? "ascending" : "descending");
          th.querySelector(".arrow").textContent = sb.dir > 0 ? "▲" : "▼";
        } else th.removeAttribute("aria-sort");
      });
      var rows = QUESTIONS.filter(function (q) {
        if (sb.filter !== "all" && q.status !== sb.filter) return false;
        if (!sb.q) return true;
        var hay = PUBLIC ? [q.task_id, q.attachment, (q.tags || []).join(" "), Object.keys(q.tools || {}).join(" ")].join(" ")
          : q.question + " " + q.answer + " " + q.task_id;
        return hay.toLowerCase().indexOf(sb.q) >= 0;
      }).slice().sort(function (a, b) {
        var x = sortVal(a, sb.key), y = sortVal(b, sb.key);
        return (x < y ? -1 : x > y ? 1 : a.n - b.n) * (x === y ? 1 : sb.dir);
      });
      clear(tbody);
      if (!rows.length) {
        tbody.appendChild(h("tr", null, h("td", { colspan: COLS.length, class: "muted", style: "text-align:center;padding:24px" }, "No questions match.")));
        return;
      }
      rows.forEach(function (q) {
        var tools = Object.keys(q.tools || {});
        var sel = viewer.src === "gaia" && q.task_id === viewer.id;
        var cells = PUBLIC ? [
          h("td", { class: "c-n" }, h("span", { class: "qn" }, "Q" + q.n)),
          h("td", { class: "c-q" }, h("button", { type: "button", class: "qbtn idbtn", "aria-label": "Open the step skeleton for question " + q.n }, h("span", { class: "mono" }, q.task_id)), attChip(q.attachment)),
          h("td", { class: "c-a" }, (q.tags || []).length ? h("div", { class: "tags" }, q.tags.map(function (t) { return h("span", { class: "tag" + (INFO_TAGS[t] ? " info" : "") }, humanTag(t)); })) : h("span", { class: "muted" }, "no tags")),
          h("td", { class: "c-s" }, pill(q.status)),
          h("td", { class: "c-t" }, tools.length ? h("div", { class: "tools" }, tools.map(function (t) { return toolChip(t, q.tools[t]); })) : h("span", { class: "muted" }, "—")),
          h("td", { class: "r" }, q.steps || "—"),
          h("td", { class: "r" }, q.model_calls || "—"),
          h("td", { class: "r" }, fmtTok(q.tokens)),
          h("td", { class: "r" }, fmtSec(q.seconds))
        ] : [
          h("td", { class: "c-n" }, h("span", { class: "qn" }, "Q" + q.n)),
          h("td", { class: "c-q" }, h("button", { type: "button", class: "qbtn", title: q.question, "aria-label": "Open trajectory for question " + q.n + ": " + shortQ(q.question, 80) }, shortQ(q.question, 150))),
          h("td", { class: "c-a" }, q.answer ? h("span", { class: "ans" }, q.answer) : h("span", { class: "ans none" }, "no answer")),
          h("td", { class: "c-s" }, pill(q.status)),
          h("td", { class: "c-t" }, tools.length ? h("div", { class: "tools" }, tools.map(function (t) { return toolChip(t, q.tools[t]); })) : h("span", { class: "muted" }, "—")),
          h("td", { class: "r" }, q.steps || "—"),
          h("td", { class: "r" }, fmtTok(q.tokens)),
          h("td", { class: "r" }, fmtSec(q.seconds))
        ];
        cells.push(h("td", { class: "r c-m" }, h("span", null, plural(q.steps || 0, "step")), h("span", null, fmtTok(q.tokens) + " tok"), h("span", null, fmtSec(q.seconds))));
        var tr = h("tr", { "data-id": q.task_id, class: sel ? "sel" : null }, cells);
        tr.addEventListener("click", function () { openGaia(q.task_id, true); });
        tbody.appendChild(tr);
      });
    }
    sb.body = body;
    body();
  }
  function markRows() {
    document.querySelectorAll("table.sb tbody tr").forEach(function (tr) {
      tr.classList.toggle("sel", viewer.src === "gaia" && tr.getAttribute("data-id") === viewer.id);
    });
    document.querySelectorAll(".prow").forEach(function (r) {
      r.classList.toggle("sel", viewer.src === "practice" && r.getAttribute("data-run") === viewer.run && r.getAttribute("data-id") === viewer.id);
    });
  }

  /* ------------------------------------------------------------------ code + text rendering */

  var KW = "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(" ");
  var KWSET = {};
  KW.forEach(function (k) { KWSET[k] = 1; });
  var TOK = /(#[^\n]*)|("""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|[rRbBfFuU]{0,2}"(?:\\.|[^"\\\n])*"?|[rRbBfFuU]{0,2}'(?:\\.|[^'\\\n])*'?)|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;

  function highlight(code) {
    var frag = document.createDocumentFragment();
    var tail = null, m0 = TRUNC_RE.exec(code);
    if (m0) { tail = code.slice(m0.index); code = code.slice(0, m0.index); }
    var last = 0, m;
    TOK.lastIndex = 0;
    while ((m = TOK.exec(code))) {
      if (m.index > last) frag.appendChild(document.createTextNode(code.slice(last, m.index)));
      var cls = null;
      if (m[1]) cls = "t-com";
      else if (m[2]) cls = "t-str";
      else if (m[3]) cls = "t-num";
      else if (m[4]) {
        if (KWSET[m[4]]) cls = "t-kw";
        else if (m[4] === "final_answer") cls = "t-fa";
        else if (KNOWN[m[4]] && code.slice(TOK.lastIndex).match(/^\s*\(/)) cls = "t-tool";
      }
      frag.appendChild(cls ? h("span", { class: cls }, m[0]) : document.createTextNode(m[0]));
      last = TOK.lastIndex;
      if (m[0] === "") TOK.lastIndex++;
    }
    if (last < code.length) frag.appendChild(document.createTextNode(code.slice(last)));
    if (tail) add(frag, textNodes(tail));
    return frag;
  }

  function thoughtOf(out) {
    if (out == null) return "";
    var t = typeof out === "string" ? out : JSON.stringify(out, null, 2);
    t = t.replace(/<code>[\s\S]*?(?:<\/code>|$)/g, "")
      .replace(/```(?:py|python)?\s*\n[\s\S]*?(?:```|$)/g, "")
      .replace(/^\s*Thoughts?:\s*/i, "")
      .replace(/<end_code>/g, "")
      .trim();
    return t;
  }

  function renderPlan(text) {
    var box = h("div", { class: "plan" });
    String(text || "").replace(/<end_plan>/g, "").trim().split("\n").forEach(function (line, i) {
      if (i) box.appendChild(document.createTextNode("\n"));
      var mh = /^\s*#{1,6}\s+(.*)$/.exec(line);
      if (mh) box.appendChild(h("span", { class: "h" }, mh[1]));
      else add(box, textNodes(line));
    });
    return box;
  }

  function renderObservation(obs) {
    var wrap = h("div");
    var str = String(obs);
    var pre = h("pre", { class: "obs" });
    var m = /^Execution logs:\n?([\s\S]*?)(?:\n?Last output from code snippet:\n?([\s\S]*))?$/.exec(str);
    if (m) {
      var logs = (m[1] || "").replace(/\s+$/, "");
      var ret = (m[2] || "").trim();
      if (logs) { pre.appendChild(h("span", { class: "obs-sub" }, "Printed")); add(pre, textNodes(logs)); }
      if (ret && ret !== "None") {
        if (logs) pre.appendChild(document.createTextNode("\n"));
        pre.appendChild(h("span", { class: "obs-sub" }, "Returned"));
        add(pre, textNodes(ret));
      }
      if (!logs && (!ret || ret === "None")) pre.appendChild(h("span", { class: "muted" }, "(no output)"));
    } else add(pre, textNodes(str));
    wrap.appendChild(pre);
    var lines = str.split("\n").length;
    if (str.length > 1400 || lines > 18) {
      pre.classList.add("clamped");
      var btn = h("button", { type: "button", class: "linkish", "aria-expanded": "false" }, "Show full observation (" + fmtInt(lines) + " lines)");
      btn.addEventListener("click", function () {
        var open = pre.classList.toggle("clamped");
        btn.setAttribute("aria-expanded", open ? "false" : "true");
        btn.textContent = open ? "Show full observation (" + fmtInt(lines) + " lines)" : "Show less";
      });
      wrap.appendChild(h("div", { style: "margin-top:6px" }, btn));
    }
    return wrap;
  }

  function blk(label, content, right) {
    return h("div", null, h("div", { class: "blk-lbl" }, h("span", null, label), right ? h("span", null, right) : null), content);
  }

  /* ------------------------------------------------------------------ trajectory viewer */

  /* One viewer, two sources. A practice trajectory is shown in full (our own task); a GAIA one is
   * shown in full only in a private build, and as a step skeleton in the public build. */
  var viewer = { src: null, run: null, id: null };
  var traceCache = {};
  var renderSeq = 0;

  function practiceList(run) { return ((PR && PR.items) || {})[run] || []; }
  function practiceItem(run, id) { return practiceList(run).filter(function (it) { return it.id === id; })[0] || null; }
  function hasPracticeTrace(run, id) {
    return !!((((PR || {}).traces || {})[run] || {})[id] || (((PR || {}).trace_files || {})[run] || {})[id]);
  }
  function hasPractice() { return PRUNS.some(function (r) { return practiceList(r.run_id).length; }); }
  function hasGaia() { return QUESTIONS.length > 0 || Object.keys(TRACES).length > 0; }

  /* cb(trace) once; for a trace that must be fetched, cb(undefined) first while it loads. */
  function getTrace(ref, cb) {
    if (ref.src === "gaia") { cb(TRACES[ref.id] || null); return; }
    var inl = (((PR || {}).traces || {})[ref.run] || {})[ref.id];
    if (inl) { cb(inl); return; }
    var key = ref.run + "/" + ref.id;
    if (Object.prototype.hasOwnProperty.call(traceCache, key)) { cb(traceCache[key]); return; }
    var rel = (((PR || {}).trace_files || {})[ref.run] || {})[ref.id];
    if (!rel) { cb(null); return; }
    cb(undefined);
    fetch((PR.trace_base || "traces/") + rel)
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (tr) { traceCache[key] = tr; learnTools(tr); cb(tr); })
      .catch(function (e) { cb({ _error: String(e && e.message || e) }); });
  }

  function firstPracticeId(run) {
    var items = practiceList(run);
    var traced = items.filter(function (it) { return hasPracticeTrace(run, it.id); });
    // The showpiece: a correct answer that took a few steps, so the trajectory has something to show.
    var pick = traced.filter(function (it) { return it.correct && (it.steps || 0) >= 3; })[0] || traced[0] || items[0];
    return pick ? pick.id : null;
  }
  function firstGaiaId() {
    for (var i = 0; i < QUESTIONS.length; i++) if (QUESTIONS[i].has_trace) return QUESTIONS[i].task_id;
    return QUESTIONS.length ? QUESTIONS[0].task_id : Object.keys(TRACES)[0] || null;
  }
  function setDefaultRef() {
    if (hasPractice()) {
      var run = prac.run || PR.default_run || PRUNS[0].run_id;
      viewer.src = "practice"; viewer.run = run; viewer.id = firstPracticeId(run);
    } else if (hasGaia()) {
      viewer.src = "gaia"; viewer.run = null; viewer.id = firstGaiaId();
    }
  }

  function setHash() {
    var hsh = viewer.src === "practice" ? "#/p/" + encodeURIComponent(viewer.run) + "/" + encodeURIComponent(viewer.id)
      : "#/q/" + encodeURIComponent(viewer.id);
    try { history.replaceState(null, "", hsh); } catch (e) { /* file:// in some browsers */ }
  }
  function scrollToViewer() {
    $("#trajectory").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }
  function openPractice(run, id, scroll) {
    if (!PRUN[run] || !practiceItem(run, id)) return;
    viewer.src = "practice"; viewer.run = run; viewer.id = id;
    if (prac.run !== run && prac.select) prac.select(run);
    renderViewer(); markRows(); setHash();
    if (scroll) scrollToViewer();
  }
  function openGaia(id, scroll) {
    if (!id || (!QBYID[id] && !TRACES[id])) return;
    viewer.src = "gaia"; viewer.run = null; viewer.id = id;
    renderViewer(); markRows(); setHash();
    if (scroll) scrollToViewer();
  }
  var openTrace = openGaia; // charts and the failure list open GAIA items

  function runLabel(r) {
    return r.label + " · " + (isNum(r.accuracy) ? fmtPct(r.accuracy, 0) : "—") + " · " + r.run_id;
  }

  function renderViewer() {
    var root = clear($("#viewer"));
    if (!hasPractice() && !hasGaia()) {
      root.appendChild(empty("No trajectories yet", "Every task the agent solves writes a step-by-step trace to runs/traces/<run_id>/<task_id>.json; they appear here.", "python -m evals.run_practice"));
      return;
    }
    if (!viewer.src || (viewer.src === "practice" && !hasPractice()) || (viewer.src === "gaia" && !hasGaia())) setDefaultRef();
    var seq = ++renderSeq;
    var bar = h("div", { class: "viewer-bar" });

    if (hasPractice() && hasGaia()) {
      var seg = h("div", { class: "seg", role: "group", "aria-label": "Which set to browse" });
      [["practice", "Practice set"], ["gaia", PUBLIC ? "GAIA run · skeleton" : "GAIA run"]].forEach(function (o) {
        var b = h("button", { type: "button", "aria-pressed": viewer.src === o[0] ? "true" : "false" }, o[1]);
        b.addEventListener("click", function () {
          if (viewer.src === o[0]) return;
          if (o[0] === "gaia") openGaia(firstGaiaId(), false);
          else { var run = prac.run || PR.default_run; openPractice(run, firstPracticeId(run), false); }
        });
        seg.appendChild(b);
      });
      bar.appendChild(seg);
    }

    var list;
    if (viewer.src === "practice") {
      if (PRUNS.length > 1) {
        var rsel = h("select", { class: "run-sel", "aria-label": "Practice run" });
        PRUNS.forEach(function (r) {
          var o = h("option", { value: r.run_id }, runLabel(r));
          if (r.run_id === viewer.run) o.selected = true;
          rsel.appendChild(o);
        });
        rsel.addEventListener("change", function () {
          var keep = practiceItem(rsel.value, viewer.id) ? viewer.id : firstPracticeId(rsel.value);
          openPractice(rsel.value, keep, false);
        });
        bar.appendChild(rsel);
      }
      list = practiceList(viewer.run).map(function (it) {
        var t = PTASKS[it.id] || {};
        var mark = it.correct === true ? "✓ " : it.correct === false ? "✗ " : "· ";
        return { id: it.id, label: mark + it.id + " · " + shortQ(t.question, 64) + (hasPracticeTrace(viewer.run, it.id) ? "" : " (no trace)") };
      });
    } else {
      var qs = QUESTIONS.length ? QUESTIONS : Object.keys(TRACES).map(function (k, i) { return { n: i + 1, task_id: k, question: TRACES[k].question, status: "answered" }; });
      list = qs.map(function (q) {
        return { id: q.task_id, label: "Q" + q.n + " · " + (STATUS[q.status] || "") + " · " + qLabel(q, 70) + (TRACES[q.task_id] ? "" : " (no trace)") };
      });
    }
    var idx = list.findIndex(function (x) { return x.id === viewer.id; });
    var sel = h("select", { "aria-label": viewer.src === "practice" ? "Choose a practice task" : "Choose a GAIA question" });
    list.forEach(function (x) {
      var o = h("option", { value: x.id }, x.label);
      if (x.id === viewer.id) o.selected = true;
      sel.appendChild(o);
    });
    function go(id) { if (viewer.src === "practice") openPractice(viewer.run, id, false); else openGaia(id, false); }
    sel.addEventListener("change", function () { go(sel.value); });
    var prev = h("button", { type: "button", class: "btn", "aria-label": "Previous", disabled: idx <= 0 }, icon(I.prev), h("span", { class: "btn-t" }, "Prev"));
    var next = h("button", { type: "button", class: "btn", "aria-label": "Next", disabled: idx < 0 || idx >= list.length - 1 }, h("span", { class: "btn-t" }, "Next"), icon(I.next));
    prev.addEventListener("click", function () { if (idx > 0) go(list[idx - 1].id); });
    next.addEventListener("click", function () { if (idx < list.length - 1) go(list[idx + 1].id); });
    var expandBtn = h("button", { type: "button", class: "btn ghost" }, icon(I.expand), h("span", { class: "btn-t" }, "Expand all"));
    var collapseBtn = h("button", { type: "button", class: "btn ghost" }, icon(I.collapse), h("span", { class: "btn-t" }, "Collapse all"));
    bar.appendChild(sel);
    bar.appendChild(h("div", { class: "bar-btns" }, prev, next, expandBtn, collapseBtn));
    root.appendChild(bar);

    var bodyBox = h("div", { "aria-live": "polite" });
    root.appendChild(bodyBox);
    var ref = { src: viewer.src, run: viewer.run, id: viewer.id };
    getTrace(ref, function (tr) {
      if (seq !== renderSeq) return;
      clear(bodyBox);
      if (tr === undefined) { bodyBox.appendChild(h("div", { class: "card pad muted" }, "Loading the trajectory…")); return; }
      renderTraceBody(bodyBox, ref, tr && !tr._error ? tr : null, tr && tr._error, expandBtn, collapseBtn);
    });
  }

  function practiceHead(ref, tr) {
    var it = practiceItem(ref.run, ref.id) || {};
    var t = PTASKS[ref.id] || {};
    var run = PRUN[ref.run] || {};
    var head = h("div", { class: "card pad" });
    var ansCls = it.correct === true ? " ok" : it.correct === false ? " bad" : "";
    var answers = h("div", { class: "tr-answer" },
      h("span", { class: "lbl" }, "Agent answered"),
      it.answer ? h("span", { class: "val" + ansCls }, it.answer) : h("span", { class: "val none" }, it.skipped ? "skipped" : "no answer"),
      h("span", { class: "lbl", style: "margin-top:6px" }, "Gold answer"),
      h("span", { class: "val gold" }, it.gold == null ? "—" : it.gold));
    head.appendChild(h("div", { class: "tr-head" },
      h("div", null,
        h("div", { class: "tr-id" }, verdictPill(it), h("span", null, ref.id), attChip(t.attachment), h("span", null, "· " + (run.label || "") + " · " + ref.run)),
        h("p", { class: "tr-q" }, t.question || (tr && tr.question) || "(question text not recorded)"),
        (it.skills || []).length ? h("div", { class: "skills" }, it.skills.map(function (k) { return h("span", { class: "skill" }, k); })) : null),
      answers));
    if (it.correct === true && it.correct_strict === false)
      head.appendChild(h("p", { class: "note", style: "margin-top:12px" }, icon(I.info, 14), h("span", null, "Correct under GAIA's normalized match, but not under the course's strict string match: right content, wrong shape.")));
    if (t.trap) head.appendChild(h("p", { class: "derive" }, h("b", null, "How the gold answer is known. "), t.trap));
    if (it.error) head.appendChild(h("div", { class: "callout err", style: "margin-top:14px" }, textNodes(it.error)));
    return { head: head, q: { steps: it.steps, seconds: it.seconds, tools: it.tools || {} } };
  }

  function gaiaHead(ref, tr) {
    var q = QBYID[ref.id] || { n: 0, task_id: ref.id, question: (tr || {}).question, answer: (tr || {}).final_answer, status: "answered", tools: {} };
    var head = h("div", { class: "card pad" });
    if (PUBLIC) {
      head.appendChild(h("div", { class: "tr-head" },
        h("div", null,
          h("div", { class: "tr-id" }, h("span", { class: "qn" }, "Q" + q.n), pill(q.status), h("span", null, "task " + ref.id), attChip(q.attachment), tr && tr.run_id ? h("span", null, "· " + tr.run_id) : null),
          h("p", { class: "redact" }, icon(I.lock, 15), h("span", null, "GAIA item: question, answer, reasoning, code and tool output are withheld under the ", h("a", { href: "#policy" }, "contamination policy"), ". The skeleton below keeps each step's type, tools, tokens, timing and error type."))),
        (q.tags || []).length ? h("div", { class: "tr-answer" }, h("span", { class: "lbl" }, "Failure tags"), h("div", { class: "tags" }, q.tags.map(function (t) { return h("span", { class: "tag" + (INFO_TAGS[t] ? " info" : "") }, humanTag(t)); }))) : null));
      if (q.error_type) head.appendChild(h("div", { class: "callout err", style: "margin-top:14px" }, q.error_type));
    } else {
      head.appendChild(h("div", { class: "tr-head" },
        h("div", null,
          h("div", { class: "tr-id" }, h("span", { class: "qn" }, "Q" + q.n), pill(q.status), h("span", null, ref.id), attChip(q.attachment), tr && tr.run_id ? h("span", null, "· " + tr.run_id) : null),
          h("p", { class: "tr-q" }, q.question || "(question text not recorded)")),
        h("div", { class: "tr-answer" }, h("span", { class: "lbl" }, "Submitted answer"),
          q.answer ? h("span", { class: "val" }, q.answer) : h("span", { class: "val none" }, "no answer"))));
      if (q.error) head.appendChild(h("div", { class: "callout err", style: "margin-top:14px" }, textNodes(q.error)));
    }
    return { head: head, q: q };
  }

  function renderTraceBody(root, ref, tr, loadError, expandBtn, collapseBtn) {
    var made = ref.src === "practice" ? practiceHead(ref, tr) : gaiaHead(ref, tr);
    var head = made.head, q = made.q;
    var totals = (tr && tr.totals) || {};
    var tools = Object.keys(q.tools || {});
    head.appendChild(h("div", { class: "tr-stats" },
      h("span", null, h("b", null, String(totals.steps || q.steps || 0)), " steps"),
      totals.model_calls ? h("span", null, h("b", null, String(totals.model_calls)), " model calls") : null,
      totals.sub_agent_calls ? h("span", null, h("b", null, String(totals.sub_agent_calls)), totals.sub_agent_calls === 1 ? " delegation" : " delegations") : null,
      totals.step_errors ? h("span", null, h("b", null, String(totals.step_errors)), totals.step_errors === 1 ? " step error" : " step errors") : null,
      h("span", { title: totals.token_usage_complete === false ? "The provider did not report usage for every step, so these totals are a lower bound." : null }, h("b", null, fmtTok(totals.input_tokens)), " in · ", h("b", null, fmtTok(totals.output_tokens)), " out tokens" + (totals.token_usage_complete === false ? " (lower bound)" : "")),
      h("span", null, h("b", null, fmtSec(q.seconds != null ? q.seconds : tr && tr.seconds)), " wall time"),
      tools.length ? h("span", { class: "tools" }, tools.map(function (t) { return toolChip(t, q.tools[t]); })) : null
    ));

    var redacted = !!(tr && tr.redacted);
    expandBtn.disabled = collapseBtn.disabled = redacted || !tr;
    if (!tr) {
      root.appendChild(head);
      var why = loadError ? "The trajectory file could not be loaded (" + loadError + "). Pages served from file:// cannot fetch it; open the site over http."
        : "Only the answer line was recorded; the runner saved no trajectory for this task.";
      root.appendChild(h("div", { style: "margin-top:16px" }, empty(loadError ? "Trajectory not loaded" : "No trace for this task", why)));
      return;
    }
    var steps = tr.steps || [];
    var durs = steps.map(function (st) { return isNum(st.duration_s) && st.type !== "task" && st.type !== "final" ? st.duration_s : 0; });
    var durTotal = durs.reduce(function (a, b) { return a + b; }, 0);
    if (durTotal > 0) {
      var bar = h("div", { class: "wf-bar", role: "group", "aria-label": "Time per step" });
      steps.forEach(function (st, i) {
        if (!durs[i]) return;
        var label = stepName(st) + ", " + fmtSec(st.duration_s) + (st.error ? ", error" : "") + ((st.sub_agents || []).length ? ", delegated" : "");
        var seg = h("button", { type: "button", class: "wf-seg " + st.type + (st.is_final_answer ? " final" : "") + (st.error ? " err" : ""), style: "flex:" + Math.max(durs[i], durTotal * 0.012) + " 1 0", "aria-label": label }, durs[i] / durTotal > 0.07 ? h("span", null, st.type === "planning" ? "P" : String(stepNo(st))) : null);
        bindTip(seg, label);
        seg.addEventListener("click", function () {
          var el = document.getElementById("step-" + i);
          if (!el) return;
          var d = el.querySelector("details");
          if (d) d.open = true;
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        });
        bar.appendChild(seg);
      });
      head.appendChild(h("div", { class: "waterfall" },
        h("div", { class: "wf-label" }, h("span", null, "Where the time went"), h("span", { class: "num", style: "text-transform:none;letter-spacing:0" }, fmtSec(durTotal) + " across " + plural(durs.filter(Boolean).length, "model call"))),
        bar,
        h("div", { class: "legend" },
          h("span", null, h("i", { style: "background:var(--plan)" }), "plan"),
          h("span", null, h("i", { style: "background:var(--accent)" }), "action"),
          h("span", null, h("i", { style: "background:var(--ok)" }), "final answer"),
          h("span", null, h("i", { style: "background:var(--err)" }), "error"))
      ));
    }
    root.appendChild(head);

    if (!steps.length) {
      root.appendChild(h("div", { style: "margin-top:16px" }, empty("Empty trace", "The run recorded no steps for this task.")));
      return;
    }
    var maxTok = Math.max.apply(null, steps.map(function (st) { return (st.input_tokens || 0) + (st.output_tokens || 0); }).concat([1]));
    var ol = h("ol", { class: "timeline" + (redacted ? " skel" : "") });
    if (redacted) {
      steps.forEach(function (st, i) { ol.appendChild(renderSkelStep(st, i, maxTok, 0)); });
      root.appendChild(ol);
      return;
    }
    var actions = steps.filter(function (st) { return st.type === "action"; }).length;
    var many = actions > 5;
    var firstAction = steps.findIndex(function (st) { return st.type === "action"; });
    steps.forEach(function (st, i) {
      ol.appendChild(renderStep(st, i, maxTok, defaultOpen(st, i, many, firstAction), 0));
    });
    root.appendChild(ol);
    var hidden = Array.prototype.filter.call(ol.children, function (li) { var d = li.querySelector("details"); return d && !d.open && !li.classList.contains("task"); }).length;
    if (hidden) root.appendChild(h("p", { class: "note" }, icon(I.info, 14), h("span", null, plural(hidden, "routine step") + " collapsed. Errors, delegations, the first action and the answer stay open.")));
    expandBtn.onclick = function () { ol.querySelectorAll("details").forEach(function (d) { d.open = true; }); };
    collapseBtn.onclick = function () { ol.querySelectorAll("details").forEach(function (d) { d.open = false; }); };
  }

  function stepNo(st) { return st.step_number || st.index; }
  function stepName(st) {
    if (st.type === "planning") return "Plan";
    if (st.type === "final") return "Result";
    if (st.type === "task") return "Task";
    return "Step " + stepNo(st) + (st.is_final_answer ? " · answer" : "");
  }
  function defaultOpen(st, i, many, firstAction) {
    if (st.type === "task") return false;
    if (!many) return true;
    if (st.type === "final" || st.error || st.is_final_answer || (st.sub_agents || []).length) return true;
    return i === firstAction;
  }

  function stepHeader(st, maxTok, depth) {
    var tokIn = st.input_tokens || 0, tokOut = st.output_tokens || 0;
    var kind = st.type === "action" ? (st.is_final_answer ? "Answer" : "Action")
      : ({ planning: "Plan", final: "Result", task: "Task" }[st.type] || st.type);
    var title = st.type === "planning" ? "Planning"
      : st.type === "task" ? (depth ? "Task from the manager" : "Task given to the agent")
      : st.type === "final" ? (st.error ? "Run ended with an error" : "Final answer")
      : "Step " + stepNo(st);
    var subs = st.sub_agents || [];
    return [
      h("span", { class: "st-title" }, h("span", { class: "st-kind" }, kind), title,
        st.error ? h("span", { class: "st-kind e", title: st.error_type || null }, st.error_type ? String(st.error_type).replace(/^Agent/, "") : "error") : null,
        subs.length ? h("span", { class: "st-kind d" }, "delegates") : null),
      (st.tools || []).length ? h("span", { class: "tools" }, uniqCounts(st.tools).map(function (p) { return toolChip(p[0], p[1]); })) : null,
      h("span", { class: "st-meta" },
        isNum(st.duration_s) ? h("span", null, fmtSec(st.duration_s)) : null,
        tokIn || tokOut ? h("span", { title: fmtInt(tokIn) + " input tokens, " + fmtInt(tokOut) + " output tokens" }, fmtTok(tokIn) + " in · " + fmtTok(tokOut) + " out") : null,
        tokIn || tokOut ? h("span", { class: "tokbar", "aria-hidden": "true" },
          h("span", { class: "in", style: "width:" + (tokIn / maxTok * 100) + "%" }),
          h("span", { class: "out", style: "width:" + (tokOut / maxTok * 100) + "%" })) : null)
    ];
  }
  function stepClass(st) {
    var isAnswer = st.type === "final" || (st.type === "action" && st.is_final_answer);
    return "st " + st.type + (isAnswer ? " answer" : "") + (st.error ? " has-err" : "");
  }
  function stepNode(st) {
    return st.type === "planning" ? "P" : st.type === "task" ? "T" : st.type === "final" ? (st.error ? "!" : "✓") : String(stepNo(st));
  }

  /* Public GAIA skeleton: the step's header line only, nothing to expand. */
  function renderSkelStep(st, i, maxTok, depth) {
    var card = h("div", { class: "st-card" }, h("div", { class: "st-line" }, stepHeader(st, maxTok, depth)));
    (st.sub_agents || []).forEach(function (run) {
      var steps = run.steps || [];
      var sm = Math.max.apply(null, steps.map(function (x) { return (x.input_tokens || 0) + (x.output_tokens || 0); }).concat([1]));
      var box = h("div", { class: "deleg skel-deleg" }, h("div", { class: "deleg-head" },
        h("span", { class: "deleg-agent" }, "↳ delegated to ", h("b", null, run.agent)),
        h("span", { class: "muted" }, [plural(steps.filter(function (x) { return x.type === "action"; }).length, "step"), isNum(run.seconds) ? fmtSec(run.seconds) : null].filter(Boolean).join(" · "))));
      var ol = h("ol", { class: "timeline nested skel" });
      steps.forEach(function (x, j) { if (x.type !== "task") ol.appendChild(renderSkelStep(x, j, sm, depth + 1)); });
      box.appendChild(ol);
      card.appendChild(box);
    });
    return h("li", { class: stepClass(st), id: depth ? null : "step-" + i }, h("span", { class: "st-node", "aria-hidden": "true" }, stepNode(st)), card);
  }

  function renderStep(st, i, maxTok, open, depth) {
    var cls = stepClass(st);
    var chev = icon(I.chev, 14);
    chev.setAttribute("class", "chev");
    var subs = st.sub_agents || [];
    var summary = h("summary", null, chev, stepHeader(st, maxTok, depth));
    var bodyEl = h("div", { class: "st-body" });
    if (st.type === "planning") {
      if (st.model_output) bodyEl.appendChild(blk("Plan", renderPlan(st.model_output)));
    } else if (st.type === "task") {
      if (st.model_output) bodyEl.appendChild(h("div", { class: "thought" }, textNodes(st.model_output)));
    } else if (st.type === "final") {
      if (st.model_output != null && st.model_output !== "") bodyEl.appendChild(blk("Answer returned", h("pre", { class: "code" }, textNodes(st.model_output))));
      else if (!st.error) bodyEl.appendChild(h("p", { class: "muted" }, "The run ended without an answer."));
    } else {
      var th = thoughtOf(st.model_output);
      if (th) bodyEl.appendChild(blk("Thought", h("div", { class: "thought" }, textNodes(th))));
      if (st.code) bodyEl.appendChild(blk("Code", h("pre", { class: "code" }, highlight(String(st.code))), "python"));
      if (!st.code) {
        (st.tool_calls || []).forEach(function (tc) {
          bodyEl.appendChild(blk("Tool call · " + tc.name, h("pre", { class: "code" }, textNodes(tc.arguments == null ? "" : String(tc.arguments)))));
        });
      }
      subs.forEach(function (run) { bodyEl.appendChild(renderSubRun(run, depth)); });
      if (st.observation != null && st.observation !== "") bodyEl.appendChild(blk("Observation", renderObservation(st.observation)));
      else if (st.action_output != null && st.action_output !== "") bodyEl.appendChild(blk("Returned", h("pre", { class: "obs" }, textNodes(st.action_output))));
      if (st.images) bodyEl.appendChild(h("p", { class: "muted", style: "font-size:12.5px" }, plural(st.images, "image") + " passed to the model with this observation (not stored in the trace)."));
    }
    if (st.error) bodyEl.appendChild(blk("Error" + (st.error_type ? " · " + st.error_type : ""), h("div", { class: "callout err" }, textNodes(st.error))));
    if (!bodyEl.childNodes.length) bodyEl.appendChild(h("p", { class: "muted" }, "Nothing recorded for this step."));
    var det = h("details", null, summary, bodyEl);
    det.open = open;
    return h("li", { class: cls, id: depth ? null : "step-" + i }, h("span", { class: "st-node", "aria-hidden": "true" }, stepNode(st)), det);
  }

  /* A managed agent's run, nested inside the parent step that called it. */
  function renderSubRun(run, depth) {
    var steps = run.steps || [];
    var actions = steps.filter(function (s) { return s.type === "action"; }).length;
    var maxTok = Math.max.apply(null, steps.map(function (s) { return (s.input_tokens || 0) + (s.output_tokens || 0); }).concat([1]));
    var box = h("div", { class: "deleg" });
    box.appendChild(h("div", { class: "deleg-head" },
      h("span", { class: "deleg-agent" }, "↳ delegated to ", h("b", null, run.agent)),
      h("span", { class: "muted" }, [plural(actions, "step"), isNum(run.seconds) ? fmtSec(run.seconds) : null, run.partial ? "last call only" : null].filter(Boolean).join(" · "))));
    if (run.task) box.appendChild(h("div", { class: "deleg-task" }, textNodes(run.task)));
    if (steps.length) {
      var ol = h("ol", { class: "timeline nested" });
      steps.forEach(function (s, i) {
        if (s.type === "task") return; // the task is shown above
        ol.appendChild(renderStep(s, i, maxTok, !!s.error || s.is_final_answer, depth + 1));
      });
      box.appendChild(ol);
    }
    if (run.elided_steps) box.appendChild(h("p", { class: "muted", style: "font-size:12.5px" }, plural(run.elided_steps, "deeper step") + " not drawn."));
    if (run.final_answer != null && run.final_answer !== "") box.appendChild(blk("Returned to the manager", h("pre", { class: "obs" }, textNodes(run.final_answer))));
    return box;
  }

  function uniqCounts(arr) {
    var out = [], idx = {};
    arr.forEach(function (t) { if (idx[t] === undefined) { idx[t] = out.length; out.push([t, 0]); } out[idx[t]][1]++; });
    return out;
  }

  /* ------------------------------------------------------------------ charts */

  function niceTicks(max, target) {
    if (!(max > 0)) return [0, 1];
    var raw = max / (target || 4);
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (m) { return m * p; }).find(function (v) { return v >= raw; });
    var ticks = [];
    for (var v = 0; v <= max + step * 0.001; v += step) ticks.push(+v.toFixed(6));
    if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(6));
    return ticks;
  }

  /* Axis ticks for durations: round seconds below two minutes, whole minutes above. */
  function timeTicks(max, target) {
    var steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    var raw = Math.max(max, 1) / (target || 4);
    var step = steps.find(function (v) { return v >= raw; }) || 3600;
    var ticks = [];
    for (var v = 0; v < max + step; v += step) ticks.push(v);
    if (ticks.length < 2) ticks.push(step);
    return ticks;
  }
  function fmtTick(sec) {
    if (sec === 0) return "0";
    if (sec < 60) return sec + "s";
    var m = Math.floor(sec / 60), r = sec % 60;
    return m + "m" + (r ? r + "s" : "");
  }

  var drawers = [];
  function chartCard(title, sub, draw, legend) {
    var box = h("div", { class: "chart" });
    var card = h("figure", { class: "card pad fig", style: "margin:0" }, h("h3", null, title), h("p", { class: "sub" }, sub), box, legend || null);
    drawers.push({ box: box, draw: draw });
    return card;
  }
  function redrawCharts() {
    drawers.forEach(function (d) {
      var w = Math.max(260, Math.floor(d.box.clientWidth));
      if (d.w === w) return;
      d.w = w;
      clear(d.box).appendChild(d.draw(w));
    });
  }
  function statusLegend(statuses) {
    return h("div", { class: "legend" }, statuses.map(function (k) {
      return h("span", null, h("i", { style: "background:var(--" + { answered: "ok", error: "err", timeout: "warn", empty: "muted" }[k] + ")" }), STATUS[k]);
    }));
  }
  function msg(w, text) {
    return s("svg", { viewBox: "0 0 " + w + " 80", role: "img", "aria-label": text }, s("text", { x: w / 2, y: 44, "text-anchor": "middle" }, text));
  }

  function drawTools(w) {
    var rows = AN.tool_usage || [];
    if (!rows.length) return msg(w, "No tool calls recorded");
    var rowH = 38, top = 4, H = top + rows.length * rowH;
    var max = Math.max.apply(null, rows.map(function (r) { return r.calls; }));
    var barMax = w;
    var svg = s("svg", { viewBox: "0 0 " + w + " " + H, role: "img", "aria-label": "Tool calls per tool: " + rows.map(function (r) { return r.name + " " + r.calls; }).join(", ") });
    rows.forEach(function (r, i) {
      var y = top + i * rowH;
      var bw = Math.max(2, r.calls / max * barMax);
      var g = s("g", { tabindex: 0 });
      g.appendChild(s("text", { x: 0, y: y + 12, class: "mono lbl-ink" }, r.name));
      g.appendChild(s("rect", { x: 0, y: y + 18, width: barMax, height: 10, rx: 3, class: "grid", fill: "none", "stroke-opacity": 0 }));
      g.appendChild(s("rect", { x: 0, y: y + 18, width: bw, height: 10, rx: 3, class: "bar " + toolClass(r.name) }));
      g.appendChild(s("text", { x: w, y: y + 12, "text-anchor": "end" }, plural(r.calls, "call") + " · " + plural(r.questions, "question")));
      bindTip(g, r.name + ": " + plural(r.calls, "call") + " across " + plural(r.questions, "question"));
      svg.appendChild(g);
    });
    return svg;
  }

  function drawStepsHist(w) {
    var qs = QUESTIONS.filter(function (q) { return q.steps; });
    if (!qs.length) return msg(w, "No step counts recorded");
    var maxS = Math.max.apply(null, qs.map(function (q) { return q.steps; }));
    var keys = ["answered", "empty", "timeout", "error"];
    var bins = [];
    for (var i = 1; i <= maxS; i++) bins.push({ k: i, c: { answered: 0, empty: 0, timeout: 0, error: 0 }, n: 0, ids: [] });
    qs.forEach(function (q) { var b = bins[q.steps - 1]; b.c[q.status]++; b.n++; b.ids.push("Q" + q.n); });
    var H = 210, L = 28, R = 8, T = 10, B = 28;
    var maxN = Math.max.apply(null, bins.map(function (b) { return b.n; }));
    var ticks = niceTicks(maxN, 4).filter(function (t) { return Math.round(t) === t; });
    var top = ticks[ticks.length - 1];
    var iw = w - L - R, ih = H - T - B;
    var bw = iw / bins.length;
    var svg = s("svg", { viewBox: "0 0 " + w + " " + H, role: "img", "aria-label": "Histogram of steps per question" });
    ticks.forEach(function (t) {
      var y = T + ih - t / top * ih;
      svg.appendChild(s("line", { x1: L, x2: w - R, y1: y, y2: y, class: "grid" }));
      svg.appendChild(s("text", { x: L - 6, y: y + 4, "text-anchor": "end" }, String(t)));
    });
    var every = Math.ceil(bins.length / Math.floor(iw / 22));
    bins.forEach(function (b, i) {
      var x = L + i * bw + Math.min(4, bw * 0.15), width = Math.max(2, bw - 2 * Math.min(4, bw * 0.15));
      var y0 = T + ih;
      var g = s("g", { tabindex: b.n ? 0 : null });
      keys.forEach(function (k) {
        if (!b.c[k]) return;
        var hh = b.c[k] / top * ih;
        y0 -= hh;
        g.appendChild(s("rect", { x: x, y: y0, width: width, height: Math.max(0, hh - 1), rx: 2, class: "s-" + k }));
      });
      if (b.n) bindTip(g, plural(b.k, "step") + ": " + plural(b.n, "question") + " (" + b.ids.join(", ") + ")");
      svg.appendChild(g);
      if (i % every === 0 || i === bins.length - 1) svg.appendChild(s("text", { x: L + i * bw + bw / 2, y: H - 10, "text-anchor": "middle" }, String(b.k)));
    });
    svg.appendChild(s("line", { x1: L, x2: w - R, y1: T + ih, y2: T + ih, class: "axis" }));
    return svg;
  }

  function drawLatency(w) {
    var qs = QUESTIONS.filter(function (q) { return isNum(q.seconds); });
    if (!qs.length) return msg(w, "No timings recorded");
    var L = 12, R = 16, T = 34, r = 7;
    var max = Math.max.apply(null, qs.map(function (q) { return q.seconds; }));
    var ticks = timeTicks(max, w < 420 ? 3 : 5);
    var top = ticks[ticks.length - 1];
    var iw = w - L - R;
    function X(v) { return L + v / top * iw; }
    // Beeswarm: place each dot on the lowest lane where it does not overlap a neighbour.
    var placed = [];
    qs.slice().sort(function (a, b) { return a.seconds - b.seconds; }).forEach(function (q) {
      var x = X(q.seconds), lane = 0;
      while (placed.some(function (p) { return p.lane === lane && Math.abs(p.x - x) < 2 * r + 1; })) lane++;
      placed.push({ q: q, x: x, lane: lane });
    });
    var lanes = Math.max.apply(null, placed.map(function (p) { return p.lane; })) + 1;
    var plotH = Math.max(60, lanes * (2 * r + 2) + 16);
    var H = T + plotH + 26;
    var base = T + plotH - r - 4;
    var svg = s("svg", { viewBox: "0 0 " + w + " " + H, role: "img", "aria-label": "Time per question, one dot per question" });
    ticks.forEach(function (t) {
      svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: T, y2: T + plotH, class: "grid" }));
      svg.appendChild(s("text", { x: X(t), y: H - 8, "text-anchor": "middle" }, fmtTick(t)));
    });
    var md = (AN.seconds || {}).median, p90 = (AN.seconds || {}).p90;
    var close = isNum(md) && isNum(p90) && Math.abs(X(p90) - X(md)) < 120;
    [[md, "median " + fmtSec(md), 0], [p90, "p90 " + fmtSec(p90), close ? 14 : 0]].forEach(function (rl) {
      if (!isNum(rl[0])) return;
      var x = X(rl[0]);
      svg.appendChild(s("line", { x1: x, x2: x, y1: T - 16 + rl[2], y2: T + plotH, class: "rule" }));
      var right = x > w - 110;
      svg.appendChild(s("text", { x: x + (right ? -5 : 5), y: T - 8 + rl[2], "text-anchor": right ? "end" : "start", class: "lbl-ink" }, rl[1]));
    });
    placed.forEach(function (p) {
      var c = s("circle", { cx: p.x, cy: base - p.lane * (2 * r + 2), r: r, class: "dot s-" + p.q.status, tabindex: 0, role: "button", "aria-label": "Question " + p.q.n + ", " + fmtSec(p.q.seconds) + ", " + STATUS[p.q.status] + ". Open trajectory." });
      bindTip(c, "Q" + p.q.n + " · " + fmtSec(p.q.seconds) + " · " + STATUS[p.q.status] + " — " + qLabel(p.q, 60));
      c.addEventListener("click", function () { openTrace(p.q.task_id, true); });
      c.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openTrace(p.q.task_id, true); } });
      svg.appendChild(c);
    });
    svg.appendChild(s("line", { x1: L, x2: w - R, y1: T + plotH, y2: T + plotH, class: "axis" }));
    return svg;
  }

  function drawScatter(w) {
    var qs = QUESTIONS.filter(function (q) { return isNum(q.seconds) && q.tokens; });
    if (qs.length < 2) return msg(w, "Needs tokens and timings for 2+ questions");
    var H = 230, L = 44, R = 12, T = 12, B = 34;
    var maxX = Math.max.apply(null, qs.map(function (q) { return q.tokens; }));
    var maxY = Math.max.apply(null, qs.map(function (q) { return q.seconds; }));
    var tx = niceTicks(maxX, w < 420 ? 3 : 4), ty = timeTicks(maxY, 4);
    var topX = tx[tx.length - 1], topY = ty[ty.length - 1];
    var iw = w - L - R, ih = H - T - B;
    function X(v) { return L + v / topX * iw; }
    function Y(v) { return T + ih - v / topY * ih; }
    var svg = s("svg", { viewBox: "0 0 " + w + " " + H, role: "img", "aria-label": "Tokens against time, one dot per question" });
    ty.forEach(function (t) {
      svg.appendChild(s("line", { x1: L, x2: w - R, y1: Y(t), y2: Y(t), class: "grid" }));
      svg.appendChild(s("text", { x: L - 6, y: Y(t) + 4, "text-anchor": "end" }, fmtTick(t)));
    });
    tx.forEach(function (t) {
      svg.appendChild(s("text", { x: X(t), y: H - 16, "text-anchor": "middle" }, t ? fmtTok(t) : "0"));
    });
    svg.appendChild(s("text", { x: L + iw / 2, y: H - 2, "text-anchor": "middle" }, "tokens (input + output)"));
    svg.appendChild(s("line", { x1: L, x2: w - R, y1: T + ih, y2: T + ih, class: "axis" }));
    qs.forEach(function (q) {
      var c = s("circle", { cx: X(q.tokens), cy: Y(q.seconds), r: 6.5, class: "dot s-" + q.status, tabindex: 0, role: "button", "aria-label": "Question " + q.n + ", " + fmtTok(q.tokens) + " tokens, " + fmtSec(q.seconds) + ". Open trajectory." });
      bindTip(c, "Q" + q.n + " · " + fmtTok(q.tokens) + " tokens · " + fmtSec(q.seconds) + " · " + plural(q.steps || 0, "step"));
      c.addEventListener("click", function () { openTrace(q.task_id, true); });
      c.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openTrace(q.task_id, true); } });
      svg.appendChild(c);
    });
    return svg;
  }

  function renderCharts() {
    var root = $("#charts");
    if (!QUESTIONS.length) {
      root.appendChild(empty("No GAIA run to chart yet", "Tool usage, step counts and latency are computed from the GAIA run's answers and traces. Practice-set costs are in Evals and Usage."));
      return;
    }
    var present = ["answered", "empty", "timeout", "error"].filter(function (k) { return (AN.status || {})[k]; });
    var toolLegend = h("div", { class: "legend" },
      h("span", null, h("i", { style: "background:var(--accent)" }), "web · Wikipedia"),
      h("span", null, h("i", { style: "background:var(--media)" }), "files · audio · vision"),
      (AN.tool_usage || []).some(function (r) { return toolClass(r.name) === "other"; }) ? h("span", null, h("i", { style: "background:var(--muted)" }), "sub-agents · other") : null);
    root.appendChild(h("div", { class: "grid-2" },
      chartCard("Tool usage", "Calls per tool across the run, read from the code each step executed.", drawTools, toolLegend),
      chartCard("Steps per question", "How many model calls each question took, coloured by outcome.", drawStepsHist, statusLegend(present)),
      chartCard("Time per question", "One dot per question. Click a dot to open its trajectory.", drawLatency, statusLegend(present)),
      chartCard("Tokens against time", "Cost against latency. Outliers are usually video, audio or a search loop.", drawScatter, statusLegend(present))
    ));
  }

  /* ------------------------------------------------------------------ failures */

  var TAGS = {
    retrieval_miss: "Never found the page or passage that holds the answer.",
    wrong_source: "Relied on a source that was wrong or out of date.",
    reasoning: "Had the facts and drew the wrong conclusion.",
    reasoning_error: "Had the facts and drew the wrong conclusion.",
    format: "Right content, wrong shape for exact match.",
    format_mismatch: "Right content, wrong shape for exact match.",
    answer_format: "Right content, wrong shape for exact match.",
    tool_failure: "A tool raised or returned unusable output.",
    tool_error: "A tool raised or returned unusable output.",
    tool_output_format: "A tool's output needed cleaning before it could be used.",
    timeout: "Hit the per-question time budget.",
    step_limit: "Ran out of steps before answering.",
    max_steps: "Ran out of steps before answering.",
    provider_rate_limit: "The free model endpoint throttled the run.",
    rate_limit: "The free model endpoint throttled the run.",
    model_error: "The model endpoint failed or returned nothing usable.",
    vision: "Depended on reading an image or video frame.",
    vision_uncertainty: "Depended on reading an image or video frame.",
    audio: "Depended on an audio transcription.",
    transcription: "Depended on an audio transcription.",
    hallucination: "Stated something no tool output supports.",
    gave_up: "Stopped without committing to a best guess.",
    no_answer: "Stopped without committing to a best guess.",
    unbounded_paging: "Repeated the same action without changing strategy.",
    loop: "Repeated the same action without changing strategy.",
    misread_question: "Answered a different question from the one asked.",
    guard_block: "The GAIA guard blocked a source, by design.",
    file_handling: "Could not read or interpret the attached file.",
    arithmetic: "Computed the wrong number from the right inputs."
  };

  function tagDesc(E, t) {
    var d = (E.descriptions || {})[t] || TAGS[t];
    if (d) return d;
    var base = String(t).split(":")[0];
    if ((E.descriptions || {})[base] || TAGS[base]) return ((E.descriptions || {})[base] || TAGS[base]) + " (" + String(t).split(":").slice(1).join(":") + ")";
    return "";
  }

  function renderFailures() {
    var root = $("#taxonomy");
    var E = DATA.errors;
    if (!E) {
      root.appendChild(empty("Not run yet", "The error-analysis step reads every GAIA trace and writes runs/error_analysis.json: each doubtful question, its failure tags, and the evidence.", "python -m analysis.error_analysis"));
      return;
    }
    var counts = E.counts || {};
    var tags = Object.keys(counts);
    if (!tags.length && !(E.items || []).length) {
      root.appendChild(empty("No failures tagged", "The error analysis ran and tagged nothing."));
      return;
    }
    var max = Math.max.apply(null, tags.map(function (t) { return counts[t]; }).concat([1]));
    var list = h("ul", { class: "tax-list" });
    tags.forEach(function (t) {
      var d = tagDesc(E, t);
      list.appendChild(h("li", { class: "tax-row" + (INFO_TAGS[t] ? " info" : "") },
        h("div", { class: "top" }, h("span", { class: "name" }, humanTag(t)), h("span", { class: "num muted" }, plural(counts[t], "question"))),
        d ? h("div", { class: "desc" }, d) : null,
        h("div", { class: "bar", "aria-hidden": "true" }, h("i", { style: "width:" + (counts[t] / max * 100) + "%" }))));
    });
    var items = h("div", { class: "card" });
    (E.items || []).forEach(function (it) {
      var qb = it.n ? h("button", { type: "button", class: "linkish qn", "aria-label": "Open trajectory for question " + it.n }, "Q" + it.n) : h("span", { class: "qn" }, "—");
      if (it.n) qb.addEventListener("click", function () { openTrace(it.task_id, true); });
      items.appendChild(h("div", { class: "fail-item" },
        qb,
        h("div", null, it.tags.map(function (t) { return h("span", { class: "tag" + (INFO_TAGS[t] ? " info" : ""), title: tagDesc(E, t) || null }, humanTag(t)); }),
          h("div", { style: "font-size:13.5px" }, it.question ? shortQ(it.question, 140) : h("span", { class: "mono muted" }, "task " + it.task_id, attChip((QBYID[it.task_id] || {}).attachment)))),
        it.evidence ? h("div", { class: "ev" }, textNodes(it.evidence)) : null));
    });
    if (!(E.items || []).length) items.appendChild(h("div", { class: "pad muted" }, "No per-question evidence recorded."));
    if (E.redacted) items.appendChild(h("div", { class: "pad muted", style: "font-size:12.5px" }, "Evidence lines quote the trace, so the public build lists tags only."));
    var head = h("div", { class: "tax-sum" });
    if (isNum(E.n_tasks)) {
      var flagged = (E.items || []).filter(function (it) { return it.tags.some(function (t) { return !INFO_TAGS[t]; }); }).length;
      head.appendChild(h("span", null, h("b", null, String(flagged)), " of " + E.n_tasks + " questions show a failure behaviour"));
      if (isNum(E.clean)) head.appendChild(h("span", null, h("b", null, String(E.clean)), " ran clean"));
    }
    root.appendChild(head);
    root.appendChild(h("div", { class: "tax" }, h("div", { class: "card pad" }, h("h3", { style: "margin-bottom:14px" }, "Tags"), list), items));
    if (E.note) root.appendChild(h("p", { class: "note" }, icon(I.info, 14), h("span", null, E.note)));
  }

  /* ------------------------------------------------------------------ evals */

  function renderEvals() {
    var root = $("#evals-body");
    var EV = DATA.evals;
    if (!EV || !(EV.groups || []).length) {
      root.appendChild(empty("Not run yet", "The eval harness runs the agent on a practice set with its own answer key, then writes evals/results/<run_id>.json per run. Repeats and ablations show up here."));
      return;
    }
    var groups = EV.groups;
    var thead = h("tr", null, ["Configuration", "Set", "Runs", "Accuracy · 95% CI", "Δ vs baseline", "Steps", "Time", "Tokens / item", "Δ tokens"].map(function (t, i) {
      return h("th", { class: i >= 2 && i !== 3 ? "r" : i === 3 ? "r" : null, scope: "col" }, t);
    }));
    var tbody = h("tbody");
    groups.forEach(function (g) {
      var detail = [g.model, g.arch === "multi" ? "multi-agent" : "single", g.verify ? "verify on" : null, g.disabled_tools.length ? "off: " + g.disabled_tools.join(", ") : null].filter(Boolean).join(" · ");
      var acc;
      if (isNum(g.accuracy)) {
        var lo = g.ci95 ? g.ci95[0] : g.accuracy, hi = g.ci95 ? g.ci95[1] : g.accuracy;
        var ci = h("span", { class: "ci", role: "img", "aria-label": "95% interval " + fmtPct(lo, 0) + " to " + fmtPct(hi, 0) },
          h("span", { class: "trk" }), h("span", { class: "rng", style: "left:" + lo * 100 + "%;width:" + Math.max(1, (hi - lo) * 100) + "%" }),
          h("span", { class: "pt", style: "left:" + g.accuracy * 100 + "%" }));
        bindTip(ci, "Mean " + fmtPct(g.accuracy) + (g.runs > 1 ? " (runs ranged " + fmtPct(g.accuracy_min, 0) + "–" + fmtPct(g.accuracy_max, 0) + ")" : "") + (g.ci95 ? "; 95% Wilson interval " + fmtPct(lo, 0) + "–" + fmtPct(hi, 0) : ""));
        acc = h("span", { class: "acc" }, h("span", null, h("b", { class: "num" }, fmtPct(g.accuracy)),
          isNum(g.accuracy_strict) && Math.abs(g.accuracy_strict - g.accuracy) > 0.0005 ? h("span", { class: "noise", title: "Exact string match with no normalization. The gap is answers that were right but formatted differently." }, "strict " + fmtPct(g.accuracy_strict)) : null), ci);
      } else acc = h("span", { class: "muted" }, g.set === "gaia" ? "no key" : "—");
      var d = g.is_baseline ? h("span", { class: "base-tag" }, "baseline") : isNum(g.delta_pp) ? h("span", null, deltaSpan(g.delta_pp, " pp", false), isNum(g.mcnemar_p) ? h("span", { class: "noise", title: "Exact McNemar test on the tasks both configs ran (from evals/ablate.py). Below 0.05 means the difference is unlikely to be chance." }, "McNemar p = " + (g.mcnemar_p < 0.001 ? "<0.001" : g.mcnemar_p.toFixed(g.mcnemar_p < 0.01 ? 3 : 2))) : g.separable === false ? h("span", { class: "noise", title: "The 95% intervals of this config and the baseline overlap, so this difference is not established." }, "within noise") : null) : h("span", { class: "muted" }, "—");
      var dt = isNum(g.delta_tokens_pct) ? deltaSpan(g.delta_tokens_pct, "%", true) : h("span", { class: "muted" }, "—");
      tbody.appendChild(h("tr", { class: g.is_baseline ? "base" : null },
        h("td", { "data-l": "Configuration" }, h("div", { class: "cfg" }, g.label, h("small", null, detail))),
        h("td", { "data-l": "Set" }, g.set),
        h("td", { class: "r", "data-l": "Runs" }, g.runs + " × " + g.items_per_run),
        h("td", { class: "r", "data-l": "Accuracy" }, acc),
        h("td", { class: "r", "data-l": "Δ vs baseline" }, d),
        h("td", { class: "r", "data-l": "Steps" }, isNum(g.mean_steps) ? g.mean_steps.toFixed(1) : "—"),
        h("td", { class: "r", "data-l": "Time" }, fmtSec(g.mean_seconds)),
        h("td", { class: "r", "data-l": "Tokens / item" }, fmtTok(g.tokens_per_item)),
        h("td", { class: "r", "data-l": "Δ tokens" }, dt)));
    });
    root.appendChild(h("div", { class: "card" }, h("table", { class: "ab" }, h("caption", { class: "sr-only" }, "Eval configurations and ablations"), h("thead", null, thead), tbody)));
    root.appendChild(h("p", { class: "note" }, icon(I.info, 14), h("span", null,
      "Accuracy is the GAIA-normalized match, averaged over runs; \u201cstrict\u201d is raw exact match. The bar is a 95% Wilson interval on all judged items pooled. Δ is tested with an exact McNemar test on paired tasks where the ablation report has one; otherwise overlapping intervals are flagged as noise. Δ tokens is cost per item against the baseline.")));

    var skillGroups = groups.filter(function (g) { return (g.by_skill || []).length; });
    if (skillGroups.length) {
      var ssel = h("select", { "aria-label": "Configuration for skill breakdown" });
      skillGroups.forEach(function (g, i) { ssel.appendChild(h("option", { value: String(i) }, g.label)); });
      var sbox = h("div", { class: "chart" });
      var sd = { box: sbox, draw: function (w) { return drawSkills(w, skillGroups[+ssel.value || 0]); } };
      drawers.push(sd);
      ssel.addEventListener("change", function () { sd.w = null; redrawCharts(); });
      root.appendChild(h("div", { class: "card pad", style: "margin-top:16px" },
        h("div", { class: "passk-head" }, h("h3", null, "Accuracy by skill"), ssel),
        h("p", { class: "muted", style: "font-size:12.5px;margin:0 0 10px" }, "Each practice task is tagged with the skills it needs; a task with two skills counts toward both. Pooled over the configuration's runs."),
        sbox));
    }

    var pk = groups.filter(function (g) { return g.passk && g.passk.k.length > 1; });
    var card = h("div", { class: "card pad", style: "margin-top:16px" });
    if (!pk.length) {
      card.appendChild(h("h3", null, "pass@k and pass^k"));
      card.appendChild(h("p", { class: "muted", style: "margin-top:6px" }, "Run one configuration at least twice on the practice set to see how much of its score is reliable and how much is luck."));
      root.appendChild(card);
    } else {
      var sel = h("select", { "aria-label": "Configuration for pass@k" });
      pk.forEach(function (g, i) { sel.appendChild(h("option", { value: String(i) }, g.label + " · " + g.runs + " runs")); });
      var chartBox = h("div", { class: "chart" });
      var caption = h("p", { class: "muted", style: "font-size:12.5px;margin:0 0 10px" });
      card.appendChild(h("div", { class: "passk-head" }, h("h3", null, "pass@k and pass^k"), sel));
      card.appendChild(caption);
      card.appendChild(chartBox);
      card.appendChild(h("div", { class: "explain" },
        h("div", null, h("b", null, "pass@k"), " — the item is solved at least once in k attempts (Chen et al., 2021). It rises with k: the agent's ceiling."),
        h("div", { class: "hat" }, h("b", null, "pass^k"), " — the item is solved on every one of k attempts (Yao et al., 2024, τ-bench). It falls with k: the agent's reliability.")));
      root.appendChild(card);
      var drawPk = function (w) { return drawPassK(w, pk[+sel.value || 0]); };
      var d = { box: chartBox, draw: drawPk };
      drawers.push(d);
      var upd = function () {
        var g = pk[+sel.value || 0];
        caption.textContent = "Per-item unbiased estimates from " + g.passk.runs + " runs over " + g.passk.items + " items, averaged. The gap between the lines at k is the share of the score that depends on luck.";
        d.w = null;
        redrawCharts();
      };
      sel.addEventListener("change", upd);
      upd();
    }

    var runs = EV.runs || [];
    if (runs.length) {
      var rt = h("tbody");
      runs.forEach(function (r) {
        var canOpen = r.set !== "gaia" && PRUN[r.run_id];
        var runCell = canOpen ? h("button", { type: "button", class: "linkish mono", title: "Show this run's tasks and trajectories" }, r.run_id) : h("span", { class: "mono" }, r.run_id);
        if (canOpen) runCell.addEventListener("click", function () { if (prac.select) prac.select(r.run_id, true); });
        rt.appendChild(h("tr", null,
          h("td", { "data-l": "Run" }, runCell),
          h("td", { "data-l": "Config" }, r.label),
          h("td", { "data-l": "Set" }, r.set),
          h("td", { class: "r", "data-l": "Items" }, String(r.n)),
          h("td", { class: "r", "data-l": "Accuracy" }, isNum(r.accuracy) ? fmtPct(r.accuracy) + (isNum(r.correct) ? " (" + r.correct + "/" + r.judged + ")" : "") : "—"),
          h("td", { class: "r", "data-l": "Errors" }, String(r.errors || 0))));
      });
      root.appendChild(h("details", { class: "card", style: "margin-top:16px" },
        h("summary", { class: "pad", style: "cursor:pointer;font-weight:600" }, "All " + plural(runs.length, "eval run")),
        h("table", { class: "ab" }, h("thead", null, h("tr", null, ["Run", "Config", "Set", "Items", "Accuracy", "Errors"].map(function (t, i) { return h("th", { class: i >= 3 ? "r" : null }, t); }))), rt)));
    }
  }

  function deltaSpan(v, unit, invert) {
    var cls = Math.abs(v) < 0.05 ? "flat" : (v > 0) !== !!invert ? "up" : "down";
    return h("span", { class: "delta " + cls }, (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(1) + unit);
  }

  function drawSkills(w, g) {
    var rows = g.by_skill || [];
    if (!rows.length) return msg(w, "No skill tags on these tasks");
    var rowH = 36, H = rows.length * rowH + 4;
    var svg = s("svg", { viewBox: "0 0 " + w + " " + H, role: "img", "aria-label": "Accuracy by skill for " + g.label + ": " + rows.map(function (r) { return r.skill + " " + fmtPct(r.accuracy, 0); }).join(", ") });
    rows.forEach(function (r, i) {
      var y = i * rowH + 4;
      var gr = s("g", { tabindex: 0 });
      gr.appendChild(s("text", { x: 0, y: y + 11, class: "lbl-ink" }, r.skill.replace(/_/g, " ")));
      gr.appendChild(s("text", { x: w, y: y + 11, "text-anchor": "end" }, fmtPct(r.accuracy, 0) + "  ·  " + r.correct + "/" + r.n));
      gr.appendChild(s("rect", { x: 0, y: y + 17, width: w, height: 9, rx: 3, class: "bar-q" }));
      gr.appendChild(s("rect", { x: 0, y: y + 17, width: Math.max(2, r.accuracy * w), height: 9, rx: 3, class: "bar" }));
      bindTip(gr, r.skill + ": " + r.correct + " of " + r.n + " correct");
      svg.appendChild(gr);
    });
    return svg;
  }

  function drawPassK(w, g) {
    var P = g.passk, H = 220, L = 40, R = 16, T = 14, B = 30;
    var iw = w - L - R, ih = H - T - B;
    var ks = P.k;
    function X(k) { return ks.length === 1 ? L + iw / 2 : L + (k - ks[0]) / (ks[ks.length - 1] - ks[0]) * iw; }
    function Y(v) { return T + ih - v * ih; }
    var svg = s("svg", { viewBox: "0 0 " + w + " " + H, role: "img", "aria-label": "pass@k and pass^k for " + g.label + ": " + ks.map(function (k, i) { return "k=" + k + " pass@k " + fmtPct(P.pass_at_k[i], 0) + ", pass^k " + fmtPct(P.pass_hat_k[i], 0); }).join("; ") });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      svg.appendChild(s("line", { x1: L, x2: w - R, y1: Y(t), y2: Y(t), class: "grid" }));
      svg.appendChild(s("text", { x: L - 6, y: Y(t) + 4, "text-anchor": "end" }, t * 100 + "%"));
    });
    ks.forEach(function (k) { svg.appendChild(s("text", { x: X(k), y: H - 10, "text-anchor": "middle" }, "k=" + k)); });
    var band = "M" + ks.map(function (k, i) { return X(k) + "," + Y(P.pass_at_k[i]); }).join("L") + "L" + ks.slice().reverse().map(function (k, i) { return X(k) + "," + Y(P.pass_hat_k[ks.length - 1 - i]); }).join("L") + "Z";
    svg.appendChild(s("path", { d: band, class: "band" }));
    [["pass_at_k", "at"], ["pass_hat_k", "hat"]].forEach(function (pair) {
      var vals = P[pair[0]];
      svg.appendChild(s("path", { d: "M" + ks.map(function (k, i) { return X(k) + "," + Y(vals[i]); }).join("L"), class: "line-" + pair[1] }));
      ks.forEach(function (k, i) {
        var c = s("circle", { cx: X(k), cy: Y(vals[i]), r: 4.5, class: "pt-" + pair[1], tabindex: 0 });
        bindTip(c, (pair[1] === "at" ? "pass@" : "pass^") + k + " = " + fmtPct(vals[i]));
        svg.appendChild(c);
      });
      var lastI = ks.length - 1;
      svg.appendChild(s("text", { x: X(ks[lastI]) - 6, y: Y(vals[lastI]) + (pair[1] === "at" ? -10 : 18), "text-anchor": "end", class: "lbl-ink" }, (pair[1] === "at" ? "pass@k " : "pass^k ") + fmtPct(vals[lastI], 0)));
    });
    return svg;
  }

  /* ------------------------------------------------------------------ practice set (shown in full) */

  function cfgDetail(cfg) {
    cfg = cfg || {};
    var off = cfg.disabled_tools || [];
    return [cfg.model, cfg.arch === "multi" ? "multi-agent" : "single agent", cfg.verify ? "verifier on" : null,
      off.length ? "off: " + off.join(", ") : null].filter(Boolean).join(" · ");
  }

  function renderPractice() {
    var root = $("#practice-body");
    if (!PR || !PRUNS.length) {
      root.appendChild(empty("Practice set not run yet", "Sixteen original tasks with verifiable answers. Each run is graded and every step is recorded; the full trajectories appear here.", "python -m evals.run_practice --max-requests 120"));
      return;
    }
    var state = { filter: "all" };
    var rsel = h("select", { class: "run-sel", "aria-label": "Practice run" });
    PRUNS.forEach(function (r) { rsel.appendChild(h("option", { value: r.run_id }, runLabel(r))); });
    rsel.value = prac.run;
    rsel.addEventListener("change", function () { prac.select(rsel.value); });
    var seg = h("div", { class: "seg", role: "group", "aria-label": "Filter by result" });
    var sumBox = h("div", { class: "psum" });
    var listBox = h("div", { class: "card plist", role: "list", "aria-label": "Practice tasks" });
    root.appendChild(h("div", { class: "controls" }, rsel, seg));
    root.appendChild(sumBox);
    root.appendChild(listBox);
    root.appendChild(h("p", { class: "note" }, icon(I.info, 14), h("span", null,
      "Graded by a port of GAIA's quasi-exact scorer; “format” marks an answer that passes it but fails the course's strict string match. Dots show the same task across repeat runs of this configuration. Click a task for its full trajectory.")));

    function groupRuns(run) {
      var g = (PRUN[run] || {}).group;
      return g ? PRUNS.filter(function (r) { return r.group === g; }) : [PRUN[run]];
    }

    function drawSegs(items) {
      clear(seg);
      var c = { all: items.length, ok: 0, bad: 0 };
      items.forEach(function (it) { if (it.correct === true) c.ok++; else if (it.correct === false) c.bad++; });
      [["all", "All"], ["ok", "Correct"], ["bad", "Wrong"]].forEach(function (o) {
        if (o[0] !== "all" && !c[o[0]]) return;
        var b = h("button", { type: "button", "aria-pressed": state.filter === o[0] ? "true" : "false" }, o[1], h("span", { class: "c" }, c[o[0]]));
        b.addEventListener("click", function () { state.filter = o[0]; draw(); });
        seg.appendChild(b);
      });
    }

    function draw() {
      var run = prac.run, r = PRUN[run] || {}, items = practiceList(run);
      if (state.filter !== "all" && !items.some(function (it) { return state.filter === "ok" ? it.correct === true : it.correct === false; })) state.filter = "all";
      drawSegs(items);
      clear(sumBox);
      var traced = items.filter(function (it) { return hasPracticeTrace(run, it.id); }).length;
      var fmtGap = items.filter(function (it) { return it.correct === true && it.correct_strict === false; }).length;
      var cells = [
        ["Accuracy", isNum(r.accuracy) ? fmtPct(r.accuracy, 0) : "—", isNum(r.correct) ? r.correct + " of " + r.judged + " graded" : ""],
        ["Strict match", isNum(r.accuracy_strict) ? fmtPct(r.accuracy_strict, 0) : "—", fmtGap ? plural(fmtGap, "format miss").replace("misss", "misses") : "no format misses"],
        ["Trajectories", traced + " / " + items.length, traced ? "full step-by-step traces" : "no traces saved"],
        ["Skipped", String(r.skipped || 0), r.skipped ? "budget stop" : "none"]
      ];
      sumBox.appendChild(h("div", { class: "psum-cfg" }, h("span", { class: "eyebrow", style: "margin:0" }, r.label || "run"), h("span", { class: "mono" }, cfgDetail(r.config))));
      sumBox.appendChild(h("div", { class: "psum-k" }, cells.map(function (x) {
        return h("div", null, h("div", { class: "k" }, x[0]), h("div", { class: "v" }, x[1]), h("div", { class: "s" }, x[2]));
      })));

      clear(listBox);
      var reps = groupRuns(run).filter(Boolean);
      var rows = items.filter(function (it) {
        return state.filter === "all" || (state.filter === "ok" ? it.correct === true : it.correct === false);
      });
      rows.forEach(function (it) {
        var t = PTASKS[it.id] || {};
        var vcls = it.skipped ? "skip" : it.correct === true ? "ok" : it.correct === false ? "bad" : "skip";
        var mark = h("span", { class: "vmark " + vcls, "aria-hidden": "true" }, vcls === "ok" ? icon(I.check, 14) : vcls === "bad" ? icon(I.cross, 14) : "–");
        var ans = it.skipped ? h("span", { class: "ans none" }, "skipped")
          : it.answer ? h("span", { class: "ans " + vcls }, it.answer) : h("span", { class: "ans none" }, "no answer");
        var dots = reps.length > 1 ? h("span", { class: "reps", title: "Same task on each repeat run of this configuration" }, reps.map(function (g) {
          var o = practiceItem(g.run_id, it.id) || {};
          var c = o.skipped || o.correct == null ? "skip" : o.correct ? "ok" : "bad";
          return h("i", { class: c + (g.run_id === run ? " cur" : ""), title: g.run_id + ": " + (c === "ok" ? "correct" : c === "bad" ? "wrong" : "not graded") });
        })) : null;
        var btn = h("button", { type: "button", class: "prow", role: "listitem", "data-run": run, "data-id": it.id,
          "aria-label": (vcls === "ok" ? "Correct" : vcls === "bad" ? "Wrong" : "Ungraded") + ": " + it.id + ". " + shortQ(t.question, 90) + ". Open trajectory." },
          mark,
          h("span", { class: "p-main" },
            h("span", { class: "p-top" }, h("span", { class: "mono p-id" }, it.id), attChip(t.attachment), (it.skills || []).slice(0, 3).map(function (k) { return h("span", { class: "skill" }, k); })),
            h("span", { class: "p-q" }, t.question || "")),
          h("span", { class: "p-ans" }, ans,
            it.correct === false && it.gold != null ? h("span", { class: "p-gold" }, "gold ", h("b", null, it.gold)) : null,
            it.correct === true && it.correct_strict === false ? h("span", { class: "p-gold" }, "strict match fails") : null),
          h("span", { class: "p-meta" }, dots,
            h("span", { class: "num" }, [it.steps ? plural(it.steps, "step") : null, it.tokens ? fmtTok(it.tokens) + " tok" : null, isNum(it.seconds) ? fmtSec(it.seconds) : null].filter(Boolean).join(" · ") || "—"),
            hasPracticeTrace(run, it.id) ? null : h("span", { class: "muted" }, "no trace")));
        btn.addEventListener("click", function () { openPractice(run, it.id, true); });
        listBox.appendChild(btn);
      });
      if (!rows.length) listBox.appendChild(h("div", { class: "pad muted" }, "No tasks match."));
      markRows();
    }

    prac.select = function (run, scroll) {
      if (!PRUN[run]) return;
      prac.run = run;
      rsel.value = run;
      draw();
      if (scroll) $("#practice").scrollIntoView({ behavior: "smooth", block: "start" });
    };
    draw();
  }

  /* ------------------------------------------------------------------ usage (runs/usage.jsonl) */

  function renderUsage() {
    var root = $("#usage-body");
    var U = USAGE;
    if (!U || !U.totals || !U.totals.requests) {
      root.appendChild(empty("No model requests logged yet", "The provider layer appends one row per model request to runs/usage.jsonl: served model, latency, tokens, retries, fallbacks and error kinds."));
      return;
    }
    var T = U.totals;
    if (U.all_free === false) {
      root.appendChild(h("div", { class: "callout err", style: "margin-bottom:14px;font-family:var(--sans)" },
        "A model without a :free id served requests: " + U.non_free_models.join(", ") + ". The project budget is $0; check the provider guard."));
    }
    var k = h("div", { class: "kpis", style: "margin-top:0" });
    [
      ["Requests", fmtInt(T.requests), T.failed ? fmtInt(T.failed) + " failed after retries" : "all succeeded"],
      ["Tokens", fmtTok(T.input_tokens + T.output_tokens), fmtTok(T.input_tokens) + " in · " + fmtTok(T.output_tokens) + " out"],
      ["Retries", fmtInt(T.retries), "backoff on 429s and timeouts"],
      ["Fallbacks", fmtInt(T.fallback_served), "requests served by a backup model"],
      ["Latency", fmtSec(T.latency_median), isNum(T.latency_p90) ? "median · p90 " + fmtSec(T.latency_p90) : "median"],
      ["Waited", fmtSec(T.waited_s), "rate limiter + backoff"],
      ["Cost", U.all_free === false ? "check" : "$0", U.all_free === false ? "non-free model used" : "every model id is :free"]
    ].forEach(function (x) { k.appendChild(h("div", { class: "kpi" }, h("div", { class: "k" }, x[0]), h("div", { class: "v" }, x[1]), h("div", { class: "s" }, x[2]))); });
    root.appendChild(k);

    var sc = U.scopes || {};
    var scopeNames = { gaia: "GAIA run", practice: "practice evals", other: "smoke tests · other" };
    var kinds = U.kinds || {};
    root.appendChild(h("div", { class: "chips", style: "margin-top:14px" },
      ["gaia", "practice", "other"].filter(function (x) { return sc[x]; }).map(function (x) {
        return h("span", { class: "chip" }, h("b", null, scopeNames[x]), h("span", null, plural(sc[x].requests, "request")));
      }),
      Object.keys(kinds).map(function (x) { return h("span", { class: "chip" }, h("b", null, x), h("span", null, plural(kinds[x], "request"))); }),
      U.first ? h("span", { class: "chip" }, h("b", null, "Window"), h("span", null, fmtDate(U.first) + " → " + fmtDate(U.last))) : null));

    var max = Math.max.apply(null, U.models.map(function (m) { return m.requests; }).concat([1]));
    var tb = h("tbody");
    U.models.forEach(function (m) {
      var errs = Object.keys(m.attempt_errors || {});
      tb.appendChild(h("tr", null,
        h("td", { "data-l": "Model" }, h("div", { class: "cfg" }, h("span", { class: "mono", style: "font-size:12.5px;font-weight:600" }, m.model),
          h("small", null, m.free ? "free tier" : "not a :free id"))),
        h("td", { class: "r", "data-l": "Served" }, h("span", { class: "ureq" },
          h("span", { class: "num" }, fmtInt(m.ok), m.failed ? h("small", { class: "ufail" }, " +" + m.failed + " failed") : null),
          h("span", { class: "ubar", "aria-hidden": "true" }, h("i", { style: "width:" + (m.ok / max * 100) + "%" })))),
        h("td", { class: "r", "data-l": "Tokens in · out" }, m.requests ? fmtTok(m.input_tokens) + " · " + fmtTok(m.output_tokens) : "—"),
        h("td", { class: "r", "data-l": "Latency p50" }, m.requests ? fmtSec(m.latency_median) : "—"),
        h("td", { class: "r", "data-l": "As fallback" }, m.fallback_served ? fmtInt(m.fallback_served) : "—"),
        h("td", { "data-l": "Attempt errors" }, errs.length ? h("div", { class: "tags" }, errs.map(function (e) {
          return h("span", { class: "tag" + (/^rate|upstream_rate|skipped/.test(e) ? " warn" : "") }, e.replace(/_/g, " ") + " ×" + m.attempt_errors[e]);
        })) : h("span", { class: "muted" }, "none"))));
    });
    root.appendChild(h("div", { class: "card", style: "margin-top:16px" }, h("table", { class: "ab ut" },
      h("caption", { class: "sr-only" }, "Requests by model"),
      h("thead", null, h("tr", null, ["Model", "Served", "Tokens in · out", "Latency p50", "As fallback", "Attempt errors"].map(function (t, i) {
        return h("th", { class: i && i < 5 ? "r" : null, scope: "col" }, t);
      }))), tb)));
    root.appendChild(h("p", { class: "note" }, icon(I.info, 14), h("span", null,
      "Read from runs/usage.jsonl, one row per request. “Served” counts requests a model answered (a request that failed after every retry and fallback counts against the model it asked for); attempt errors count failed tries against that model before a retry or a fallback took over. Error messages are not copied here, only their kind and HTTP status.")));
  }

  /* ------------------------------------------------------------------ contamination policy */

  function renderPolicy() {
    var box = $("#policy");
    box.appendChild(h("div", { class: "policy-h" }, icon(I.lock, 16), h("h3", null, "Contamination policy")));
    box.appendChild(h("p", null, META.policy || ""));
    if (!PUBLIC) box.appendChild(h("p", { class: "callout err", style: "margin-top:10px;font-family:var(--sans)" },
      "This is a PRIVATE build: GAIA question text, answers and full trajectories are included below. Do not publish it."));
    var lede = $("#questions-lede");
    if (lede) lede.textContent = PUBLIC
      ? "One row per course question in the submitted run: id, attachment type, status, tools, cost and failure tags. Click a row for its step skeleton."
      : "One row per course question in the submitted run. Click a question to open its full trajectory. Columns sort.";
  }

  /* ------------------------------------------------------------------ architecture diagram */

  function dgBox(svg, b) {
    var g = s("g");
    g.appendChild(s("rect", { x: b.x, y: b.y, width: b.w, height: b.h, rx: 10, class: "dg-box " + (b.cls || "") }));
    var lines = [b.t].concat(b.s || []);
    var lh = 17, y0 = b.y + b.h / 2 - (lines.length - 1) * lh / 2 + 5;
    lines.forEach(function (ln, i) {
      g.appendChild(s("text", { x: b.x + (b.left ? 14 : b.w / 2), y: y0 + i * lh, "text-anchor": b.left ? "start" : "middle", class: i ? (b.mono ? "dg-m" : "dg-s") : "dg-t" }, ln));
    });
    svg.appendChild(g);
  }
  /* Marker ids must be unique per SVG: a url(#id) that resolves into the hidden diagram draws no arrowhead. */
  function dgArrow(svg, pts, accent, label) {
    svg.appendChild(s("path", { d: "M" + pts.map(function (p) { return p.join(","); }).join("L"), class: "dg-arrow" + (accent ? " accent" : ""), "marker-end": "url(#" + svg._mk + (accent ? "a" : "") + ")" }));
    if (label) svg.appendChild(s("text", { x: label[0], y: label[1], "text-anchor": label[2] || "middle", class: "dg-s" }, label[3]));
  }
  function dgDefs(svg, prefix) {
    svg._mk = prefix + "-ah";
    var defs = s("defs");
    [[svg._mk, "dg-head"], [svg._mk + "a", "dg-head accent"]].forEach(function (m) {
      defs.appendChild(s("marker", { id: m[0], viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0L10,5L0,10z", class: m[1] })));
    });
    svg.appendChild(defs);
  }
  function dgList(svg, x, y, title, items, step) {
    svg.appendChild(s("text", { x: x, y: y, class: "dg-t" }, title));
    items.forEach(function (t, i) { svg.appendChild(s("text", { x: x, y: y + 22 + i * step, class: "dg-m" }, t)); });
  }

  function wideDiagram() {
    var svg = s("svg", { viewBox: "0 0 1080 540", role: "img", "aria-labelledby": "dg-title-w" }, s("title", { id: "dg-title-w" }, "Architecture: the course API feeds run.py, which drives a CodeAgent loop of model, code, executor and observation; the executor calls web and media tools; the answer passes a guard, is saved with its trace, then submitted."));
    dgDefs(svg, "dgw");
    dgBox(svg, { x: 250, y: 20, w: 500, h: 380, cls: "ghost", t: "" });
    svg.appendChild(s("text", { x: 270, y: 46, class: "dg-cap" }, "CodeAgent · smolagents"));
    dgBox(svg, { x: 790, y: 20, w: 270, h: 380, cls: "ghost", t: "" });
    svg.appendChild(s("text", { x: 810, y: 46, class: "dg-cap" }, "Tools"));

    dgBox(svg, { x: 20, y: 40, w: 190, h: 64, cls: "accent", t: "Course API", s: ["GET /questions · /files"] });
    dgBox(svg, { x: 20, y: 170, w: 190, h: 84, t: "run.py", s: ["cache · resume", "per-question timeout"] });
    dgBox(svg, { x: 280, y: 70, w: 200, h: 70, cls: "accent", t: "Model", s: ["free tier · OpenAI-compatible"] });
    dgBox(svg, { x: 520, y: 70, w: 200, h: 70, cls: "plan", t: "Thought + Python", s: ["one code action per step"] });
    dgBox(svg, { x: 520, y: 230, w: 200, h: 70, t: "Python executor", s: ["authorized imports only"] });
    dgBox(svg, { x: 280, y: 230, w: 200, h: 70, t: "Observation", s: ["printed output · errors"] });
    svg.appendChild(s("text", { x: 500, y: 182, "text-anchor": "middle", class: "dg-s" }, "loop until final_answer"));
    svg.appendChild(s("text", { x: 500, y: 199, "text-anchor": "middle", class: "dg-s" }, "or the step budget"));
    dgBox(svg, { x: 280, y: 330, w: 440, h: 52, cls: "ok", t: "Answer guard", s: ["shape check, then normalize for exact match"] });

    dgList(svg, 810, 74, "Web · Wikipedia", WEB, 19);
    svg.appendChild(s("rect", { x: 808, y: 200, width: 234, height: 26, rx: 7, class: "dg-box ok" }));
    svg.appendChild(s("text", { x: 925, y: 217, "text-anchor": "middle", class: "dg-s" }, "GAIA guard blocks answer sources"));
    dgList(svg, 810, 256, "Files · audio · vision", MEDIA, 18);

    dgBox(svg, { x: 20, y: 450, w: 200, h: 70, t: "report/ · evals/", s: ["this site · practice evals"] });
    dgBox(svg, { x: 250, y: 450, w: 250, h: 70, t: "runs/", s: ["answers.jsonl · traces/"], mono: true });
    dgBox(svg, { x: 530, y: 450, w: 220, h: 70, t: "submit.py", s: ["dry run unless --yes"] });
    dgBox(svg, { x: 790, y: 450, w: 270, h: 70, cls: "accent", t: "Course API", s: ["POST /submit → official score"] });

    dgArrow(svg, [[115, 104], [115, 166]]);
    dgArrow(svg, [[210, 212], [240, 212], [240, 105], [276, 105]]);
    dgArrow(svg, [[480, 105], [516, 105]], true);
    dgArrow(svg, [[620, 140], [620, 226]], true);
    dgArrow(svg, [[520, 265], [484, 265]], true);
    dgArrow(svg, [[380, 230], [380, 144]], true);
    dgArrow(svg, [[720, 255], [786, 255]], false, [753, 246, "middle", "calls"]);
    dgArrow(svg, [[790, 280], [724, 280]], false, [753, 297, "middle", "returns"]);
    dgArrow(svg, [[620, 300], [620, 326]]);
    dgArrow(svg, [[375, 382], [375, 446]]);
    dgArrow(svg, [[500, 485], [526, 485]]);
    dgArrow(svg, [[750, 485], [786, 485]]);
    dgArrow(svg, [[250, 485], [224, 485]]);
    return svg;
  }

  function tallDiagram() {
    var svg = s("svg", { viewBox: "0 0 360 1230", role: "img", "aria-labelledby": "dg-title-t" }, s("title", { id: "dg-title-t" }, "Architecture, top to bottom: course API, run.py, the CodeAgent loop, its tools, the answer guard, saved runs, submission and this report."));
    dgDefs(svg, "dgt");
    dgBox(svg, { x: 20, y: 10, w: 320, h: 60, cls: "accent", t: "Course API", s: ["GET /questions · /files"] });
    dgBox(svg, { x: 20, y: 100, w: 320, h: 64, t: "run.py", s: ["cache · resume · per-question timeout"] });
    dgBox(svg, { x: 20, y: 194, w: 320, h: 340, cls: "ghost", t: "" });
    svg.appendChild(s("text", { x: 324, y: 218, "text-anchor": "end", class: "dg-cap" }, "CodeAgent · smolagents"));
    dgBox(svg, { x: 36, y: 236, w: 132, h: 70, cls: "accent", t: "Model", s: ["free tier"] });
    dgBox(svg, { x: 192, y: 236, w: 132, h: 70, cls: "plan", t: "Thought + code", s: ["one per step"] });
    dgBox(svg, { x: 192, y: 362, w: 132, h: 70, t: "Executor", s: ["safe imports"] });
    dgBox(svg, { x: 36, y: 362, w: 132, h: 70, t: "Observation", s: ["output · errors"] });
    svg.appendChild(s("text", { x: 180, y: 340, "text-anchor": "middle", class: "dg-s" }, "loop"));
    dgBox(svg, { x: 36, y: 458, w: 288, h: 56, cls: "ok", t: "Answer guard", s: ["shape check · normalize"] });
    dgArrow(svg, [[168, 271], [188, 271]], true);
    dgArrow(svg, [[258, 306], [258, 358]], true);
    dgArrow(svg, [[192, 397], [172, 397]], true);
    dgArrow(svg, [[102, 362], [102, 310]], true);
    dgArrow(svg, [[180, 70], [180, 96]]);
    dgArrow(svg, [[102, 164], [102, 232]]);
    dgArrow(svg, [[300, 432], [300, 454]]);

    dgBox(svg, { x: 20, y: 574, w: 320, h: 380, cls: "ghost", t: "" });
    svg.appendChild(s("text", { x: 36, y: 598, class: "dg-cap" }, "Tools the code calls"));
    dgArrow(svg, [[240, 534], [240, 570]], false, [248, 556, "start", "calls ↔ returns"]);
    dgList(svg, 36, 626, "Web · Wikipedia", WEB, 19);
    svg.appendChild(s("rect", { x: 34, y: 750, width: 292, height: 26, rx: 7, class: "dg-box ok" }));
    svg.appendChild(s("text", { x: 180, y: 767, "text-anchor": "middle", class: "dg-s" }, "GAIA guard blocks answer sources"));
    dgList(svg, 36, 802, "Files · audio · vision", MEDIA, 18);

    dgBox(svg, { x: 20, y: 984, w: 320, h: 60, t: "runs/", s: ["answers.jsonl · traces/"], mono: true });
    svg.appendChild(s("path", { d: "M36,486L10,486L10,1014L16,1014", class: "dg-arrow", "marker-end": "url(#" + svg._mk + ")" }));
    dgBox(svg, { x: 20, y: 1074, w: 150, h: 60, t: "submit.py", s: ["dry run first"] });
    dgBox(svg, { x: 190, y: 1074, w: 150, h: 60, t: "report/", s: ["this site · evals"] });
    dgArrow(svg, [[95, 1044], [95, 1070]]);
    dgArrow(svg, [[265, 1044], [265, 1070]]);
    dgBox(svg, { x: 20, y: 1160, w: 320, h: 60, cls: "accent", t: "Course API", s: ["POST /submit → official score"] });
    dgArrow(svg, [[95, 1134], [95, 1156]]);
    return svg;
  }

  function renderHow() {
    var d = $("#diagram");
    d.appendChild(h("div", { class: "dg-wide" }, wideDiagram()));
    d.appendChild(h("div", { class: "dg-tall" }, tallDiagram()));
    var P = [
      ["The loop", ["The model writes a short thought and one block of Python per step.", "The code runs in smolagents' local executor with an allow-list of imports; its printed output is the next observation.", "final_answer is checked for shape (not empty, not essay-length) and normalized for exact match before the run may end."]],
      ["The tools", ["Web search and a page reader that turns HTML, tables and PDFs into text.", "Wikipedia search, page sections and the revision live on a given date, for 'as of' questions.", "Spreadsheets, code, audio (local Whisper), images and YouTube frames through a vision model."]],
      ["Reliability", ["Every answer is cached, so a crashed run resumes where it stopped.", "Each question has a time budget; a stuck question cannot sink the run.", "Each run writes a full trace. This page, the error analysis and the evals all read the same files."]]
    ];
    var wrap = $("#pillars");
    P.forEach(function (p) { wrap.appendChild(h("div", { class: "card pad" }, h("h3", null, p[0]), h("ul", null, p[1].map(function (x) { return h("li", null, x); })))); });

    var src = $("#sources");
    var labels = { answers: "answers", traces: "GAIA traces", submissions: "submissions", error_analysis: "error analysis", evals: "evals", practice_traces: "practice traces", usage: "usage log" };
    Object.keys(labels).forEach(function (k) {
      src.appendChild(h("span", { class: (META.sources || {})[k] ? "on" : "", title: (META.sources || {})[k] ? "present" : "not run yet" }, labels[k]));
    });
  }

  /* ------------------------------------------------------------------ boot */

  function refFromHash() {
    var hs = location.hash || "", m;
    try {
      if ((m = /^#\/q\/(.+)$/.exec(hs))) return { src: "gaia", id: decodeURIComponent(m[1]) };
      if ((m = /^#\/p\/([^/]+)\/(.+)$/.exec(hs))) return { src: "practice", run: decodeURIComponent(m[1]), id: decodeURIComponent(m[2]) };
    } catch (e) { /* malformed hash */ }
    return null;
  }
  function validRef(r) {
    if (!r) return false;
    if (r.src === "gaia") return !!(QBYID[r.id] || TRACES[r.id]);
    return !!(PRUN[r.run] && practiceItem(r.run, r.id));
  }

  initTheme();
  var initial = refFromHash();
  if (!validRef(initial)) initial = null;
  if (initial) {
    viewer.src = initial.src; viewer.run = initial.run || null; viewer.id = initial.id;
    if (initial.src === "practice") prac.run = initial.run;
  }
  renderHero();
  renderPolicy();
  renderPractice();
  renderScoreboard();
  renderViewer();
  markRows();
  renderCharts();
  renderFailures();
  renderEvals();
  renderUsage();
  renderHow();
  initNav();
  redrawCharts();

  var rt;
  if ("ResizeObserver" in window) {
    var ro = new ResizeObserver(function () { clearTimeout(rt); rt = setTimeout(redrawCharts, 80); });
    drawers.forEach(function (d) { ro.observe(d.box); });
  } else window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(redrawCharts, 120); });

  window.addEventListener("hashchange", function () {
    var r = refFromHash();
    if (!validRef(r) || (r.src === viewer.src && r.id === viewer.id && (r.run || null) === (viewer.run || null))) return;
    if (r.src === "gaia") openGaia(r.id, true); else openPractice(r.run, r.id, true);
  });
  if (initial) {
    var go = function () { var t = $("#trajectory"); if (t) window.scrollTo(0, t.getBoundingClientRect().top + window.scrollY - 64); };
    if (document.readyState === "complete") go(); else window.addEventListener("load", function () { requestAnimationFrame(go); });
  }
})();
