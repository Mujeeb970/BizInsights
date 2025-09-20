# app.py — Business Insights Agent (Final)
# - Groq only (free), key via .env
# - Web + Academic search, quality ranking, parallel fetch
# - 5Rs (expanded) + Feedback Loops + Enablers + Barriers + Key Metrics
# - Saves Markdown + CSV to Downloads/BusinessInsightsReports

import os, re, time, threading
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY from .env in the same folder

# ---- UI
import tkinter as tk
from tkinter import ttk, messagebox

# ---- Search & parsing
try:
    from ddgs import DDGS  # pip install ddgs
except ImportError:
    raise SystemExit("Missing ddgs. Run: pip install ddgs")
import httpx
from bs4 import BeautifulSoup        # pip install beautifulsoup4
from readability import Document      # pip install readability-lxml
import tldextract                     # pip install tldextract

# ---- Data
import pandas as pd                   # pip install pandas
from io import StringIO
import feedparser                     # pip install feedparser

# ---- LLM (Groq)
from groq import Groq                 # pip install groq

# =================== CONFIG ===================
REGION_DEFAULT = "wt-wt"
MAX_SOURCES_DEFAULT = 24            # UI allows up to 40
PER_DOMAIN_LIMIT_DEFAULT = 2
FETCH_CONCURRENCY = 8               # parallel fetch threads (6–12 is good)
REQUEST_TIMEOUT = 25                # seconds

DEFAULT_MODEL_PRIMARY   = "llama-3.3-70b-versatile"
DEFAULT_MODEL_FALLBACK  = "llama-3.1-8b-instant"
MODEL_TEMPERATURE = 0.1             # lower = more precise

DOWNLOADS = Path(os.path.expanduser("~")) / "Downloads"
REPORT_DIR = DOWNLOADS / "BusinessInsightsReports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Source quality boosts (trusted, data-heavy domains)
QUALITY_WEIGHTS = {
    # Government / legal / official
    ".gov.uk": 6, ".gov": 6, "legislation.gov.uk": 6, "data.gov.uk": 6,
    ".nhs.uk": 6, ".who.int": 6, "ec.europa.eu": 6, ".europa.eu": 6, "ema.europa.eu": 6,
    "parliament.uk": 5,

    # Academia / stats / multilaterals
    ".ac.uk": 5, ".edu": 5, "ons.gov.uk": 6, "oecd.org": 5, "worldbank.org": 5,
    "data.worldbank.org": 5, "imf.org": 5, "ourworldindata.org": 5, "iea.org": 5,

    # Health science
    "cdc.gov": 6, "nih.gov": 5, "ecdc.europa.eu": 6,

    # Journals & preprints
    "arxiv.org": 5, "nature.com": 5, "science.org": 5, "sciencedirect.com": 4,
    "springer.com": 4, "wiley.com": 4, "nejm.org": 5, "thelancet.com": 5,
    "acm.org": 5, "ieee.org": 5,

    # Reputable media (light boost)
    "bbc.co.uk": 3, "ft.com": 3, "economist.com": 3, "reuters.com": 3, "bloomberg.com": 3,

    # (optional) commercial stats portals
    "statista.com": 3
}

# Low-signal / social / UGC
BLOCKLIST_PARTIALS = [
    "pinterest.", "reddit.", "quora.", "facebook.", "tiktok.", "instagram.",
    "medium.com/@", "slideshare.", "scribd.", "fandom.com", "wattpad."
]
# ==============================================

def now_london():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def sanitize_filename(s: str, limit=90):
    return re.sub(r"[^A-Za-z0-9_. -]+", "_", s)[:limit].strip("_ ")

def domain_name(url:str):
    try:
        ext = tldextract.extract(url)
        return ".".join([p for p in [ext.domain, ext.suffix] if p]).lower()
    except:
        return ""

# -------- Web search (DuckDuckGo via ddgs) --------
def web_search_text(query: str, max_results: int = 30, region: str = REGION_DEFAULT):
    out = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=max_results, safesearch="moderate", region=region):
            out.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "snippet": r.get("body"),
                "date": r.get("date")
            })
    # Deduplicate by URL
    seen, dedup = set(), []
    for r in out:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u); dedup.append(r)
    return dedup

# -------- Academic search (OpenAlex + arXiv + Crossref) --------
def search_openalex(query: str, max_results:int=15):
    url = "https://api.openalex.org/works"
    params = {"search": query, "per_page": max_results, "sort": "relevance_score:desc"}
    out = []
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as s:
            r = s.get(url, params=params)
            if r.status_code != 200: return out
            for w in r.json().get("results", []):
                title = w.get("title")
                pub = w.get("host_venue", {}).get("display_name") or ""
                year = w.get("publication_year") or ""
                loc = w.get("open_access", {}).get("oa_url") or \
                      (w.get("primary_location",{}) or {}).get("source",{}).get("home_page_url") or \
                      w.get("primary_location",{}).get("landing_page_url") or \
                      w.get("id")
                abstract = w.get("abstract_inverted_index")
                snippet = (" ".join(sorted(abstract.keys())[:60]) if isinstance(abstract, dict) else pub)
                if loc:
                    out.append({"title": title, "url": loc, "snippet": snippet, "date": str(year)})
    except Exception:
        pass
    return out

def search_arxiv(query:str, max_results:int=12):
    try:
        api = f"http://export.arxiv.org/api/query?search_query=all:{httpx.utils.quote(query)}&start=0&max_results={max_results}&sortBy=relevance"
        feed = feedparser.parse(api)
        out = []
        for e in feed.entries:
            title = e.get("title", "").replace("\n"," ").strip()
            link = e.get("link")
            summary = e.get("summary","").replace("\n"," ").strip()
            date = (e.get("updated") or e.get("published") or "")[:10]
            if link:
                out.append({"title": title, "url": link, "snippet": summary, "date": date})
        return out
    except Exception:
        return []

def search_crossref(query:str, max_results:int=12):
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": max_results, "sort": "relevance"}
    out = []
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, headers={"User-Agent":"BizInsights/1.2 (mailto:example@example.com)"}) as s:
            r = s.get(url, params=params)
            if r.status_code != 200: return out
            for it in r.json().get("message", {}).get("items", []):
                title = " ".join(it.get("title") or [])[:300]
                url_primary = None
                for li in it.get("link", []):
                    if li.get("URL"): url_primary = li["URL"]; break
                url_primary = url_primary or it.get("URL") or (f"https://doi.org/{it.get('DOI')}" if it.get("DOI") else None)
                date_parts = it.get("issued",{}).get("date-parts", [[]])
                year = str(date_parts[0][0]) if date_parts and date_parts[0] else ""
                pub = (it.get("container-title") or [""])[0]
                snippet = pub or (it.get("publisher") or "")
                if url_primary:
                    out.append({"title": title, "url": url_primary, "snippet": snippet, "date": year})
    except Exception:
        pass
    return out

# -------- Fetch & clean page (parallel-friendly) --------
def fetch_and_clean_single(url: str):
    headers = {"User-Agent": "BusinessInsightsBot/1.3 (+local research use)"}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=True, headers=headers) as s:
            r = s.get(url); r.raise_for_status(); html = r.text
    except Exception as e:
        return {"url": url, "title": None, "text": f"FETCH_ERROR: {e}", "ok": False}
    try:
        doc = Document(html)
        title = doc.short_title()
        soup = BeautifulSoup(doc.summary(), "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        text = re.sub(r"\s+"," ", soup.get_text(" ", strip=True))[:160_000]
        return {"url": url, "title": title, "text": text, "ok": True}
    except Exception as e:
        return {"url": url, "title": None, "text": f"PARSING_ERROR: {e}", "ok": False}

# -------- Ranking & filtering --------
def quality_score(url: str):
    d = domain_name(url); score = 0
    for k, w in QUALITY_WEIGHTS.items():
        if d.endswith(k.replace("*","")) or k in d:
            score = max(score, w)
    return score

def is_blocked(url: str):
    u = url.lower()
    return any(b in u for b in BLOCKLIST_PARTIALS)

def choose_top_sources(results, per_domain_limit=PER_DOMAIN_LIMIT_DEFAULT, max_total=30):
    ranked = []
    for r in results:
        u = r.get("url")
        if not u: continue
        if is_blocked(u): continue
        ranked.append((quality_score(u), r))
    ranked.sort(key=lambda x: x[0], reverse=True)

    kept, used = [], {}
    for _, r in ranked:
        d = domain_name(r["url"]); used[d] = used.get(d, 0)
        if used[d] < per_domain_limit:
            kept.append(r); used[d] += 1
        if len(kept) >= max_total:
            break
    return kept

# -------- Stats-first system prompt (with expanded 5Rs) --------
BUSINESS_SYSTEM_INSTRUCTIONS = """
You are BusinessResearcher, a neutral analyst.
Use only the numbered web sources provided. No invented citations or URLs.

Write for senior decision-makers: crisp, structured, and **statistics-first**.
Whenever numbers exist, ALWAYS include value, **unit**, **year**, and citation [n].

Output structure:
- Key Metrics at a Glance (5–10 bullets; each bullet a concrete metric with units, year, and [n])
- Executive Summary (6–10 bullets; big takeaways with citations and key numbers)
- Evidence Table [# | Source | Publisher | Date | Key statistic or finding | URL] (10–25 rows)
- 5Rs (each section should contain **at least 5–10 detailed bullets**. 
  Go beyond simple statements:
  – explain background and context,
  – give real-world examples,
  – name key organisations or programmes,
  – include figures such as budgets, market sizes or workforce numbers where available,
  – and cite every claim.)
  1) Rules – laws, regulations, standards, government policies and international agreements.
  2) Roles – key actors, agencies, companies and their responsibilities; include overlaps or mandate conflicts.
  3) Relationships – the wider ecosystem: partnerships, supply chains, interdependencies; note how these affect outcomes.
  4) Resources – funding streams, human resources, technologies, infrastructure, datasets; provide quantitative capacity indicators.
  5) Results – measurable outcomes, KPIs, trends and benchmarks; include statistics (%, £, $, people, tonnes) and the data year.
- Feedback Loops (leading/lagging indicators; causal links; quantify where possible)
- Enablers (policies, funding, technology; quantify scale/impact if possible)
- Barriers (constraints, risks, costs, bottlenecks; quantify impact if possible)
- Consensus vs Disagreements (bulleted, with [n])
- Limits & Unknowns (what data is missing/weak)
- How to Verify (replicable searches; specific datasets/reports to check)

Style: bullet-led, professional, **data-rich**, no fluff. Tie every non-trivial claim to [n].
"""

def build_business_prompt(topic:str, fetched:list, searched_when:str):
    lines = []
    lines.append(f"Topic: {topic}")
    lines.append(f"Searched on: {searched_when}")
    lines.append("\nFollow the structure and style in the system instructions exactly.")
    lines.append("\nSources (numbered):")
    for i, src in enumerate(fetched, 1):
        title = src.get("title") or "(no title)"
        url = src["url"]
        date = src.get("date") or ""
        lines.append(f"[{i}] {title} — {url}" + (f" ({date})" if date else ""))
    lines.append("\nShort excerpts from sources (for grounding):")
    for i, src in enumerate(fetched, 1):
        txt = (src.get("text") or "")[:1400]
        lines.append(f"\nFrom source [{i}] — {src.get('title') or '(no title)'}:\n{txt}\n")
    lines.append("""
After the main report, also output a compact CSV block between <CSV>...</CSV> with columns:
#,[Source Title],Publisher,Date,One-line finding,URL
Use 10–25 of the most important rows. Avoid commas in URL; quote fields if needed.
""")
    return "\n".join(lines)

# --------------- Core pipeline (parallel fetch) ----------------
def run_pipeline(topic:str, region:str, per_domain_limit:int, max_sources:int, include_academia:bool, progress_cb=None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("No GROQ_API_KEY found. Create a .env next to app.py with: GROQ_API_KEY=gsk_your_key_here")

    # Gather widely, then select top
    if progress_cb: progress_cb("Searching the web…")
    base_hits = web_search_text(topic, max_results=max_sources*3, region=region)

    if include_academia:
        if progress_cb: progress_cb("Adding academic sources (OpenAlex, arXiv, Crossref)…")
        base_hits += search_openalex(topic, max_results=15)
        base_hits += search_arxiv(topic, max_results=12)
        base_hits += search_crossref(topic, max_results=12)

    if not base_hits:
        raise RuntimeError("No results found. Try a broader query or switch region to 'wt-wt'.")

    picked_meta = choose_top_sources(base_hits, per_domain_limit=per_domain_limit, max_total=max_sources)

    # Parallel fetch of pages
    if progress_cb: progress_cb(f"Fetching {len(picked_meta)} sources in parallel…")
    fetched = [None] * len(picked_meta)
    with ThreadPoolExecutor(max_workers=FETCH_CONCURRENCY) as ex:
        future_map = {ex.submit(fetch_and_clean_single, r["url"]): idx for idx, r in enumerate(picked_meta)}
        done_count = 0
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                info = fut.result()
            except Exception as e:
                info = {"url": picked_meta[idx]["url"], "title": None, "text": f"FETCH_ERROR: {e}", "ok": False}
            meta = picked_meta[idx]
            if not info.get("title"): info["title"] = meta.get("title")
            info["date"] = meta.get("date")
            info["snippet"] = meta.get("snippet")
            fetched[idx] = info

            done_count += 1
            if progress_cb and (done_count == len(picked_meta) or done_count % 2 == 0):
                progress_cb(f"Fetched {done_count}/{len(picked_meta)}…")

    # Ask the model
    if progress_cb: progress_cb("Asking the model…")
    searched_when = now_london()
    prompt = build_business_prompt(topic, fetched, searched_when)

    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL_PRIMARY,
            messages=[
                {"role":"system","content": BUSINESS_SYSTEM_INSTRUCTIONS},
                {"role":"user","content": prompt}
            ],
            temperature=MODEL_TEMPERATURE
        )
    except Exception as e:
        if progress_cb: progress_cb(f"Primary model failed ({e}); trying fallback…")
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL_FALLBACK,
            messages=[
                {"role":"system","content": BUSINESS_SYSTEM_INSTRUCTIONS},
                {"role":"user","content": prompt}
            ],
            temperature=MODEL_TEMPERATURE
        )

    text = resp.choices[0].message.content

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_topic = sanitize_filename(topic, 90)
    md_path = REPORT_DIR / f"{ts}_{safe_topic}_BUSINESS_WEB.md"
    md_path.write_text(text, encoding="utf-8")

    # Extract CSV (if present)
    m = re.search(r"<CSV>(.*?)</CSV>", text, flags=re.DOTALL|re.IGNORECASE)
    csv_path = None
    if m:
        csv_raw = m.group(1).strip()
        try:
            df = pd.read_csv(StringIO(csv_raw))
            csv_path = REPORT_DIR / f"{ts}_{safe_topic}_EvidenceTable.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
        except Exception:
            pass

    return md_path, csv_path

# --------------- Pretty Tkinter UI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Business Insights Agent — Fast & Detailed")
        self.geometry("1000x640")
        self.minsize(940, 580)

        # ttk styling
        style = ttk.Style()
        try: style.theme_use("clam")
        except: pass
        style.configure("TButton", padding=8, font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("Subheader.TLabel", font=("Segoe UI", 11), foreground="#555")
        style.configure("TEntry", padding=6, font=("Segoe UI", 10))
        style.configure("TCombobox", padding=6, font=("Segoe UI", 10))

        # Header
        header = ttk.Frame(self, padding=(16, 12)); header.pack(fill="x")
        ttk.Label(header, text="Business Insights Agent", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="Web + Academic research • 5Rs (expanded) + Feedback Loops + Enablers + Barriers • Stats-first",
                  style="Subheader.TLabel").pack(anchor="w", pady=(2,0))

        # Prompt
        prompt_frame = ttk.Frame(self, padding=(16, 8)); prompt_frame.pack(fill="x")
        ttk.Label(prompt_frame, text="Your prompt / topic:").grid(row=0, column=0, sticky="w")
        self.ent_prompt = ttk.Entry(prompt_frame)
        self.ent_prompt.grid(row=1, column=0, columnspan=8, sticky="we", pady=(4,0))
        prompt_frame.columnconfigure(7, weight=1)

        # Options
        opts = ttk.Frame(self, padding=(16, 8)); opts.pack(fill="x")
        ttk.Label(opts, text="Region:").grid(row=0, column=0, sticky="w")
        self.cmb_region = ttk.Combobox(opts, values=["wt-wt","uk-en","us-en","in-en","pk-en"], width=12)
        self.cmb_region.set(REGION_DEFAULT); self.cmb_region.grid(row=0, column=1, padx=(6,18))

        ttk.Label(opts, text="Max sources (6–40):").grid(row=0, column=2, sticky="w")
        self.spn_sources = ttk.Spinbox(opts, from_=6, to=40, width=6)
        self.spn_sources.set(MAX_SOURCES_DEFAULT); self.spn_sources.grid(row=0, column=3, padx=(6,18))

        ttk.Label(opts, text="Per-domain limit:").grid(row=0, column=4, sticky="w")
        self.spn_perdomain = ttk.Spinbox(opts, from_=1, to=4, width=6)
        self.spn_perdomain.set(PER_DOMAIN_LIMIT_DEFAULT); self.spn_perdomain.grid(row=0, column=5, padx=(6,18))

        self.var_academia = tk.BooleanVar(value=True)
        self.chk_academia = ttk.Checkbutton(opts, text="Include academic sources (OpenAlex, arXiv, Crossref)", variable=self.var_academia)
        self.chk_academia.grid(row=0, column=6, sticky="w")

        # Buttons
        btns = ttk.Frame(self, padding=(16, 4)); btns.pack(fill="x")
        self.btn_run = ttk.Button(btns, text="Run", command=self.on_run); self.btn_run.pack(side="left")
        ttk.Button(btns, text="Open Reports Folder", command=self.open_reports).pack(side="left", padx=10)

        # Progress + log
        prog = ttk.Frame(self, padding=(16, 6)); prog.pack(fill="x")
        self.progress = ttk.Progressbar(prog, mode="indeterminate"); self.progress.pack(fill="x")
        logf = ttk.Frame(self, padding=(16, 6)); logf.pack(fill="both", expand=True)
        self.txt = tk.Text(logf, height=16, font=("Consolas", 10)); self.txt.pack(fill="both", expand=True)

        self.log(f"Reports will be saved to: {REPORT_DIR}")
        if not os.environ.get("GROQ_API_KEY"):
            self.log("⚠ No GROQ_API_KEY found. Create a .env next to app.py with: GROQ_API_KEY=gsk_your_key_here")

    def log(self, msg):
        self.txt.insert("end", msg + "\n"); self.txt.see("end"); self.update_idletasks()

    def set_busy(self, busy=True):
        if busy:
            self.progress.start(12); self.btn_run.config(state="disabled")
        else:
            self.progress.stop(); self.btn_run.config(state="normal")

    def on_run(self):
        topic = self.ent_prompt.get().strip()
        if not topic:
            messagebox.showwarning("Missing", "Please enter a prompt/topic."); return
        if not os.environ.get("GROQ_API_KEY"):
            messagebox.showerror("Missing key", "No GROQ_API_KEY found in .env next to app.py."); return

        region = self.cmb_region.get().strip() or REGION_DEFAULT
        try:
            max_sources = int(self.spn_sources.get()); per_domain = int(self.spn_perdomain.get())
        except:
            max_sources = MAX_SOURCES_DEFAULT; per_domain = PER_DOMAIN_LIMIT_DEFAULT

        include_academia = bool(self.var_academia.get())

        self.log(f"Running: {topic}")
        self.log(f"Region={region}, max_sources={max_sources}, per_domain={per_domain}, academia={include_academia}")
        self.set_busy(True)
        threading.Thread(
            target=self._run_thread,
            args=(topic, region, per_domain, max_sources, include_academia),
            daemon=True
        ).start()

    def _run_thread(self, topic, region, per_domain, max_sources, include_academia):
        try:
            md_path, csv_path = run_pipeline(
                topic=topic,
                region=region,
                per_domain_limit=per_domain,
                max_sources=max_sources,
                include_academia=include_academia,
                progress_cb=self.log
            )
            self.log(f"✅ Saved Markdown: {md_path}")
            if csv_path: self.log(f"✅ Saved CSV: {csv_path}")
            else: self.log("ℹ️ No CSV extracted; copy from report if needed.")
        except Exception as e:
            self.log(f"❌ Error: {e}")
        finally:
            self.set_busy(False)

    def open_reports(self):
        try: os.startfile(REPORT_DIR)
        except Exception as e: self.log(f"Open folder failed: {e}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
