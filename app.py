# app.py — BizInsights (Streamlit, shareable link, private code)
# - Uses your existing pipeline (web + academic search, Groq models)
# - Reads GROQ_API_KEY from Streamlit Secrets (fallback to env)
# - Saves report on server AND offers user-side downloads (Markdown + CSV)

import os, re, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO, BytesIO

import streamlit as st
import httpx
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from readability import Document
from groq import Groq

# Optional libs you already use
try:
    from ddgs import DDGS
except ImportError:
    raise SystemExit("Missing ddgs. Add 'ddgs' to requirements.txt and redeploy.")
import tldextract

# =================== CONFIG ===================
REGION_DEFAULT = "wt-wt"
MAX_SOURCES_DEFAULT = 24
PER_DOMAIN_LIMIT_DEFAULT = 2
FETCH_CONCURRENCY = 8
REQUEST_TIMEOUT = 25

DEFAULT_MODEL_PRIMARY   = "llama-3.3-70b-versatile"
DEFAULT_MODEL_FALLBACK  = "llama-3.1-8b-instant"
MODEL_TEMPERATURE = 0.1

# Server-side report folder (for your records)
DOWNLOADS = Path.home() / "Downloads"
REPORT_DIR = DOWNLOADS / "BusinessInsightsReports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_WEIGHTS = {
    ".gov.uk": 6, ".gov": 6, "legislation.gov.uk": 6, "data.gov.uk": 6,
    ".nhs.uk": 6, ".who.int": 6, "ec.europa.eu": 6, ".europa.eu": 6,
    "parliament.uk": 5,
    ".ac.uk": 5, ".edu": 5, "ons.gov.uk": 6, "oecd.org": 5, "worldbank.org": 5,
    "data.worldbank.org": 5, "imf.org": 5, "ourworldindata.org": 5, "iea.org": 5,
    "cdc.gov": 6, "nih.gov": 5, "ecdc.europa.eu": 6,
    "arxiv.org": 5, "nature.com": 5, "science.org": 5, "sciencedirect.com": 4,
    "springer.com": 4, "wiley.com": 4, "nejm.org": 5, "thelancet.com": 5,
    "acm.org": 5, "ieee.org": 5,
    "bbc.co.uk": 3, "ft.com": 3, "economist.com": 3, "reuters.com": 3, "bloomberg.com": 3,
    "statista.com": 3
}
BLOCKLIST_PARTIALS = [
    "pinterest.", "reddit.", "quora.", "facebook.", "tiktok.", "instagram.",
    "medium.com/@", "slideshare.", "scribd.", "fandom.com", "wattpad."
]
# ==============================================

def now_london():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def sanitize_filename(s: str, limit=90):
    import re
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

# -------- Academic search --------
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

# -------- Fetch & clean page --------
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

BUSINESS_SYSTEM_INSTRUCTIONS = """
You are BusinessResearcher, a neutral analyst.
Use only the numbered web sources provided. No invented citations or URLs.
Write for senior decision-makers: crisp, structured, and statistics-first.
Whenever numbers exist, ALWAYS include value, unit, year, and citation [n].
Output structure:
- Key Metrics at a Glance (5–10 bullets; metric + unit + year + [n])
- Executive Summary (6–10 bullets with citations and key numbers)
- Evidence Table [# | Source | Publisher | Date | Key finding | URL] (10–25 rows)
- 5Rs (Rules, Roles, Relationships, Resources, Results) with 5–10 detailed bullets each, all cited
- Feedback Loops, Enablers, Barriers (quantify where possible)
- Consensus vs Disagreements
- Limits & Unknowns
- How to Verify
After the main report, also output a compact CSV block between <CSV>...</CSV> with columns:
#,[Source Title],Publisher,Date,One-line finding,URL
Use 10–25 rows. Quote fields if needed.
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

def run_pipeline(topic:str, region:str, per_domain_limit:int, max_sources:int, include_academia:bool, progress_cb=None):
    # Read key from Streamlit Secrets first; fallback to env
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("No GROQ_API_KEY set. Add it in Streamlit → Settings → Secrets.")

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

    # LLM
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

    # Try to extract CSV block
    m = re.search(r"<CSV>(.*?)</CSV>", text, flags=re.DOTALL|re.IGNORECASE)
    csv_bytes = None
    csv_path = None
    if m:
        csv_raw = m.group(1).strip()
        try:
            df = pd.read_csv(StringIO(csv_raw))
            csv_path = REPORT_DIR / f"{ts}_{safe_topic}_EvidenceTable.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
        except Exception:
            # if parsing fails, still offer raw CSV text
            csv_bytes = csv_raw.encode("utf-8")

    return text.encode("utf-8"), csv_bytes, md_path, csv_path

# =================== STREAMLIT UI ===================
st.set_page_config(page_title="BizInsights", page_icon="📊", layout="centered")
st.title("Research & Evidence Generator")
st.caption("Web + Academic search • Stats-first report • Evidence table CSV")

with st.sidebar:
    st.subheader("Options")
    region = st.selectbox("Region", ["wt-wt","uk-en","us-en","in-en","pk-en"], index=0)
    max_sources = st.slider("Max sources", 6, 40, MAX_SOURCES_DEFAULT, step=2)
    per_domain = st.slider("Per-domain limit", 1, 4, PER_DOMAIN_LIMIT_DEFAULT)
    include_academia = st.checkbox("Include academic sources (OpenAlex, arXiv, Crossref)", value=True)
    st.markdown("---")
    # st.markdown("**Secrets status:** " + ("✅ Found GROQ_API_KEY" if (st.secrets.get('GROQ_API_KEY') or os.environ.get('GROQ_API_KEY')) else "❌ Missing GROQ_API_KEY"))

topic = st.text_input("Your prompt / topic", value="", placeholder="e.g., NHS AI adoption metrics in 2024; UK EV charging policy; Cybersecurity in autonomous vehicles")
run = st.button("Run")

log_area = st.empty()
md_download = st.empty()
csv_download = st.empty()

def log(msg):
    prev = st.session_state.get("log_text","")
    st.session_state["log_text"] = prev + msg + "\n"
    log_area.code(st.session_state["log_text"])

if run:
    st.session_state["log_text"] = ""
    try:
        log("Searching…")
        md_bytes, csv_bytes, md_path, csv_path = run_pipeline(
            topic=topic.strip(),
            region=region,
            per_domain_limit=per_domain,
            max_sources=max_sources,
            include_academia=include_academia,
            progress_cb=log
        )
        log(f"✅ Saved Markdown on server: {md_path}")
        if csv_path: log(f"✅ Saved CSV on server: {csv_path}")
        else: log("ℹ️ No CSV extracted by model; you can still download the Markdown report.")

        # Offer downloads to the user (their computer)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_topic = sanitize_filename(topic or "report", 60)
        md_fname = f"{ts}_{safe_topic}_BUSINESS_WEB.md"

        md_download.download_button(
            label="⬇️ Download Markdown report",
            data=md_bytes,
            file_name=md_fname,
            mime="text/markdown"
        )

        if csv_bytes:
            csv_download.download_button(
                label="⬇️ Download Evidence Table (CSV)",
                data=csv_bytes,
                file_name=f"{ts}_{safe_topic}_EvidenceTable.csv",
                mime="text/csv"
            )

    except Exception as e:
        log(f"❌ Error: {e}")


