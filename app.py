# app.py — BizInsights
# Web + academic research using Groq
# Generates a Markdown report + downloadable CSV evidence table

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from urllib.parse import quote

import streamlit as st
import httpx
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from readability import Document
from groq import Groq
import tldextract

try:
    from ddgs import DDGS
except ImportError:
    raise SystemExit(
        "Missing ddgs. Add 'ddgs' to requirements.txt and redeploy."
    )


# ============================================================
# CONFIG
# ============================================================

REGION_DEFAULT = "wt-wt"
MAX_SOURCES_DEFAULT = 24
PER_DOMAIN_LIMIT_DEFAULT = 2
FETCH_CONCURRENCY = 8
REQUEST_TIMEOUT = 25

# Current Groq models
DEFAULT_MODEL_PRIMARY = "openai/gpt-oss-120b"
DEFAULT_MODEL_FALLBACK = "openai/gpt-oss-20b"

MODEL_TEMPERATURE = 0.1


# Folder used by Streamlit server
DOWNLOADS = Path.home() / "Downloads"
REPORT_DIR = DOWNLOADS / "BusinessInsightsReports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


QUALITY_WEIGHTS = {
    ".gov.uk": 6,
    ".gov": 6,
    "legislation.gov.uk": 6,
    "data.gov.uk": 6,
    ".nhs.uk": 6,
    ".who.int": 6,
    "ec.europa.eu": 6,
    ".europa.eu": 6,
    "parliament.uk": 5,
    ".ac.uk": 5,
    ".edu": 5,
    "ons.gov.uk": 6,
    "oecd.org": 5,
    "worldbank.org": 5,
    "data.worldbank.org": 5,
    "imf.org": 5,
    "ourworldindata.org": 5,
    "iea.org": 5,
    "cdc.gov": 6,
    "nih.gov": 5,
    "ecdc.europa.eu": 6,
    "arxiv.org": 5,
    "nature.com": 5,
    "science.org": 5,
    "sciencedirect.com": 4,
    "springer.com": 4,
    "wiley.com": 4,
    "nejm.org": 5,
    "thelancet.com": 5,
    "acm.org": 5,
    "ieee.org": 5,
    "bbc.co.uk": 3,
    "ft.com": 3,
    "economist.com": 3,
    "reuters.com": 3,
    "bloomberg.com": 3,
    "statista.com": 3,
}


BLOCKLIST_PARTIALS = [
    "pinterest.",
    "reddit.",
    "quora.",
    "facebook.",
    "tiktok.",
    "instagram.",
    "medium.com/@",
    "slideshare.",
    "scribd.",
    "fandom.com",
    "wattpad.",
]


# ============================================================
# HELPERS
# ============================================================

def now_london():
    return (
        datetime.now(timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    )


def sanitize_filename(s: str, limit=90):
    return re.sub(
        r"[^A-Za-z0-9_. -]+",
        "_",
        s
    )[:limit].strip("_ ")


def domain_name(url: str):
    try:
        ext = tldextract.extract(url)

        return ".".join(
            p
            for p in [ext.domain, ext.suffix]
            if p
        ).lower()

    except Exception:
        return ""


# ============================================================
# WEB SEARCH
# ============================================================

def web_search_text(
    query: str,
    max_results: int = 30,
    region: str = REGION_DEFAULT
):
    out = []

    with DDGS() as ddg:
        for r in ddg.text(
            query,
            max_results=max_results,
            safesearch="moderate",
            region=region
        ):
            out.append(
                {
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body"),
                    "date": r.get("date"),
                }
            )

    # Remove duplicate URLs
    seen = set()
    dedup = []

    for r in out:
        url = r.get("url")

        if url and url not in seen:
            seen.add(url)
            dedup.append(r)

    return dedup


# ============================================================
# OPENALEX
# ============================================================

def search_openalex(
    query: str,
    max_results: int = 15
):
    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per_page": max_results,
        "sort": "relevance_score:desc",
    }

    out = []

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(url, params=params)

            if response.status_code != 200:
                return out

            for work in response.json().get("results", []):

                title = work.get("title")

                primary_location = (
                    work.get("primary_location") or {}
                )

                source = (
                    primary_location.get("source") or {}
                )

                publication = (
                    source.get("display_name") or ""
                )

                year = (
                    work.get("publication_year") or ""
                )

                open_access = (
                    work.get("open_access") or {}
                )

                location = (
                    open_access.get("oa_url")
                    or primary_location.get("landing_page_url")
                    or source.get("homepage_url")
                    or work.get("id")
                )

                abstract = work.get(
                    "abstract_inverted_index"
                )

                if isinstance(abstract, dict):
                    snippet = " ".join(
                        list(abstract.keys())[:60]
                    )
                else:
                    snippet = publication

                if location:
                    out.append(
                        {
                            "title": title,
                            "url": location,
                            "snippet": snippet,
                            "date": str(year),
                        }
                    )

    except Exception:
        pass

    return out


# ============================================================
# ARXIV
# ============================================================

def search_arxiv(
    query: str,
    max_results: int = 12
):
    try:
        encoded_query = quote(query)

        api = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{encoded_query}"
            f"&start=0"
            f"&max_results={max_results}"
            f"&sortBy=relevance"
        )

        feed = feedparser.parse(api)

        out = []

        for entry in feed.entries:

            title = (
                entry.get("title", "")
                .replace("\n", " ")
                .strip()
            )

            link = entry.get("link")

            summary = (
                entry.get("summary", "")
                .replace("\n", " ")
                .strip()
            )

            date = (
                entry.get("updated")
                or entry.get("published")
                or ""
            )[:10]

            if link:
                out.append(
                    {
                        "title": title,
                        "url": link,
                        "snippet": summary,
                        "date": date,
                    }
                )

        return out

    except Exception:
        return []


# ============================================================
# CROSSREF
# ============================================================

def search_crossref(
    query: str,
    max_results: int = 12
):
    url = "https://api.crossref.org/works"

    params = {
        "query": query,
        "rows": max_results,
        "sort": "relevance",
    }

    out = []

    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent":
                "BizInsights/1.3 research application"
            }
        ) as client:

            response = client.get(
                url,
                params=params
            )

            if response.status_code != 200:
                return out

            items = (
                response.json()
                .get("message", {})
                .get("items", [])
            )

            for item in items:

                title = " ".join(
                    item.get("title") or []
                )[:300]

                primary_url = None

                for link in item.get("link", []):
                    if link.get("URL"):
                        primary_url = link["URL"]
                        break

                if not primary_url:
                    primary_url = item.get("URL")

                if (
                    not primary_url
                    and item.get("DOI")
                ):
                    primary_url = (
                        f"https://doi.org/{item['DOI']}"
                    )

                date_parts = (
                    item.get("issued", {})
                    .get("date-parts", [[]])
                )

                year = ""

                if date_parts and date_parts[0]:
                    year = str(date_parts[0][0])

                publication = (
                    item.get("container-title")
                    or [""]
                )[0]

                snippet = (
                    publication
                    or item.get("publisher")
                    or ""
                )

                if primary_url:
                    out.append(
                        {
                            "title": title,
                            "url": primary_url,
                            "snippet": snippet,
                            "date": year,
                        }
                    )

    except Exception:
        pass

    return out


# ============================================================
# FETCH AND CLEAN WEB PAGE
# ============================================================

def fetch_and_clean_single(url: str):

    headers = {
        "User-Agent":
        "BusinessInsightsBot/1.3 (+research use)"
    }

    try:
        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=headers
        ) as client:

            response = client.get(url)
            response.raise_for_status()

            html = response.text

    except Exception as e:
        return {
            "url": url,
            "title": None,
            "text": f"FETCH_ERROR: {e}",
            "ok": False,
        }

    try:
        doc = Document(html)

        title = doc.short_title()

        soup = BeautifulSoup(
            doc.summary(),
            "html.parser"
        )

        for tag in soup(
            ["script", "style", "noscript"]
        ):
            tag.decompose()

        text = re.sub(
            r"\s+",
            " ",
            soup.get_text(
                " ",
                strip=True
            )
        )[:160000]

        return {
            "url": url,
            "title": title,
            "text": text,
            "ok": True,
        }

    except Exception as e:
        return {
            "url": url,
            "title": None,
            "text": f"PARSING_ERROR: {e}",
            "ok": False,
        }


# ============================================================
# SOURCE RANKING
# ============================================================

def quality_score(url: str):

    domain = domain_name(url)
    score = 0

    for key, weight in QUALITY_WEIGHTS.items():

        if (
            domain.endswith(
                key.replace("*", "")
            )
            or key in domain
        ):
            score = max(
                score,
                weight
            )

    return score


def is_blocked(url: str):

    url = url.lower()

    return any(
        blocked in url
        for blocked in BLOCKLIST_PARTIALS
    )


def choose_top_sources(
    results,
    per_domain_limit=PER_DOMAIN_LIMIT_DEFAULT,
    max_total=30
):

    ranked = []

    for result in results:

        url = result.get("url")

        if not url:
            continue

        if is_blocked(url):
            continue

        ranked.append(
            (
                quality_score(url),
                result
            )
        )

    ranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    kept = []
    used_domains = {}

    for _, result in ranked:

        domain = domain_name(
            result["url"]
        )

        used_domains[domain] = (
            used_domains.get(domain, 0)
        )

        if (
            used_domains[domain]
            < per_domain_limit
        ):
            kept.append(result)
            used_domains[domain] += 1

        if len(kept) >= max_total:
            break

    return kept


# ============================================================
# MODEL INSTRUCTIONS
# ============================================================

BUSINESS_SYSTEM_INSTRUCTIONS = """
You are BusinessResearcher, a neutral research analyst.

IMPORTANT RULES:

1. Use only the numbered sources provided.
2. Never invent citations.
3. Never invent URLs.
4. Never invent statistics.
5. Do not present a number unless it is supported by the supplied evidence.
6. Clearly identify uncertainty and missing evidence.
7. Prefer authoritative government, academic, standards-body and reputable institutional sources.
8. Every important factual claim should have a citation such as [1], [2] or [3].
9. Whenever numbers exist, include the value, unit, year/date and citation.
10. Do not claim a source says something unless the supplied source text supports it.

Write a structured, evidence-based report.

Use this exact overall structure:

## Key Metrics at a Glance

Provide 5–10 important metrics where evidence supports them.

Use a Markdown table with:

| # | Metric | Value | Unit | Year / Date | Source |

Do not force 10 metrics when the evidence does not contain enough reliable numbers.

## Executive Summary

Provide 6–10 concise evidence-based points with citations.

## Evidence Table

IMPORTANT:
Produce this as a proper Markdown table.

Use these exact columns:

| # | Source | Publisher | Date | Key finding | URL |
|---|---|---|---|---|---|

Include approximately 10–25 useful sources when available.

Only include sources actually provided to you.

Do not include sources whose content could not be meaningfully verified unless clearly marked.

## 5Rs Analysis

### Rules

Relevant laws, policies, regulations, standards and governance.

### Roles

Relevant organisations, stakeholders and responsibilities.

### Relationships

Relationships and dependencies between stakeholders.

### Resources

Financial, human, technical and organisational resources.

### Results

Measured results, outcomes and trends.

## Feedback Loops

Explain important positive or negative feedback loops.

## Enablers

Factors enabling progress or success.

## Barriers

Factors preventing or slowing progress.

## Consensus vs Disagreements

Explain where the evidence agrees and where it differs.

## Limits & Unknowns

Clearly describe data gaps, uncertainty, weak evidence and limitations.

## How to Verify

Explain how important findings could independently be checked.

At the END of the report, also produce a CSV version of the Evidence Table.

The CSV MUST appear exactly between these tags:

<CSV>
#,[Source Title],Publisher,Date,One-line finding,URL
...
</CSV>

Do not put Markdown formatting inside the CSV block.

Quote CSV fields when necessary.

The CSV must contain approximately 10–25 rows where suitable.
"""


# ============================================================
# BUILD MODEL PROMPT
# ============================================================

def build_business_prompt(
    topic: str,
    fetched: list,
    searched_when: str
):

    lines = []

    lines.append(
        f"Topic: {topic}"
    )

    lines.append(
        f"Searched on: {searched_when}"
    )

    lines.append(
        "\nFollow the system instructions exactly."
    )

    lines.append(
        "\nNUMBERED SOURCES:"
    )

    for i, source in enumerate(
        fetched,
        1
    ):

        title = (
            source.get("title")
            or "(no title)"
        )

        url = source.get("url", "")

        date = (
            source.get("date")
            or ""
        )

        line = (
            f"[{i}] {title} — {url}"
        )

        if date:
            line += f" ({date})"

        lines.append(line)

    lines.append(
        "\nSOURCE EXCERPTS:"
    )

    for i, source in enumerate(
        fetched,
        1
    ):

        text = (
            source.get("text")
            or source.get("snippet")
            or ""
        )[:1800]

        lines.append(
            f"\nSOURCE [{i}]\n"
            f"Title: "
            f"{source.get('title') or '(no title)'}\n"
            f"URL: {source.get('url', '')}\n"
            f"Content:\n{text}\n"
        )

    lines.append(
        """
IMPORTANT:
End the report with the CSV evidence table between:

<CSV>

and

</CSV>

Do not omit these tags.
"""
    )

    return "\n".join(lines)


# ============================================================
# GROQ CALL
# ============================================================

def ask_groq(
    client,
    model,
    prompt
):

    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content":
                BUSINESS_SYSTEM_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=MODEL_TEMPERATURE,
    )


# ============================================================
# CSV EXTRACTION
# ============================================================

def extract_csv_block(text: str):

    match = re.search(
        r"<CSV>\s*(.*?)\s*</CSV>",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not match:
        return None

    csv_raw = match.group(1).strip()

    if not csv_raw:
        return None

    try:
        df = pd.read_csv(
            StringIO(csv_raw)
        )

        if df.empty:
            return None

        return df

    except Exception:
        return None


def markdown_evidence_table_to_df(
    text: str
):

    """
    Fallback:
    Extract the Markdown Evidence Table if the
    model forgot to produce a valid CSV block.
    """

    section_match = re.search(
        r"(?:^|\n)##?\s*Evidence Table\s*\n"
        r"(.*?)(?=\n##?\s+|\n5Rs Analysis|\Z)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not section_match:
        return None

    section = section_match.group(1)

    table_lines = [
        line.strip()
        for line in section.splitlines()
        if line.strip().startswith("|")
        and line.strip().endswith("|")
    ]

    if len(table_lines) < 2:
        return None

    rows = []

    for line in table_lines:

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        # Ignore separator row:
        # |---|---|---|
        separator = all(
            bool(
                re.fullmatch(
                    r":?-{3,}:?",
                    cell.replace(" ", "")
                )
            )
            for cell in cells
        )

        if separator:
            continue

        rows.append(cells)

    if len(rows) < 2:
        return None

    header = rows[0]

    data_rows = []

    for row in rows[1:]:

        if len(row) < len(header):
            row = (
                row
                + [""] * (
                    len(header) - len(row)
                )
            )

        elif len(row) > len(header):

            # Keep additional text within the final column
            row = (
                row[:len(header) - 1]
                + [
                    " | ".join(
                        row[len(header) - 1:]
                    )
                ]
            )

        data_rows.append(row)

    try:
        df = pd.DataFrame(
            data_rows,
            columns=header
        )

        if df.empty:
            return None

        return df

    except Exception:
        return None


def create_evidence_csv(
    text: str,
    csv_path: Path
):

    # Method 1:
    # Use explicit model CSV block
    df = extract_csv_block(text)

    # Method 2:
    # Convert Markdown Evidence Table
    if df is None:
        df = markdown_evidence_table_to_df(
            text
        )

    if df is None:
        return None

    # Clean column names
    df.columns = [
        str(column)
        .strip()
        .replace("[", "")
        .replace("]", "")
        for column in df.columns
    ]

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    csv_bytes = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    return csv_bytes


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    topic: str,
    region: str,
    per_domain_limit: int,
    max_sources: int,
    include_academia: bool,
    progress_cb=None
):

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    try:
        api_key = (
            st.secrets.get("GROQ_API_KEY")
            or os.environ.get("GROQ_API_KEY")
        )

    except Exception:
        api_key = os.environ.get(
            "GROQ_API_KEY"
        )

    if not api_key:
        raise RuntimeError(
            "No GROQ_API_KEY found. "
            "Go to Streamlit → Settings → Secrets "
            "and add your GROQ_API_KEY."
        )


    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    if progress_cb:
        progress_cb(
            "Searching the web…"
        )

    base_hits = web_search_text(
        topic,
        max_results=max_sources * 3,
        region=region
    )


    # --------------------------------------------------------
    # ACADEMIC SOURCES
    # --------------------------------------------------------

    if include_academia:

        if progress_cb:
            progress_cb(
                "Adding academic sources "
                "(OpenAlex, arXiv, Crossref)…"
            )

        base_hits += search_openalex(
            topic,
            max_results=15
        )

        base_hits += search_arxiv(
            topic,
            max_results=12
        )

        base_hits += search_crossref(
            topic,
            max_results=12
        )


    if not base_hits:
        raise RuntimeError(
            "No results found. "
            "Try a broader query or "
            "switch the region to wt-wt."
        )


    # --------------------------------------------------------
    # PICK BEST SOURCES
    # --------------------------------------------------------

    picked_meta = choose_top_sources(
        base_hits,
        per_domain_limit=per_domain_limit,
        max_total=max_sources
    )


    # --------------------------------------------------------
    # FETCH SOURCES
    # --------------------------------------------------------

    if progress_cb:
        progress_cb(
            f"Fetching "
            f"{len(picked_meta)} "
            f"sources in parallel…"
        )

    fetched = [
        None
    ] * len(picked_meta)

    with ThreadPoolExecutor(
        max_workers=FETCH_CONCURRENCY
    ) as executor:

        future_map = {
            executor.submit(
                fetch_and_clean_single,
                result["url"]
            ): index

            for index, result
            in enumerate(picked_meta)
        }

        done_count = 0

        for future in as_completed(
            future_map
        ):

            index = future_map[future]

            try:
                info = future.result()

            except Exception as e:
                info = {
                    "url":
                    picked_meta[index]["url"],

                    "title": None,

                    "text":
                    f"FETCH_ERROR: {e}",

                    "ok": False,
                }

            metadata = picked_meta[index]

            if not info.get("title"):
                info["title"] = (
                    metadata.get("title")
                )

            info["date"] = (
                metadata.get("date")
            )

            info["snippet"] = (
                metadata.get("snippet")
            )

            fetched[index] = info

            done_count += 1

            if progress_cb and (
                done_count
                == len(picked_meta)
                or done_count % 2 == 0
            ):
                progress_cb(
                    f"Fetched "
                    f"{done_count}/"
                    f"{len(picked_meta)}…"
                )

    fetched = [
        source
        for source in fetched
        if source is not None
    ]


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    if progress_cb:
        progress_cb(
            "Preparing evidence…"
        )

    searched_when = now_london()

    prompt = build_business_prompt(
        topic,
        fetched,
        searched_when
    )


    # --------------------------------------------------------
    # GROQ
    # --------------------------------------------------------

    client = Groq(
        api_key=api_key
    )

    if progress_cb:
        progress_cb(
            "Asking primary model "
            f"({DEFAULT_MODEL_PRIMARY})…"
        )

    try:
        response = ask_groq(
            client,
            DEFAULT_MODEL_PRIMARY,
            prompt
        )

    except Exception as primary_error:

        if progress_cb:
            progress_cb(
                "Primary model failed. "
                "Trying fallback "
                f"({DEFAULT_MODEL_FALLBACK})…"
            )

        try:
            response = ask_groq(
                client,
                DEFAULT_MODEL_FALLBACK,
                prompt
            )

        except Exception as fallback_error:

            raise RuntimeError(
                "\n\nBoth Groq models failed.\n\n"
                f"Primary model: "
                f"{DEFAULT_MODEL_PRIMARY}\n"
                f"Error: "
                f"{primary_error}\n\n"
                f"Fallback model: "
                f"{DEFAULT_MODEL_FALLBACK}\n"
                f"Error: "
                f"{fallback_error}\n\n"
                "Check your GROQ_API_KEY "
                "and Groq model access."
            )


    text = (
        response
        .choices[0]
        .message
        .content
    )


    # --------------------------------------------------------
    # FILE NAMES
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    safe_topic = sanitize_filename(
        topic,
        90
    )

    md_path = (
        REPORT_DIR
        /
        f"{timestamp}_"
        f"{safe_topic}_"
        f"BUSINESS_WEB.md"
    )

    csv_path = (
        REPORT_DIR
        /
        f"{timestamp}_"
        f"{safe_topic}_"
        f"EvidenceTable.csv"
    )


    # --------------------------------------------------------
    # SAVE MARKDOWN
    # --------------------------------------------------------

    md_path.write_text(
        text,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # CREATE CSV
    # --------------------------------------------------------

    if progress_cb:
        progress_cb(
            "Creating evidence CSV…"
        )

    csv_bytes = create_evidence_csv(
        text,
        csv_path
    )

    if csv_bytes is None:

        # Extra reliable fallback:
        # create evidence directly from fetched sources

        fallback_rows = []

        for i, source in enumerate(
            fetched,
            1
        ):

            snippet = (
                source.get("snippet")
                or source.get("text")
                or ""
            )

            snippet = re.sub(
                r"\s+",
                " ",
                snippet
            )[:400]

            fallback_rows.append(
                {
                    "#": i,
                    "Source Title":
                    source.get("title")
                    or "(no title)",

                    "Publisher":
                    domain_name(
                        source.get("url", "")
                    ),

                    "Date":
                    source.get("date")
                    or "",

                    "One-line finding":
                    snippet,

                    "URL":
                    source.get("url")
                    or "",
                }
            )

        if fallback_rows:

            fallback_df = pd.DataFrame(
                fallback_rows
            )

            fallback_df.to_csv(
                csv_path,
                index=False,
                encoding="utf-8-sig"
            )

            csv_bytes = (
                fallback_df
                .to_csv(index=False)
                .encode("utf-8-sig")
            )

        else:
            csv_path = None


    return (
        text.encode("utf-8"),
        csv_bytes,
        md_path,
        csv_path,
    )


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="BizInsights",
    page_icon="📊",
    layout="centered"
)


st.title(
    "Research & Evidence Generator"
)


st.caption(
    "Web + Academic search • "
    "Stats-first report • "
    "Evidence table CSV"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader(
        "Options"
    )

    region = st.selectbox(
        "Region",
        [
            "wt-wt",
            "uk-en",
            "us-en",
            "in-en",
            "pk-en",
        ],
        index=0,
    )

    max_sources = st.slider(
        "Max sources",
        min_value=6,
        max_value=40,
        value=MAX_SOURCES_DEFAULT,
        step=2,
    )

    per_domain = st.slider(
        "Per-domain limit",
        min_value=1,
        max_value=4,
        value=PER_DOMAIN_LIMIT_DEFAULT,
    )

    include_academia = st.checkbox(
        "Include academic sources "
        "(OpenAlex, arXiv, Crossref)",
        value=True,
    )

    st.markdown("---")

    st.caption(
        "Primary model: "
        f"{DEFAULT_MODEL_PRIMARY}"
    )

    st.caption(
        "Fallback model: "
        f"{DEFAULT_MODEL_FALLBACK}"
    )


# ============================================================
# USER INPUT
# ============================================================

topic = st.text_input(
    "Your prompt / topic",
    value="",
    placeholder=(
        "e.g. NHS AI adoption metrics in 2024; "
        "UK EV charging policy; "
        "Cybersecurity in autonomous vehicles"
    ),
)


run = st.button(
    "Run",
    type="primary"
)


# ============================================================
# OUTPUT AREAS
# ============================================================

log_area = st.empty()
report_area = st.empty()

download_col1, download_col2 = (
    st.columns(2)
)


# ============================================================
# LOGGING
# ============================================================

def log(message):

    previous = st.session_state.get(
        "log_text",
        ""
    )

    st.session_state["log_text"] = (
        previous
        + message
        + "\n"
    )

    log_area.code(
        st.session_state[
            "log_text"
        ]
    )


# ============================================================
# RUN
# ============================================================

if run:

    st.session_state[
        "log_text"
    ] = ""

    if not topic.strip():

        st.warning(
            "Please enter a topic first."
        )

    else:

        try:

            log(
                "Starting research…"
            )

            (
                md_bytes,
                csv_bytes,
                md_path,
                csv_path,
            ) = run_pipeline(
                topic=topic.strip(),
                region=region,
                per_domain_limit=per_domain,
                max_sources=max_sources,
                include_academia=include_academia,
                progress_cb=log,
            )

            log(
                "✅ Research completed."
            )

            log(
                "✅ Saved Markdown "
                f"on server: {md_path}"
            )

            if csv_path and csv_bytes:

                log(
                    "✅ Evidence CSV created."
                )

            else:

                log(
                    "⚠️ CSV could not be created."
                )


            # -----------------------------------------------
            # DISPLAY REPORT
            # -----------------------------------------------

            report_text = (
                md_bytes.decode(
                    "utf-8"
                )
            )

            # Hide the raw <CSV> block from the
            # on-screen report
            display_report = re.sub(
                r"<CSV>.*?</CSV>",
                "",
                report_text,
                flags=(
                    re.DOTALL
                    |
                    re.IGNORECASE
                )
            ).strip()

            report_area.markdown(
                display_report
            )


            # -----------------------------------------------
            # DOWNLOAD FILE NAMES
            # -----------------------------------------------

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y%m%d-%H%M%S"
                )
            )

            safe_topic = sanitize_filename(
                topic or "report",
                60
            )

            markdown_filename = (
                f"{timestamp}_"
                f"{safe_topic}_"
                f"BUSINESS_WEB.md"
            )

            csv_filename = (
                f"{timestamp}_"
                f"{safe_topic}_"
                f"EvidenceTable.csv"
            )


            # -----------------------------------------------
            # MARKDOWN DOWNLOAD
            # -----------------------------------------------

            with download_col1:

                st.download_button(
                    label=(
                        "⬇️ Download "
                        "Markdown Report"
                    ),
                    data=md_bytes,
                    file_name=markdown_filename,
                    mime="text/markdown",
                )


            # -----------------------------------------------
            # CSV DOWNLOAD
            # -----------------------------------------------

            if csv_bytes:

                with download_col2:

                    st.download_button(
                        label=(
                            "⬇️ Download "
                            "Evidence Table (CSV)"
                        ),
                        data=csv_bytes,
                        file_name=csv_filename,
                        mime="text/csv",
                    )


        except Exception as e:

            log(
                f"❌ Error: {e}"
            )
