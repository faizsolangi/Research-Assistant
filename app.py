import os
import requests
from typing import Tuple

import streamlit as st
from openai import OpenAI


SECTION_TITLES = [
    "1. Structured Research Summary",
    "2. Key Gaps",
    "3. Methods",
    "4. Related Work",
    "5. APA References",
    "6. Future Directions",
]

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def semantic_scholar_search(query: str, limit: int = 8) -> str:
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,journal,isOpenAccess,url"
    }
    try:
        r = requests.get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        return f"[Literature search failed: {e}]"

    records = []
    for p in data:
        if not p.get("isOpenAccess"):
            continue
        title = p.get("title", "Unknown title")
        year = p.get("year", "n.d.")
        journal = (p.get("journal") or {}).get("name", "Unknown journal")
        authors = ", ".join(a.get("name") for a in p.get("authors", [])[:5])
        abstract = p.get("abstract", "Insufficient information provided.")
        url = p.get("url", "")
        records.append(
            f"Title: {title}\nAuthors: {authors}\nYear: {year}\nJournal: {journal}\nURL: {url}\nAbstract: {abstract}"
        )

    return "\n\n---\n\n".join(records) if records else "Insufficient information provided."


def validate_section_order(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "Empty response."

    positions = []
    for title in SECTION_TITLES:
        idx = text.find(title)
        if idx == -1:
            return False, f"Missing required section title: {title}"
        positions.append(idx)

    if positions != sorted(positions):
        return False, "Section titles are present but not in the required order."

    return True, "OK"


SYSTEM_PROMPT = """You are a Research Assistant AI designed to support academic researchers.
Your role is to organize, summarize, and structure research content without inventing facts, data, citations, or interpretations.

ALLOWED SOURCE:
- Open-access literature retrieved via Semantic Scholar search results provided below

STRICT PROHIBITIONS:
- No fabricated citations
- No inferred experimental results
- No claims beyond provided abstracts

OUTPUT FORMAT (MANDATORY):

1. Structured Research Summary
2. Key Gaps
3. Methods
4. Related Work
5. APA References
6. Future Directions
"""


def build_user_prompt(topic: str, literature_block: str) -> str:
    return f"""TASK:
Using ONLY the open-access literature records provided below, produce the required output.

Research Topic:
{topic}

Open-Access Literature Records:
{literature_block}
"""


st.set_page_config(page_title="Research Assistant (Open-Access Search)", layout="wide")
st.title("Research Assistant AI — Open‑Access Literature Mode")

api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
model = st.sidebar.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

topic = st.text_input("Research topic", placeholder="e.g., inorganic passivation in perovskite solar cells")
search_limit = st.slider("Number of open‑access papers", 3, 15, 8)

if st.button("Search & Generate"):
    if not api_key:
        st.error("Missing OPENAI_API_KEY")
        st.stop()

    with st.spinner("Searching open‑access literature..."):
        literature_block = semantic_scholar_search(topic, search_limit)

    user_prompt = build_user_prompt(topic, literature_block)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=2200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    output = response.choices[0].message.content
    ok, reason = validate_section_order(output)

    st.text_area("Structured Output", output, height=520)
    st.caption(f"Format check: {reason}")

    with st.expander("Open‑Access Papers Retrieved"):
        st.text_area("Literature records", literature_block, height=350)
