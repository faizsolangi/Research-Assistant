import os
import re
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader

# OpenAI python SDK (v1+)
from openai import OpenAI


# -----------------------------
# Safety / formatting utilities
# -----------------------------

SECTION_TITLES = [
    "1. Structured Research Summary",
    "2. Key Gaps",
    "3. Methods",
    "4. Related Work",
    "5. APA References",
    "6. Future Directions",
]

def extract_text_from_pdf(file_bytes: bytes, max_chars: int = 120_000) -> str:
    """Extract text from a PDF (no OCR)."""
    try:
        reader = PdfReader(file_bytes)
        chunks = []
        total = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            if not text.strip():
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            text = text[:remaining]
            chunks.append(text)
            total += len(text)
        return "\n\n".join(chunks).strip()
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

def join_nonempty(parts: List[Tuple[str, str]]) -> str:
    """Join labeled sections where content exists."""
    out = []
    for label, content in parts:
        c = (content or "").strip()
        if c:
            out.append(f"{label}\n{c}")
    return "\n\n".join(out).strip()

def validate_section_order(text: str) -> Tuple[bool, str]:
    """
    Ensure all required section titles exist exactly once, in exact order.
    This is a *format* check, not a truth check.
    """
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

    # Light sanity: ensure headings aren't duplicated
    for title in SECTION_TITLES:
        if text.count(title) != 1:
            return False, f"Section title duplicated or malformed: {title}"

    return True, "OK"

def harden_with_correction_prompt(bad_output: str) -> str:
    """
    If the model drifts, we re-ask it to fix structure only, without adding facts.
    """
    return f"""
Your previous output failed formatting requirements.

You MUST return the output again with:
- The 6 sections present EXACTLY ONCE
- Titles EXACTLY as written
- Sections in EXACT order
- No extra sections, no merged sections

Do NOT add new facts/citations. Only restructure/trim.

Here is your previous output:
{bad_output}
""".strip()


# -----------------------------
# Prompt
# -----------------------------

SYSTEM_PROMPT = """You are a Research Assistant AI designed to support academic researchers.
Your role is to organize, summarize, and structure research content without inventing facts, data, citations, or interpretations.

You are NOT:
- a scientific authority
- a co-author
- a decision-maker
- a source of new experimental claims

You ARE:
- a structuring assistant
- a clarity and organization tool
- a literature-mapping aide

INPUT EXPECTATIONS:
The user provides research topic(s), abstracts, DOIs, uploaded PDFs, or notes.
If information is missing or insufficient, explicitly state limitations.

STRICT PROHIBITIONS:
- No fabricated citations, DOIs, authors, years, journals, or metrics
- No inferred methods or results
- No claims of novelty, superiority, or consensus

OUTPUT FORMAT (MANDATORY):

1. Structured Research Summary
2. Key Gaps
3. Methods
4. Related Work
5. APA References
6. Future Directions

Rules:
- Preserve section order exactly
- Do not merge or omit sections
- If data is missing, write “Insufficient information provided.”

STYLE:
- Formal academic tone
- Reviewer-neutral
- Cautious language
- No promotional wording

FINAL CHECK:
Ensure all claims are grounded in user-provided material only.
"""

def build_user_prompt(topic: str, notes: str, abstracts: str, dois: str, allowed_refs: str, pdf_text: str) -> str:
    """
    We explicitly tell the model what it's allowed to cite.
    If allowed_refs and dois are empty, APA References MUST be "Insufficient information provided."
    """
    topic = (topic or "").strip()
    notes = (notes or "").strip()
    abstracts = (abstracts or "").strip()
    dois = (dois or "").strip()
    allowed_refs = (allowed_refs or "").strip()
    pdf_text = (pdf_text or "").strip()

    allowed_block = allowed_refs if allowed_refs else "NONE PROVIDED"
    doi_block = dois if dois else "NONE PROVIDED"

    return f"""
TASK:
Using ONLY the user-provided material below, produce the required 6-section output.

CITATION RULE (STRICT):
- You may ONLY include items in "User-Provided DOIs" or "User-Provided Allowed References" inside section 5 (APA References).
- If BOTH lists are empty, section 5 MUST be exactly: "Insufficient information provided."
- Do NOT generate or guess missing authors/years/journals.
- If you cannot safely populate any section, write: "Insufficient information provided."

User Topic:
{topic if topic else "Insufficient information provided."}

User Notes:
{notes if notes else "Insufficient information provided."}

User Abstracts:
{abstracts if abstracts else "Insufficient information provided."}

User-Provided DOIs (allowed to repeat/cite in section 5 only):
{doi_block}

User-Provided Allowed References (allowed to repeat/cite in section 5 only):
{allowed_block}

Extracted PDF Text (if any; may be incomplete):
{pdf_text if pdf_text else "Insufficient information provided."}

Now produce the output in the mandatory section order and titles.
""".strip()


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="Research Assistant (No Hallucinations)", layout="wide")

st.title("Research Assistant AI (Strict, No Fabrication)")
st.caption("Structures and summarizes ONLY what you provide. If you provide nothing, it will politely produce… nothing.")

with st.sidebar:
    st.header("Model & Runtime")
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url = st.text_input("Base URL (optional)", value=os.getenv("OPENAI_BASE_URL", ""))
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    temperature = st.slider("Temperature", 0.0, 0.6, 0.2, 0.05)
    max_tokens = st.slider("Max tokens", 500, 4000, 2000, 100)
    st.divider()
    st.write("Render tip: set env vars in Render dashboard or render.yaml.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    topic = st.text_input("Research topic(s)", placeholder="e.g., Inorganic passivators for perovskite solar cells (2020–2025)")
    notes = st.text_area("Notes", height=160, placeholder="Paste your notes, bullet points, snippets…")
    abstracts = st.text_area("Abstracts", height=160, placeholder="Paste abstracts (one or many)…")
    dois = st.text_area("DOIs (optional)", height=120, placeholder="Paste DOI list (one per line). Only these may be cited.")
    allowed_refs = st.text_area(
        "Allowed references (APA or any format; optional)",
        height=140,
        placeholder="Paste references you ALLOW the assistant to include in APA References. If empty, it must say 'Insufficient information provided.'"
    )
    uploaded_pdfs = st.file_uploader("Upload PDFs (optional)", type=["pdf"], accept_multiple_files=True)

with col2:
    st.subheader("Output")
    generate = st.button("Generate structured output", type="primary")

    if generate:
        if not api_key:
            st.error("Missing OPENAI_API_KEY. Add it as an environment variable on Render or paste it in the sidebar.")
            st.stop()

        pdf_texts = []
        if uploaded_pdfs:
            for f in uploaded_pdfs:
                try:
                    pdf_bytes = f.read()
                    text = extract_text_from_pdf(pdf_bytes)
                    pdf_texts.append(f"[PDF: {f.name}]\n{text}")
                except Exception as e:
                    pdf_texts.append(f"[PDF: {f.name}]\n[Failed to read: {e}]")

        combined_pdf_text = "\n\n".join(t for t in pdf_texts if t.strip()).strip()

        user_prompt = build_user_prompt(
            topic=topic,
            notes=notes,
            abstracts=abstracts,
            dois=dois,
            allowed_refs=allowed_refs,
            pdf_text=combined_pdf_text,
        )

        client = OpenAI(api_key=api_key, base_url=base_url or None)

        with st.spinner("Generating (responsibly)…"):
            # First attempt
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            output = resp.choices[0].message.content or ""

            ok, reason = validate_section_order(output)

            # If formatting failed, do ONE correction pass
            if not ok:
                correction = harden_with_correction_prompt(output)
                resp2 = client.chat.completions.create(
                    model=model,
                    temperature=0.0,  # keep it rigid
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "user", "content": correction},
                    ],
                )
                output2 = resp2.choices[0].message.content or ""
                ok2, reason2 = validate_section_order(output2)

                if ok2:
                    output = output2
                    reason = "OK (after correction pass)"
                else:
                    st.warning(f"Output still failed strict formatting: {reason2}")

        st.text_area("Structured Output", value=output, height=520)

        st.caption(f"Format check: {reason}")

        # Optional: quick compliance hint (not enforcement)
        if re.search(r"\b(we (show|demonstrate)|novel|state[- ]of[- ]the[- ]art|breakthrough|best)\b", output, re.I):
            st.info("Heads-up: output contains language that *might* sound promotional. Review before using.")


st.divider()
with st.expander("What this app will NOT do (because reality matters)"):
    st.write(
        "- It will not invent citations.\n"
        "- It will not guess missing methods/results.\n"
        "- It will not pretend your PDFs are complete.\n"
        "- It will not become your co-author."
    )
