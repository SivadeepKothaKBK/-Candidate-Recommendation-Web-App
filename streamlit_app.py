# filepath: app/streamlit_app.py
from typing import List
import io

import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from docx import Document

# ---------- UI helpers ----------
def inject_modern_css() -> None:
    st.markdown(
        r"""
        <style>
        :root{
          --bg-1:#0b0c10; --bg-2:#12131a;
          --card:rgba(255,255,255,0.06); --card-border:rgba(255,255,255,0.12);
          --text:#eaeaea; --muted:#b9bcc5;
          --cta:#ff7a00;                 /* orange */
          --cta-hover:#000000;           /* hover: black */
        }
        .stApp{
          background: radial-gradient(1200px 600px at 20% 0%, #1a1c26 12%, transparent 60%),
                      radial-gradient(1000px 600px at 90% 10%, #151d29 10%, transparent 60%),
                      linear-gradient(180deg, var(--bg-1), var(--bg-2));
          color: var(--text);
        }
        .block-container{max-width:1100px; padding-top:1rem;}

        /* Top-right Google-color badge */
        .maker-badge{
          position:fixed; top:10px; right:16px; z-index:10000;
          padding:6px 12px; border-radius:999px;
          background:rgba(255,255,255,0.10);
          border:1px solid var(--card-border);
          backdrop-filter:blur(8px);
          font-weight:700; font-size:12px; letter-spacing:.3px;
          pointer-events:none;   /* do not block Streamlit menu clicks */
        }

        /* Title (medium, no symbol) */
        .hero-wrap{ background:var(--card); border:1px solid var(--card-border);
          border-radius:18px; padding:16px 18px; margin-bottom:16px; backdrop-filter:blur(10px);}
        .hero-title{
          text-align:center; margin:0;
          font-weight:700; letter-spacing:-0.02em;
          font-size:clamp(22px, 3.2vw, 34px); line-height:1.15;
          background:linear-gradient(90deg,#ffffff,#c9d6ff 60%);
          -webkit-background-clip:text; background-clip:text; color:transparent;
        }

        /* Inputs (no white boxes) */
        .stTextArea textarea{
          background:rgba(20,22,30,0.75)!important; color:var(--text)!important;
          border-radius:14px!important; border:1px solid var(--card-border)!important;
          caret-color:var(--text); font-size:14px;
        }
        .stTextArea textarea:focus{
          outline:none; border-color:#5ca7ff!important;
          box-shadow:0 0 0 3px rgba(92,167,255,0.25)!important;
        }
        [data-testid="stFileUploader"] > div{
          background:rgba(20,22,30,0.75)!important; color:var(--text)!important;
          border:1px solid var(--card-border)!important; border-radius:14px!important; padding:10px!important;
        }

        /* Buttons: orange, hover black. Force for all button kinds. */
        .stButton > button,
        button[kind],
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"]{
          background:var(--cta) !important;
          color:#ffffff !important;
          border:0 !important; border-radius:999px !important;
          padding:12px 20px !important; font-weight:700 !important;
          box-shadow:0 8px 22px rgba(255,122,0,.25) !important;
          transition: background-color .15s ease, color .15s ease !important;
        }
        .stButton > button:hover,
        button[kind]:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="baseButton-primary"]:hover{
          background:var(--cta-hover) !important;
          color:#ffffff !important;
        }

        /* Results */
        .result-card{ background:var(--card); border:1px solid var(--card-border);
          border-radius:18px; padding:16px; margin:14px 0; backdrop-filter:blur(8px);}
        .result-head{ display:flex; align-items:center; justify-content:space-between; gap:12px;}
        .result-title{ font-weight:700; font-size:18px;}
        .score-bar{ height:10px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden;}
        .score-fill{ height:100%; background:linear-gradient(90deg,#ffb86b,#ff7a00);}
        .pill{ font-size:12px; padding:4px 10px; border-radius:999px;
          background:rgba(255,255,255,0.08); border:1px solid var(--card-border);}
        </style>
        """,
        unsafe_allow_html=True,
    )

def google_badge_html(name: str = "SIVADEEP KOTHA") -> str:
    # Google brand colors
    colors = ["#4285F4", "#DB4437", "#F4B400", "#4285F4", "#0F9D58", "#DB4437"]
    spans, i = [], 0
    for ch in name:
        if ch == " ":
            spans.append("&nbsp;")
        else:
            spans.append(f'<span style="color:{colors[i % len(colors)]}">{ch}</span>')
            i += 1
    return f'<div class="maker-badge">{"".join(spans)}</div>'

def hero():
    st.markdown(google_badge_html(), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-wrap">
          <h1 class="hero-title">Candidate Recommendation Web App</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Text extraction ----------
@st.cache_resource(show_spinner=False)
def load_embedding_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(name)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    parts = []
    for p in reader.pages:
        try:
            parts.append(p.extract_text() or "")
        except Exception:
            pass
    return "\n".join(parts).strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()

def get_text_from_uploaded_file(up) -> str:
    name = (up.name or "").lower()
    data = up.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")  # prevents crash

def split_to_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r"(?<=[.!?\n])\s+", (text or "").strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# ---------- Scoring ----------
def compute_scores(job_text: str, resumes: List[str], model: SentenceTransformer):
    r_embs = model.encode(resumes, convert_to_tensor=True, show_progress_bar=False)
    j_emb = model.encode(job_text, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(j_emb, r_embs)[0]
    import torch
    return sims.detach().cpu()

def reasons_for_resume(job_text: str, resume_text: str, model: SentenceTransformer, top_n: int = 3) -> List[str]:
    sents = split_to_sentences(resume_text)
    if not sents:
        return ["(No good sentences found)"]
    s_embs = model.encode(sents, convert_to_tensor=True, show_progress_bar=False)
    j_emb = model.encode(job_text, convert_to_tensor=True, show_progress_bar=False)
    sims = util.cos_sim(j_emb, s_embs)[0].detach().cpu()
    import torch
    k = min(top_n, sims.shape[0])
    topk = torch.topk(sims, k=k)
    idxs = topk.indices.tolist()
    return [sents[i] for i in idxs]

# ---------- App ----------
def main():
    st.set_page_config(page_title="Candidate Recommendation Web App", layout="wide")
    inject_modern_css()
    hero()

    with st.form("match_form", clear_on_submit=False):
        left, right = st.columns([2, 1])
        with left:
            job_text = st.text_area(
                "Job description",
                key="job_text",
                height=240,
                placeholder="Example: Senior Python engineer with NLP, embeddings, and deployment experience.",
            )
            submit_top = st.form_submit_button("Match")
        with right:
            top_k = st.slider("Top results", min_value=1, max_value=10, value=5)
            st.caption("Models are cached for speed.")

        uploaded = st.file_uploader(
            "Upload multiple resumes (PDF / DOCX / TXT)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
        )
        pasted_resume = st.text_area("Or paste one resume (optional)")

        submit_bottom = st.form_submit_button("Match candidates", use_container_width=True)

    submitted = bool(submit_top or submit_bottom)
    if not submitted:
        return
    if not job_text:
        st.warning("Add a job description first.")
        return
    if not uploaded and not pasted_resume:
        st.warning("Upload at least one resume or paste one below.")
        return

    with st.spinner("Loading model and computing..."):
        model = load_embedding_model()
        resume_texts, resume_ids = [], []
        for up in (uploaded or []):
            txt = get_text_from_uploaded_file(up)
            if txt:
                resume_texts.append(txt)
                resume_ids.append(up.name)
        if pasted_resume:
            resume_texts.append(pasted_resume)
            resume_ids.append("pasted_resume_1")
        if not resume_texts:
            st.error("No readable resumes found.")
            return

        sims = compute_scores(job_text, resume_texts, model)
        import torch
        k = min(int(top_k), len(resume_texts))
        topk = torch.topk(sims, k=k)
        indices = topk.indices.tolist()
        scores = topk.values.tolist()

    st.success(f"Found {len(resume_texts)} resumes - showing top {k}")

    for idx, score in zip(indices, scores):
        idx = int(idx)
        percent = max(0.0, min(100.0, float(score) * 100.0))
        why = reasons_for_resume(job_text, resume_texts[idx], model, top_n=3)
        title = f"{resume_ids[idx]}"

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-head">
                <div class="result-title">{title}</div>
                <div class="pill">{percent:.2f}% match</div>
              </div>
              <div class="score-bar" style="margin:10px 0 6px 0;">
                <div class="score-fill" style="width:{percent:.2f}%"></div>
              </div>
              <div style="color:var(--muted); font-size:14px; margin-top:8px;">
                <ul style="margin: 0 0 0 18px;">
                  {''.join(f'<li>{s}</li>' for s in why)}
                </ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View first 1000 characters of resume"):
            snippet = resume_texts[idx][:1000]
            if len(resume_texts[idx]) > 1000:
                snippet += "..."
            st.code(snippet)

    st.sidebar.caption("Made with Streamlit and sentence-transformers")

if __name__ == "__main__":
    main()
