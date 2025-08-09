# Candidate-Recommendation-Web-App

Live app: https://sivadeepkotha.streamlit.app


---

## Approach (how it works)

1) Read the text  
- PDFs via PyPDF2, DOCX via python-docx, TXT as plain text.  
- If a file is not UTF-8, we fall back to latin-1 to avoid crashes.

2) Turn text into vectors (embeddings)  
- Model: `sentence-transformers/all-MiniLM-L6-v2`.  
- We embed the job description once.  
- We embed each resume (full text) once.

3) Score each resume against the job  
- Compute cosine similarity between the job vector and each resume vector.  
- Higher similarity means the resume content is closer to the job needs.  
- We sort by this score and show the top K (user chooses 1–10).

4) Explain “why this person fits”  
- Split each resume into sentences.  
- Embed those sentences and score each one vs the job.  
- Pick the top 3 sentences as quick evidence bullets.  
- This is a lightweight, deterministic explanation (no external LLM calls).

That’s it: extract text → embed → cosine similarity → top matches + reasons.

---

## What “top resumes” means here

- We rank by cosine similarity between the job description embedding and the whole-resume embedding.  
- Cosine similarity is a number roughly between −1 and 1. We display it as a percentage for readability.  
- It’s a strong, fast baseline for relevance, but it is not a hiring decision signal.

---

## Assumptions and limitations

- Resumes and job description are in English or close.  
- PDFs must contain selectable text (scanned PDFs may need OCR).  
- The model is general-purpose; it does not enforce hard requirements (years of experience, location, visa, etc.).  
- Scores are relative to the provided job text; vague jobs produce vague rankings.  
- No database; files are processed in memory for the current session only.

---

## Privacy

- Files are handled in-memory by Streamlit during the session.  
- No resumes are stored server-side by this app.  
- Remove sensitive data before sharing publicly.

---

## Quick start (local)

```bash
# Python 3.11 recommended
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
