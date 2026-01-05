import streamlit as st
import PyPDF2
import pandas as pd
import plotly.express as px
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper Functions ---
def extract_contact_info(text):
    email = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phone = re.findall(r'[\+\d]?[\d\-\s\(]{10,15}\d', text)
    return (email[0] if email else "N/A"), (phone[0] if phone else "N/A")

def extract_experience(text):
    exp_pattern = re.findall(r'(\d+\.?\d*)\s*(?:+)?\s*(?:years?|yrs?)\s*(?:experience|exp)?', text, re.IGNORECASE)
    if exp_pattern:
        return max([float(x) for x in exp_pattern])
    return 0

def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# --- Streamlit UI ---
st.set_page_config(page_title="AI Recruitment Tool", layout="wide")

with st.sidebar:
    st.title("⚙️ Recruitment Panel")
    jd_input = st.text_area("Job Description", height=200, placeholder="Define the role...")
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)
    min_exp = st.number_input("Min Experience Required", 0, 20, 2)
    analyze_btn = st.button("🔍 Run Full Analysis")

if analyze_btn and jd_input and uploaded_files:
    results = []
    for file in uploaded_files:
        raw_text = extract_text(file)
        # NLP Processing
        vec = TfidfVectorizer().fit_transform([jd_input.lower(), raw_text.lower()])
        score = cosine_similarity(vec[0:1], vec[1:])[0][0]
        
        email, phone = extract_contact_info(raw_text)
        exp = extract_experience(raw_text.lower())
        
        results.append({
            "Candidate Name": file.name,
            "Match %": round(score * 100, 2),
            "Experience": exp,
            "Email": email,
            "Phone": phone,
            "Status": "Shortlisted" if exp >= min_exp and score > 0.3 else "Rejected"
        })
    
    df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)

    # --- Results Dashboard ---
    st.header("📊 Screening Dashboard")
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Resumes", len(uploaded_files))
    c2.metric("Shortlisted", len(df[df['Status'] == 'Shortlisted']))
    c3.metric("Avg Match %", f"{round(df['Match %'].mean(),1)}%")

    st.divider()

    # Table & Chart
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.dataframe(df, use_container_width=True)
        # --- DOWNLOAD FEATURE ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Recruitment Report (CSV)", data=csv, file_name="shortlisted_candidates.csv", mime="text/csv")
    
    with col_right:
        fig = px.pie(df, names='Status', title="Candidate Selection Ratio", color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

    st.balloons()