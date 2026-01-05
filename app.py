import streamlit as st
import PyPDF2
import pandas as pd
import plotly.express as px
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="💼", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stTextArea>div>div>textarea { border-radius: 10px; }
    .status-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- Functions ---
def extract_text(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.lower()

def extract_skills(text):
    skills_db = ['python', 'java', 'javascript', 'sql', 'react', 'node', 'html', 'css', 'machine learning', 'git', 'aws', 'docker']
    found = {skill.title() for skill in skills_db if re.search(r'\b' + re.escape(skill) + r'\b', text)}
    return found

# --- UI Header ---
st.title("💼 AI Recruitment Pro: Smart Sorter")
st.info("Upload multiple resumes to find the best match based on skills and relevance.")

# --- Layout ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("🎯 Job Details")
    with st.container():
        jd_text = st.text_area("Job Description", height=250, placeholder="Paste requirements here...")
        uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
        analyze_btn = st.button("Run AI Analysis")

if analyze_btn:
    if jd_text and uploaded_files:
        target_skills = extract_skills(jd_text.lower())
        
        results = []
        for file in uploaded_files:
            resume_content = extract_text(file)
            
            # Similarity
            vec = TfidfVectorizer(stop_words='english').fit_transform([jd_text.lower(), resume_content])
            score = cosine_similarity(vec[0:1], vec[1:])[0][0]
            
            # Skills
            cand_skills = extract_skills(resume_content)
            missing = target_skills - cand_skills
            
            results.append({
                "Candidate": file.name,
                "Match %": round(score * 100, 2),
                "Missing Skills": ", ".join(missing) if missing else "None"
            })
        
        df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)

        with col2:
            st.subheader("📊 Ranking Results")
            # Chart
            fig = px.bar(df, x="Match %", y="Candidate", orientation='h', 
                         color="Match %", color_continuous_scale='Blues', text_auto=True)
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.write("### Detailed Breakdown")
            st.table(df)
            st.balloons()
    else:
        st.warning("Please provide JD and Resumes first.")