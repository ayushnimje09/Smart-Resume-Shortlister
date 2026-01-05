import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF se text nikalne wala function
def get_pdf_text(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# --- STEP A: Job Description (Yaha apni requirement likhein) ---
job_description = "We need a Python developer who knows Data Science and SQL."

# --- STEP B: Resumes Load Karein ---
path = "./resumes/"
resume_files = [f for f in os.listdir(path) if f.endswith('.pdf')]

if not resume_files:
    print("Error: 'resumes' folder mein kuch PDF files daaliye!")
else:
    # Saare resumes ka text extract karna
    resume_texts = [get_pdf_text(os.path.join(path, f)) for f in resume_files]
    
    # Matching Logic
    all_content = [job_description] + resume_texts
    vectorizer = TfidfVectorizer().fit_transform(all_content)
    vectors = vectorizer.toarray()
    
    # Compare JD (index 0) with all resumes
    job_desc_vector = vectors[0:1]
    resume_vectors = vectors[1:]
    scores = cosine_similarity(job_desc_vector, resume_vectors)[0]

    # Results Print Karna
    print("\n--- RESUME RANKING RESULTS ---")
    for i, score in enumerate(scores):
        print(f"File: {resume_files[i]} | Match Score: {round(score*100, 2)}%")