import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app layout
st.title("🔍 AI Resume Screening & Ranking System")
st.markdown("Upload multiple PDF resumes and provide a job description to rank the resumes based on relevance.")

# Input for job description
job_description = st.text_area("📄 Enter the Job Description")

# File upload for resumes
uploaded_files = st.file_uploader("📂 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

# Process and rank resumes if both job description and files are provided
if uploaded_files and job_description:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    # Rank resumes based on similarity scores
    ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

    # Display ranked resumes
    st.subheader("🏆 Ranked Resumes")
    for i, (file, score) in enumerate(ranked_resumes, start=1):
        st.write(f"{i}. {file.name} - Score: {score * 100:.2f}%")
