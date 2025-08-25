import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import io
import PyPDF2
import docx
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import re

# Page configuration
st.set_page_config(
    page_title="Resume Matcher AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ResumeMatcherAI:
    def __init__(self):
        self.client = None
        self.job_descriptions = []
        self.resumes = []
        self.embeddings_cache = {}
        
    def initialize_openai(self, api_key: str):
        """Initialize OpenAI client"""
        try:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Error initializing OpenAI: {str(e)}")
            return False
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get OpenAI embeddings for text"""
        if not self.client:
            return np.array([])
        
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            self.embeddings_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return np.array([])
    
    def calculate_similarity_score(self, resume_text: str, job_text: str) -> float:
        """Calculate similarity score between resume and job description"""
        resume_embedding = self.get_embeddings(resume_text)
        job_embedding = self.get_embeddings(job_text)
        
        if resume_embedding.size == 0 or job_embedding.size == 0:
            return 0.0
        
        similarity = cosine_similarity(
            resume_embedding.reshape(1, -1),
            job_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity * 100)  # Convert to percentage
    
    def extract_skills_and_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract skills and keywords using GPT-4"""
        if not self.client:
            return {"skills": [], "keywords": []}
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert HR assistant. Extract technical skills, soft skills, and important keywords from the given text. Return as JSON with 'skills' and 'keywords' arrays."},
                    {"role": "user", "content": f"Extract skills and keywords from this text:\n\n{text[:2000]}"}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            st.error(f"Error extracting skills: {str(e)}")
            return {"skills": [], "keywords": []}
    
    def generate_cover_letter(self, resume_text: str, job_description: str, candidate_name: str = "Candidate") -> str:
        """Generate tailored cover letter using GPT-4"""
        if not self.client:
            return "OpenAI client not initialized."
        
        try:
            prompt = f"""
            Generate a professional cover letter based on the following:
            
            Candidate Name: {candidate_name}
            Resume: {resume_text[:1500]}
            Job Description: {job_description[:1500]}
            
            Requirements:
            - Professional tone
            - Highlight relevant experience
            - Show enthusiasm for the role
            - Keep it concise (3-4 paragraphs)
            - Include specific examples from the resume
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert career counselor who writes compelling cover letters."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating cover letter: {str(e)}")
            return "Error generating cover letter."
    
    def generate_improvement_suggestions(self, resume_text: str, job_description: str) -> str:
        """Generate resume improvement suggestions using GPT-4"""
        if not self.client:
            return "OpenAI client not initialized."
        
        try:
            prompt = f"""
            Analyze this resume against the job description and provide specific improvement suggestions:
            
            Resume: {resume_text[:1500]}
            Job Description: {job_description[:1500]}
            
            Provide suggestions for:
            1. Missing skills to add
            2. Keywords to include
            3. Experience to emphasize
            4. Format improvements
            5. Content gaps to address
            
            Be specific and actionable.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert resume consultant providing detailed, actionable feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
            return "Error generating suggestions."

# Initialize the app
@st.cache_resource
def get_resume_matcher():
    return ResumeMatcherAI()

def main():
    st.markdown('<h1 class="main-header">🎯 Resume Matcher AI</h1>', unsafe_allow_html=True)
    st.markdown("**Semantic resume ranking with AI-powered insights**")
    
    matcher = get_resume_matcher()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key for embeddings and GPT-4"
        )
        
        if api_key and not matcher.client:
            if matcher.initialize_openai(api_key):
                st.success("✅ OpenAI initialized!")
            else:
                st.error("❌ Failed to initialize OpenAI")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📋 Navigation")
        page = st.selectbox(
            "Select Page",
            ["Resume Matching", "Bulk Analysis", "Cover Letter Generator", "Resume Insights"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if page == "Resume Matching":
        show_resume_matching(matcher)
    elif page == "Bulk Analysis":
        show_bulk_analysis(matcher)
    elif page == "Cover Letter Generator":
        show_cover_letter_generator(matcher)
    elif page == "Resume Insights":
        show_resume_insights(matcher)

def show_resume_matching(matcher):
    st.header("🎯 Resume-Job Matching")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Resume")
        resume_file = st.file_uploader(
            "Choose resume file",
            type=['pdf', 'docx', 'txt'],
            key="resume_upload"
        )
        
        resume_text = ""
        if resume_file:
            if resume_file.type == "application/pdf":
                resume_text = matcher.extract_text_from_pdf(resume_file)
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = matcher.extract_text_from_docx(resume_file)
            else:
                resume_text = str(resume_file.read(), "utf-8")
            
            if resume_text:
                st.success("✅ Resume uploaded successfully!")
                with st.expander("Preview Resume Text"):
                    st.text_area("Resume Content", resume_text[:500] + "...", height=200, disabled=True)
    
    with col2:
        st.subheader("💼 Job Description")
        job_text = st.text_area(
            "Paste job description here",
            placeholder="Enter the complete job description...",
            height=200
        )
        
        if job_text:
            st.success("✅ Job description added!")
    
    if resume_text and job_text and matcher.client:
        if st.button("🔍 Analyze Match", type="primary"):
            with st.spinner("Analyzing resume match..."):
                # Calculate similarity score
                similarity_score = matcher.calculate_similarity_score(resume_text, job_text)
                
                # Extract skills
                resume_skills = matcher.extract_skills_and_keywords(resume_text)
                job_skills = matcher.extract_skills_and_keywords(job_text)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Match Results")
                
                # Similarity score
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="metric-card"><div class="match-score">{similarity_score:.1f}%</div><div>Overall Match</div></div>', unsafe_allow_html=True)
                
                # Determine match quality
                if similarity_score >= 75:
                    match_quality = "Excellent"
                    color = "green"
                elif similarity_score >= 60:
                    match_quality = "Good"
                    color = "orange"
                else:
                    match_quality = "Fair"
                    color = "red"
                
                with col2:
                    st.markdown(f'<div class="metric-card"><div style="color: {color}; font-weight: bold; font-size: 1.5rem;">{match_quality}</div><div>Match Quality</div></div>', unsafe_allow_html=True)
                
                with col3:
                    recommendation = "Strong Fit" if similarity_score >= 70 else "Consider" if similarity_score >= 50 else "Not Recommended"
                    st.markdown(f'<div class="metric-card"><div style="font-weight: bold; font-size: 1.2rem;">{recommendation}</div><div>Recommendation</div></div>', unsafe_allow_html=True)
                
                # Skills analysis
                st.subheader("🛠️ Skills Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Resume Skills:**")
                    for skill in resume_skills.get("skills", [])[:10]:
                        st.write(f"• {skill}")
                
                with col2:
                    st.write("**Job Requirements:**")
                    for skill in job_skills.get("skills", [])[:10]:
                        st.write(f"• {skill}")
                
                # Generate improvement suggestions
                if st.button("💡 Get Improvement Suggestions"):
                    with st.spinner("Generating suggestions..."):
                        suggestions = matcher.generate_improvement_suggestions(resume_text, job_text)
                        st.subheader("📝 Improvement Suggestions")
                        st.write(suggestions)

def show_bulk_analysis(matcher):
    st.header("📊 Bulk Resume Analysis")
    
    st.info("Upload multiple resumes to analyze against a job description")
    
    # Job description input
    job_text = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        height=150
    )
    
    # Multiple file upload
    resume_files = st.file_uploader(
        "Upload Multiple Resumes",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        key="bulk_resumes"
    )
    
    if resume_files and job_text and matcher.client:
        if st.button("🔄 Process All Resumes", type="primary"):
            results = []
            progress_bar = st.progress(0)
            
            for i, resume_file in enumerate(resume_files):
                with st.spinner(f"Processing {resume_file.name}..."):
                    # Extract text
                    if resume_file.type == "application/pdf":
                        resume_text = matcher.extract_text_from_pdf(resume_file)
                    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = matcher.extract_text_from_docx(resume_file)
                    else:
                        resume_text = str(resume_file.read(), "utf-8")
                    
                    if resume_text:
                        # Calculate similarity
                        score = matcher.calculate_similarity_score(resume_text, job_text)
                        
                        results.append({
                            "Resume": resume_file.name,
                            "Match Score": score,
                            "Recommendation": "Strong Fit" if score >= 70 else "Consider" if score >= 50 else "Not Recommended",
                            "Text": resume_text[:500] + "..."
                        })
                
                progress_bar.progress((i + 1) / len(resume_files))
            
            # Display results
            if results:
                st.subheader("📈 Analysis Results")
                
                # Sort by score
                results_sorted = sorted(results, key=lambda x: x["Match Score"], reverse=True)
                
                # Create DataFrame
                df = pd.DataFrame(results_sorted)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_score = df["Match Score"].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
                
                with col2:
                    strong_fits = len(df[df["Match Score"] >= 70])
                    st.metric("Strong Fits", strong_fits)
                
                with col3:
                    total_resumes = len(df)
                    st.metric("Total Analyzed", total_resumes)
                
                # Results table
                st.dataframe(
                    df[["Resume", "Match Score", "Recommendation"]],
                    use_container_width=True
                )
                
                # Visualization
                fig = px.bar(
                    df,
                    x="Resume",
                    y="Match Score",
                    color="Recommendation",
                    title="Resume Match Scores"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

def show_cover_letter_generator(matcher):
    st.header("✍️ AI Cover Letter Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Candidate Information")
        candidate_name = st.text_input("Candidate Name", placeholder="John Doe")
        
        resume_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'docx', 'txt'],
            key="cover_letter_resume"
        )
    
    with col2:
        st.subheader("💼 Job Information")
        job_description = st.text_area(
            "Job Description",
            placeholder="Paste job description...",
            height=200
        )
    
    resume_text = ""
    if resume_file:
        if resume_file.type == "application/pdf":
            resume_text = matcher.extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = matcher.extract_text_from_docx(resume_file)
        else:
            resume_text = str(resume_file.read(), "utf-8")
    
    if candidate_name and resume_text and job_description and matcher.client:
        if st.button("🚀 Generate Cover Letter", type="primary"):
            with st.spinner("Generating personalized cover letter..."):
                cover_letter = matcher.generate_cover_letter(
                    resume_text, job_description, candidate_name
                )
                
                st.subheader("📄 Generated Cover Letter")
                st.write(cover_letter)
                
                # Download button
                st.download_button(
                    label="📥 Download Cover Letter",
                    data=cover_letter,
                    file_name=f"cover_letter_{candidate_name.replace(' ', '_')}.txt",
                    mime="text/plain"
                )

def show_resume_insights(matcher):
    st.header("🧠 Resume Insights & Analytics")
    
    resume_file = st.file_uploader(
        "Upload Resume for Analysis",
        type=['pdf', 'docx', 'txt'],
        key="insights_resume"
    )
    
    if resume_file and matcher.client:
        # Extract text
        if resume_file.type == "application/pdf":
            resume_text = matcher.extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = matcher.extract_text_from_docx(resume_file)
        else:
            resume_text = str(resume_file.read(), "utf-8")
        
        if resume_text:
            with st.spinner("Analyzing resume..."):
                # Extract skills and keywords
                skills_data = matcher.extract_skills_and_keywords(resume_text)
                
                # Basic metrics
                word_count = len(resume_text.split())
                char_count = len(resume_text)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", word_count)
                with col2:
                    st.metric("Characters", char_count)
                with col3:
                    st.metric("Skills Found", len(skills_data.get("skills", [])))
                with col4:
                    st.metric("Keywords", len(skills_data.get("keywords", [])))
                
                # Skills breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🛠️ Skills Identified")
                    skills = skills_data.get("skills", [])
                    if skills:
                        for skill in skills[:15]:
                            st.write(f"• {skill}")
                    else:
                        st.write("No skills identified")
                
                with col2:
                    st.subheader("🔑 Key Terms")
                    keywords = skills_data.get("keywords", [])
                    if keywords:
                        for keyword in keywords[:15]:
                            st.write(f"• {keyword}")
                    else:
                        st.write("No keywords identified")
                
                # Text analysis
                st.subheader("📊 Content Analysis")
                
                # Simple readability metrics
                sentences = len([s for s in resume_text.split('.') if s.strip()])
                avg_words_per_sentence = word_count / max(sentences, 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentences", sentences)
                with col2:
                    st.metric("Avg Words/Sentence", f"{avg_words_per_sentence:.1f}")

if __name__ == "__main__":
    main()