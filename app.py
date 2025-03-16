import gradio as gr
import groq
import os
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv


# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()

# Set up Groq API key (Replace with your actual API key)
groq.api_key = os.getenv("GROQ_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    return extract_text(pdf_path)

def extract_resume_data(text):
    """Uses LLM to extract structured resume information using Groq API."""
    client = groq.Client(api_key=groq.api_key)

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "Extract name, skills, experience, and education from this resume."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def match_resume_with_job(resume_skills, job_description):
    """Computes similarity between resume skills and job description."""
    resume_embedding = model.encode(resume_skills, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding)
    return similarity_score.item()

def process_resume(file, job_desc):
    resume_text = extract_text_from_pdf(file.name)
    extracted_info = extract_resume_data(resume_text)
    match_score = match_resume_with_job(extracted_info, job_desc)
    return extracted_info, f"Match Score: {match_score:.2f}"

iface = gr.Interface(
    fn=process_resume,
    inputs=["file", "text"],
    outputs=["text", "text"],
    title="AI-Powered Resume Screener",
    description="Upload a resume and enter a job description to evaluate candidate fit."
)

if __name__ == "__main__":
    iface.launch(share=True)
