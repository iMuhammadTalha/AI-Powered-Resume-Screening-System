# AI-Powered-Resume-Screening-System

## Overview
This project is an AI-driven resume screening system that extracts key details from resumes, compares them with job descriptions, and computes a match score using NLP techniques. The system leverages Groq's Mixtral-8x7B-32768 model, Sentence Transformers, and Gradio for an interactive user interface.

## Implementation Process
1. Extracting Text from Resume (PDF)

    The system uses pdfminer.six to extract text from uploaded PDF resumes.

    - The extract_text_from_pdf() function reads the document and returns plain text.

2. Extracting Structured Information from Resume

    The extracted text is processed using Groq's LLM (Mixtral-8x7B-32768) to extract:

    - Name

    - Skills

    - Experience

    - Education

    The extract_resume_data() function sends the extracted text to the LLM and retrieves structured information.

3. Matching Resume with Job Description

    - The system encodes both the resume details and job description into vector embeddings using SentenceTransformer('all-MiniLM-L6-v2').

    - Cosine similarity is computed between the two embeddings using pytorch_cos_sim().

    - The match_resume_with_job() function returns a match score indicating candidate suitability.

4. Building a User Interface with Gradio

    The Gradio library is used to create a simple UI where users can:

    - Upload a resume (PDF format)

    - Input a job description

    - Get extracted resume information and a match score

    The process_resume() function integrates all steps and returns results to the UI.

## Technologies Used

- Groq API: LLM for resume data extraction

- pdfminer.six: Extract text from PDFs

- Sentence Transformers: Convert text into vector embeddings

- Torch: Compute similarity scores

- Gradio: Build an interactive web interface

## Installation & Setup

1. Clone the Repository
```bash
git clone https://github.com/your-repo/AI-Resume-Screening.git
cd AI-Resume-Screening
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Set Up Environment Variables

```bash
GROQ_API_KEY="your_api_key_here" 
```

4. Run the Application
```bash
python app.py
```
## Data Sources

- Resumes: Provided by users in PDF format

- Job Descriptions: Entered manually into the UI

## Key Challenges & Solutions

1. Extracting Accurate Resume Data

    - Challenge: Extracting structured information from raw text.

    - Solution: Used Groq LLM to parse text and extract fields accurately.

2. Ensuring Resume & Job Matching is Effective

    - Challenge: Matching resumes with job descriptions using relevant skills.

    - Solution: Used Sentence Transformer embeddings for similarity computation.

3. Handling Different Resume Formats

    - Challenge: PDF resumes have varied layouts.

    - Solution: Used pdfminer.six, which extracts text from diverse formats.

## Future Improvements

- Enhance extraction accuracy using fine-tuned LLM models.

- Implement database storage for storing analyzed resumes.

- Add multi-resume comparison to rank multiple candidates.