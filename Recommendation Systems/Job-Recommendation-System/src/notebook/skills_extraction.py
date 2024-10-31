import spacy
from spacy.matcher import Matcher
import PyPDF2
import os

# Load the Spacy English model
nlp = spacy.load('en_core_web_sm')
import csv
from spacy.matcher import Matcher
import csv

# Read skills from CSV file
file_path=r'C:\Users\Admin\ML_Projects\Job_Recommendation_System\Job-Recommendation-System\src\data\skills.csv'
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    skills = [row for row in csv_reader]

# Create pattern dictionaries from skills
skill_patterns = [[{'LOWER': skill}] for skill in skills[0]]

# Create a Matcher object
matcher = Matcher(nlp.vocab)

# Add skill patterns to the matcher
for pattern in skill_patterns:
    matcher.add('Skills', [pattern])

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text
        skills.add(skill)
    return skills

# Function to extract text from PDF
def extract_text_from_pdf(file_path:str):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def skills_extractor(file_path):
        # Extract text from PDF
        path=r'C:\Users\Admin\ML_Projects\Job_Recommendation_System\Job-Recommendation-System\src\notebook'
        full_file_path = os.path.join(path, file_path)
        resume_text = extract_text_from_pdf(full_file_path)

        # Extract skills from resume text
        skills = list(extract_skills(resume_text))
        return skills


