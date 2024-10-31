import streamlit as st
import pandas as pd
import PyPDF2
from pyresparser import ResumeParser
from sklearn.neighbors import NearestNeighbors
from src.components.job_recommender import ngrams,getNearestN,jd_df
import src.notebook.skills_extraction as skills_extraction
from sklearn.feature_extraction.text import TfidfVectorizer


# Function to process the resume and recommend jobs
def process_resume(file_path):
    # Extract text from PDF resume
    resume_skills=skills_extraction.skills_extractor(file_path)

    # Perform job recommendation based on parsed resume data
    skills=[]
    skills.append(' '.join(word for word in resume_skills))
    
    
    # Feature Engineering:
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)

    
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = (jd_df['Processed_JD'].values.astype('U'))

    distances, indices = getNearestN(jd_test)
    test = list(jd_test) 
    matches = []

    for i,j in enumerate(indices):
        dist=round(distances[i][0],2)
        temp = [dist]
        matches.append(temp)
    
    matches = pd.DataFrame(matches, columns=['Match confidence'])

    # Following recommends Top 5 Jobs based on candidate resume:
    jd_df['match']=matches['Match confidence']
    
    return jd_df.head(5).sort_values('match')

# Streamlit app
def main():
    st.title("Job Recommendation App")
    st.write("Upload your resume in PDF format")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf'])

    if uploaded_file is not None:
        # Process resume and recommend jobs
        file_path=uploaded_file.name
        df_jobs = process_resume(file_path)

        # Display recommended jobs as DataFrame
        st.write("Recommended Jobs:")
        st.dataframe(df_jobs[['Job Title','Company Name','Location','Industry','Sector','Average Salary']])

# Run the Streamlit app
if __name__ == '__main__':
    main()
