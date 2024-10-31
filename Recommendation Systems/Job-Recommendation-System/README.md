# Job Recommendation System using Machine Learning
 The system is designed to provide personalized job recommendations based on user preferences and historical job data. The data for this project is scraped from Glassdoor, and the system is deployed using the Azure cloud platform.

## Business Understanding
The goal of this project is to develop a job recommendation system that helps users find relevant job opportunities based on their preferences and historical data. By leveraging machine learning techniques, we aim to provide personalized recommendations that align with the user's skills, interests, and career goals. The system will take into account various factors such as job title, salary estimate, company rating, location, industry, and more to generate accurate recommendations.

## Data Scraping
To collect the necessary data for training our recommendation system, we will scrape job-related information from Glassdoor. The following columns will be extracted:

Job Title
Salary Estimate
Job Description
Rating
Company Name
Location
Headquarters
Size
Founded
Type of Ownership
Industry
Sector
Revenue
Competitors

## Feature Engineering
Once the data is collected, we will perform feature engineering to preprocess and transform the raw data into a suitable format for training our recommendation model. This step includes:

Handling Missing Data: Deal with missing values in the dataset by either imputing them or removing the corresponding rows/columns.
Encoding Categorical Variables: Convert categorical variables such as job title, location, industry, and sector into numerical representations using techniques like one-hot encoding or label encoding.
Feature Scaling: Normalize numerical features, such as salary estimate and company rating, to ensure they have a similar scale and prevent dominance of certain features in the model.

## Machine Learning Techniques:
To provide personalized job recommendations, we employ the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique. The "job_recommender.py" component plays a crucial role in this process. It utilizes the TF-IDF vectorizer from the scikit-learn library to transform job descriptions and user preferences into numerical feature vectors. These vectors capture the importance of each word in the documents, enabling the system to find similar job opportunities based on user preferences. The Nearest Neighbors algorithm is then used to identify the most relevant job recommendations.

skill extractor segment provides functions and utilities to extract skills from a PDF file using the Spacy library and perform text processing and matching operations. These extracted skills can be used for further analysis and processing in the job recommendation system.

## Streamlit Application
To make the job recommendation system easily accessible and user-friendly, we have developed a Streamlit application. Streamlit provides an intuitive web interface where users can upload their resumes. The application processes the user input, applies the machine learning models, and displays the top-recommended jobs based on the user's preferences and historical data.

## Model Deployment using Azure Cloud
To make the job recommendation system accessible to users, we will deploy the model on the Azure cloud platform. The deployment process involves the following steps:

* Model Serialization: Serialize the trained model to a format compatible with the Azure cloud deployment.
* Model Containerization: Package the serialized model along with the necessary dependencies and environment specifications into a container using tools like Docker.
* Azure Container Registry: Create a container registry on Azure to store the model container and related artifacts securely.
* Azure Kubernetes Service (AKS): Deploy the model container as a scalable microservice using AKS, which provides orchestration and management capabilities.
* API Development: Develop an API that allows users to interact with the deployed model and request personalized job recommendations.
* Integration and Testing: Integrate the API with other components of the job recommendation system, and perform thorough testing to ensure its functionality and performance.
* Deployment Monitoring: Monitor the deployed model and API to track usage, and performance metrics, and address any potential issues or errors.

## Usage
To use the job recommendation system, follow the instructions below:

* Install the required dependencies: pip install -r requirements.txt
* Run the command: streamlit run __init__.py ( For Local Server )
* Access the deployed job recommendation API and make requests to receive personalized recommendations.


