import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
stopw  = set(stopwords.words('english'))

# Load the dataset:
unstructured_df=pd.read_csv('jd_unstructured_data.csv')

def convert_salary(value):
    if 'Unknown' in value:
        return None
    elif '-' in value:
        values = re.findall(r'\$\d+K', value)
        min_value = int(values[0].replace('$', '').replace('K', '')) if values else None
        max_value = int(values[1].replace('$', '').replace('K', '')) if len(values) > 1 else None
        if min_value and max_value:
            return (min_value + max_value) / 2
        elif min_value:
            return min_value
        elif max_value:
            return max_value
        else:
            return None
    else:
        return int(re.findall(r'\$\d+K', value)[0].replace('$', '').replace('K', ''))
    
def convert_revenue(value):
    if 'Unknown' in value:
        return None
    elif ' to ' in value:
        values = re.findall(r'\d+\.?\d*', value)
        min_revenue = float(values[0])
        max_revenue = float(values[1])
        unit = value.split()[-2]
        if unit == 'billion':
            min_revenue *= 1000
            max_revenue *= 1000
        return (min_revenue + max_revenue) / 2
    else:
        numerical_values = re.findall(r'\d+\.?\d*', value)
        if numerical_values:
            return float(numerical_values[0])
        else:
            return None

# Define a function to convert the size value
def convert_size(value):
    if 'Unknown' in value:
        return None
    elif ' to ' in value:
        sizes = value.split(' to ')
        min_size = int(sizes[0].replace('+', '').replace(',', '').split()[0])
        max_size = int(sizes[1].replace('+', '').replace(',', '').split()[0])
        return (min_size + max_size) / 2
    else:
        return int(value.replace('+', '').replace(',', '').split()[0])

# Apply the conversion function to the "Salary Column" column
unstructured_df['Average Salary'] = unstructured_df['Salary Estimate'].apply(convert_salary)

# Apply the conversion function to the "Revenue" column
unstructured_df['Average Revenue'] = unstructured_df['Revenue'].apply(convert_revenue)

# Extract the company name by splitting on '\r\n' and selecting the first element
unstructured_df['Company Name'] = unstructured_df['Company Name'].str.split('\r\n').str[0]


# Apply the conversion function to the "Size" column
unstructured_df['Size'] = unstructured_df['Size'].apply(convert_size)

# remove stopwords and pre-process Job Description Column:
unstructured_df['Processed_JD']=unstructured_df['Job Description'].apply(lambda x: ' '.join([word for word in str(x).split() if len(word)>2 and word not in (stopw)]))


# Drop Unwanted Columns:
unstructured_df=unstructured_df.drop(['Unnamed: 0','Salary Estimate','Revenue','Job Description'],axis=1)

# Check for Null Value after data pre-processing:
unstructured_df.isnull().sum()

# Replace the null values with average value of each columns:
# Calculate the average value of column B
size_average = unstructured_df['Size'].mean()
salary_average=unstructured_df['Average Salary'].mean()
revenue_average=unstructured_df['Average Revenue'].mean()

# Replace null values with the average
unstructured_df['Size'].fillna(size_average, inplace=True)
unstructured_df['Average Salary'].fillna(salary_average, inplace=True)
unstructured_df['Average Revenue'].fillna(revenue_average, inplace=True)

# Convert DataFrame to CSV file
unstructured_df.to_csv(r'C:\Users\Admin\ML_Projects\Job_Recommendation_System\Job-Recommendation-System\src\data\jd_structured_data.csv', index=False)