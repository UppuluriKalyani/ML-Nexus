# **Crop Recommendation System 🌾**

This project is a machine learning-powered tool that suggests the most suitable crop based on environmental parameters such as **temperature, humidity, pH, and rainfall**. It aims to help farmers make data-driven decisions, boosting productivity and sustainability.  

## **Features**  
- **Multiple ML Models**: Utilizes Logistic Regression, Naive Bayes, SVM, Random Forest, and more.  
- **Flask-based Web App**: User-friendly interface for real-time crop recommendations.  
- **Scalable Solution**: Adaptable for different regions and climates.  

## **Tech Stack**  
- **Python**, **Scikit-Learn** for machine learning  
- **Flask** for web application  
- **Pickle** for model persistence  
- **Jupyter Notebook** for development  

## **How to Run the Application**  
1. **Clone the repository**:  
   ```bash
   git clone <repo_url> && cd crop-recommendation-system
   ```

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional)**:  
   Use `crop_main.ipynb` to train the model or load the existing one.  
   ```python
   import pickle
   model = pickle.load(open('model.pkl', 'rb'))
   ```

4. **Run the Flask App**:  
   ```bash
   python app.py
   ```


5. **Sample Input Example**:
   ```
   N = 10, P = 10, K = 10  
   Temperature = 15.0°C  
   Humidity = 80%  
   pH = 4.5  
   Rainfall = 10 mm
   ```

## **Troubleshooting**  
- **Pickle Load Error**: Ensure the **same scikit-learn version** used for training is installed.  
- **Alternative with Joblib**:  
   ```python
   import joblib
   model = joblib.load('model.pkl')
   ```

