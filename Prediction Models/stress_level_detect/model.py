from joblib import load

# Load the trained Random Forest model
model = load('models/stress_level_detect/saved_models/random_forest_model.joblib')

def stress_level_prediction(age, freq_no_purpose, freq_distracted, restless, worry_level, difficulty_concentrating, compare_to_successful_people, feelings_about_comparisons, freq_seeking_validation, freq_feeling_depressed, interest_fluctuation, sleep_issues):
    # Feature extraction
    features = [
        float(age),
        int(freq_no_purpose),
        int(freq_distracted),
        int(restless),
        int(worry_level),
        int(difficulty_concentrating),
        int(compare_to_successful_people),
        int(feelings_about_comparisons),
        int(freq_seeking_validation),
        int(freq_feeling_depressed),
        int(interest_fluctuation),
        int(sleep_issues)
    ]

    prediction = model.predict([features])[0]
    
    return prediction

