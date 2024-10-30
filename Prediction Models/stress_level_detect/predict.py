from models.stress_level_detect.model import stress_level_prediction

def get_prediction(age, freq_no_purpose, freq_distracted, restless, worry_level, difficulty_concentrating, compare_to_successful_people, feelings_about_comparisons, freq_seeking_validation, freq_feeling_depressed, interest_fluctuation, sleep_issues):
    
    prediction = stress_level_prediction(age, freq_no_purpose, freq_distracted, restless, worry_level, difficulty_concentrating, compare_to_successful_people, feelings_about_comparisons, freq_seeking_validation, freq_feeling_depressed, interest_fluctuation, sleep_issues)

    advice = ""

    # Provide advice based on the prediction value
    if prediction < 1.5:
        advice = "You are experiencing mild stress. Keep maintaining a balanced lifestyle, and consider engaging in activities that bring you joy and relaxation."
    elif 1.5 <= prediction < 3.5:
        advice = "You have a moderate stress level. It's important to take breaks and practice stress-relief techniques like mindfulness, walking, cycling, music or exercise."
    else:
        advice = "You are experiencing high stress levels. Consider reaching out to a mental health professional or practicing stress management techniques to help cope."

    return advice