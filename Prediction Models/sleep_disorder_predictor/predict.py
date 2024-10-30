from sleep_disorder_predictor.model import disease_get_prediction

def get_prediction(Age, Sleep_Duration,  
                    Heart_Rate, Daily_Steps, 
                    Systolic, Diastolic,Occupation,Quality_of_Sleep,Gender, 
                    Physical_Activity_Level, Stress_Level, BMI_Category):
    
    prediction = disease_get_prediction(Age, Sleep_Duration,  
                    Heart_Rate, Daily_Steps, 
                    Systolic, Diastolic,Occupation,Quality_of_Sleep,Gender, 
                    Physical_Activity_Level, Stress_Level, BMI_Category)
    
    message = ""

    # Provide message based on the prediction value
    if prediction==0:
        message= "Insomnia"
    elif prediction==1:
        message = "No disorder"
    elif prediction==2:
        message = "Sleep Apnea"
    else:
        message="Invalid details."

    return message+"\n\nRecommendation - To prevent sleep disorders, maintain a balanced lifestyle with regular exercise, a healthy diet, and stress management. Stick to a consistent sleep schedule, limit caffeine and alcohol, and create a relaxing bedtime routine."