# A database of user feedback on model recommendations
user_feedback = {
    'CNN': 4.5,
    'ResNet': 4.0,
    'Logistic Regression': 3.5,
    'Random Forest': 4.2
}

def get_user_feedback(model):
    # Fetching user feedback from a database
    return user_feedback.get(model, 'No feedback available')

# Example usage:
for model in models:
    feedback = get_user_feedback(model)
    print(f"User feedback for {model}: {feedback}")
