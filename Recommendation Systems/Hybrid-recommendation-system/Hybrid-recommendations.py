def hybrid_recommendation_system(data, task_type):
    # Step 1: Content-based recommendations based on dataset type
    content_based_recommendations = recommend_based_on_data_type(data, task_type)
    
    # Step 2: Collaborative filtering - rank based on user feedback
    ranked_recommendations = sorted(content_based_recommendations, 
                                    key=lambda model: user_feedback.get(model, 0), 
                                    reverse=True)
    
    return ranked_recommendations

# Example usage:
# final_recommendations = hybrid_recommendation_system('image', 'classification')
# print(f"Hybrid recommendation for image classification: {final_recommendations}")