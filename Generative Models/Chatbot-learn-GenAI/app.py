# Install necessary libraries
# !pip install transformers
# !pip install torch
# !pip install streamlit

import streamlit as st
from transformers import pipeline

# Initialize the text generation pipeline with DialoGPT
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Define the knowledge base for Generative AI concepts
knowledge_base = {
    "What is Generative AI?": "Generative AI refers to a category of artificial intelligence techniques that create new data instances that resemble existing data. Examples include Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).",
    "What is a GAN?": "A Generative Adversarial Network (GAN) consists of two neural networks, a generator and a discriminator, that compete against each other to generate new, synthetic instances of data that resemble real data.",
    "What is a VAE?": "A Variational Autoencoder (VAE) is a type of generative model that learns to encode input data into a lower-dimensional latent space and then decodes it to generate new data samples.",
    "Applications of Generative AI": "Generative AI has various applications, including image generation, text generation, style transfer, data augmentation, and more.",
}

# Define a set of quiz questions and answers
quizzes = {
    "What is the purpose of the discriminator in a GAN?": "To distinguish between real and generated data.",
    "In a VAE, what does the encoder do?": "It compresses input data into a latent space representation."
}

# Function to generate responses based on the knowledge base
def chatbot_response(user_input):
    if user_input.startswith("Quiz:"):
        return handle_quiz(user_input[5:].strip())
    
    # Check if the user input matches any predefined questions
    response = knowledge_base.get(user_input, None)
    if response is not None:
        return response
    else:
        # Generate a response using the text generation pipeline if no match in knowledge base
        responses = chatbot(user_input, max_length=100, num_return_sequences=1)
        return responses[0]['generated_text']

# Function to handle quiz questions
def handle_quiz(question):
    answer = quizzes.get(question, None)
    if answer is not None:
        return f"Quiz: {question} (Answer: {answer})"
    else:
        return "No quiz question found for that input. Try another question or ask about Generative AI concepts."

# Streamlit code for the interactive learning chatbot
def main():
    st.title("Generative AI Learning Chatbot")
    
    st.write("Hello! I am here to help you learn about Generative AI. Ask me questions or type 'quiz' to take a quiz.")
    
    # Input from user
    user_input = st.text_input("You:", placeholder="Ask me about Generative AI or type 'quiz' for a quiz question")
    
    if user_input:
        if user_input.lower() == 'quiz':
            st.write("Here’s a quiz question for you. Type 'Quiz: [question]' to see the answer.")
        else:
            # Get response from chatbot
            response = chatbot_response(user_input)
            st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
