# Generative AI Learning Chatbot

This project is an interactive chatbot designed to help users learn about Generative AI. The chatbot is built using **Streamlit** and **DialoGPT**, and it integrates a knowledge base of Generative AI concepts along with a quiz functionality.

## Features

- **Generative AI Knowledge Base:** The chatbot can answer pre-defined questions about Generative AI concepts such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and their applications.
  
- **Quiz Functionality:** Users can engage with quiz questions related to Generative AI. Users can type `quiz` to receive a question, and then input the question in the format `Quiz: [question]` to see the answer.

- **DialoGPT Integration:** For any question that is not in the predefined knowledge base, the chatbot uses the **DialoGPT** model from Hugging Face’s transformers library to generate a text-based response.

## How to Run the Project

### Prerequisites

1. Python 3.7+
2. Install required libraries using pip:
    ```bash
    pip install streamlit transformers
    ```

### Running the Chatbot

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run chatbot.py
    ```

3. The chatbot will open in your web browser, and you can start interacting with it.

### Sample Interaction

- Ask about Generative AI:
    ```
    You: What is Generative AI?
    Chatbot: Generative AI refers to a category of artificial intelligence techniques that create new data instances that resemble existing data. Examples include Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).
    ```

- Take a quiz:
    ```
    You: quiz
    Chatbot: Here’s a quiz question for you. Type 'Quiz: [question]' to see the answer.
    ```

    ```
    You: Quiz: What is the purpose of the discriminator in a GAN?
    Chatbot: Quiz: What is the purpose of the discriminator in a GAN? (Answer: To distinguish between real and generated data.)
    ```

## Code Structure

- `chatbot.py`: Contains the main code for the chatbot, including the knowledge base, quiz handling logic, and integration with DialoGPT.
  
## Future Enhancements

- **Expand Knowledge Base:** Add more detailed information and concepts related to Generative AI.
- **Improve Quiz Functionality:** Implement more advanced quiz interactions, such as scoring or random question selection.
- **Personalization:** Integrate user-specific learning paths or history-based interactions.
