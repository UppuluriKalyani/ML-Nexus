class Chatbot:
    def __init__(self):
        self.responses = {
            "greeting": "Hello! How can I assist you today?",
            "farewell": "Goodbye! Have a great day!"
        }
        self.feedback_log = []  # To store user feedback

    def get_response(self, user_input):
        if "hello" in user_input.lower():
            return self.responses["greeting"]
        elif "bye" in user_input.lower():
            return self.responses["farewell"]
        else:
            return "I'm sorry, I don't understand that."

    def collect_feedback(self, user_input, response):
        print(f"Bot: {response}")
        feedback = input("Was this response helpful? (thumbs up 👍 / thumbs down 👎): ").strip().lower()
        if feedback in ['thumbs up', '👍']:
            self.feedback_log.append((user_input, response, True))
            print("Thank you for your feedback!")
        elif feedback in ['thumbs down', '👎']:
            self.feedback_log.append((user_input, response, False))
            print("Thank you for your feedback! We'll work on improving.")
        else:
            print("Invalid feedback. Please respond with thumbs up or thumbs down.")

    def chat(self):
        print("Welcome to the chatbot! Type 'exit' to end the chat.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = self.get_response(user_input)
            self.collect_feedback(user_input, response)
        print("Chat ended.")

# Example usage
if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.chat()
