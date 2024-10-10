# ChatDocs AI
RAG-based generative AI application enabling interactive communication with PDF documents. Developed using Streamlit and Groq AI Inference technology.

# Supported Models
This application makes use of following LLMs:
  - Chat Models — Groq AI:
      - Llama3-8b-8192
      - Llama3-70b-8192
      - Mixtral-8x7b-32768
      - Gemma-7b-it
  - Embeddings -- OpenAI
      - Text-embedding-ada-002-v2
    
# System Requirements
- Python 3.9 or later (earlier versions are not compatible).

# Installation
1. Clone the repository
```bash
git clone <repo-url>
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required Python packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run main.py
```

# Snapshots