from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Read PDF data
def read_pdf_data(pdf_docs):
    text = "".join([page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages])
    return text

# Split data into chunks
def split_data(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create vector store from PDF data
def create_vectorstore(openai_api_key, pdf_docs):
    text_chunks = split_data(read_pdf_data(pdf_docs))
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
