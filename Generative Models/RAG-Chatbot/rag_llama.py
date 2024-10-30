import ollama
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


emb = embeddings.embed_query("Hello World")
# print(len(emb))


with open("ML_Nexus.txt", "r", encoding="utf-8") as file:
    document_text = file.read()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=30)
chunks = text_splitter.split_text(document_text)
# print(len(chunks))


from langchain_core.documents import Document

document_obj = []
for i, doc_content in enumerate(chunks, start=-1):
    temp = Document(page_content=doc_content)
    document_obj.append(temp)


from uuid import uuid4

uuids = [str(uuid4()) for _ in range(len(document_obj))]


# print(document_obj)
# print(len(uuids))


final_emb = []
for doc in document_obj:
    embedding = embeddings.embed_query(doc.page_content)
    final_emb.append(embedding)
final_emb = np.array(final_emb)
# print(final_emb.shape)


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Create the Faiss index
# dimension = document_embeddings.shape[1]
# print(dimension)
emb = embeddings.embed_query("Hello World")
index = faiss.IndexFlatL2(len(emb))  # Using L2 distance for simplicity

index.add(final_emb)

index_to_docstore_id = {i: uuids[i] for i in range(len(uuids))}

# print(index.d)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id=index_to_docstore_id,
)

vector_store.add_documents(documents=document_obj, ids=uuids)


def get_context(query):
    # similar_search_result = vector_store.similarity_search(query, k=5)
    # print(similar_search_result)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    similar_search_result = retriever.invoke(query)
    # print(similar_search_result)
    compiled_context = "\n\n".join(doc.page_content for doc in similar_search_result)
    return compiled_context


def llm(question):
    compiled_context = get_context(question)

    formatted_prompt = """
	"Answer the question below with the context.\n\n"
	Context :\n\n{}\n\n----\n\n"
	"Question: {}\n\n"
	"Write an answer based on the context. "
	"If the context provides insufficient information reply "
    '"The information given in the context is insufficient. Thus, answering without context: "'
	"and then answer the question with the existing knowledge you have"
    "If quotes are present and relevant, use them in the answer."
	""".format(
        compiled_context, question
    )
    res = ollama.chat(
        model="llama3.1:latest", messages=[{"role": "user", "content": formatted_prompt}]
    )
    print(res["message"]["content"])
    print("-" * 100)
    # return res["message"]["content"]


queries = [
    "Who are the mentors of the given project?",
    "What is the maximum number of points I can earn from this repository?"
]
for query in queries:
    print(f"Question: {query}\n")
    llm(query)
    print()
