from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Get LLM response to user query
def get_llm_response(llm, prompt_template, question):
    doc_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), doc_chain)
    response = retrieval_chain.invoke({'input': question})
    return response
