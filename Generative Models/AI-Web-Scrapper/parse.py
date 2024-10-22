import os 
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)


def parse_with_llm(text_chunks , parse_description , open_ai_secreat_key):
    os.environ["OPENAI_API_Key"] = open_ai_secreat_key
    model = OpenAI(model = "gpt-3.5-turbo" , temperature= 0.7)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    parse_results = []
    
    for i, chunks in enumerate(text_chunks , 1):
        responce = chain.invoke(
            {"dom_content" : chunks , "parse_description" : parse_description}
        )
        
        print(f"Parse Batch {i} of {len(text_chunks)}")
        parse_results.append(responce)
    
    return "\n".join(parse_results)

