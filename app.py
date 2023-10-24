from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import openai
import streamlit as st

load_dotenv()
openai.api_key = st.secrets['OPENAI_API_KEY']

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="knowledge_base.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
    You should function as a AI assitant that will provide support internally to a bank's employees.The name of the bank is "Banco Santa Fe" and it is an Argentine bank.
    
    You will be provided with some FAQ related to Banco Santa Fe. You nees to answer the employee question based on the FAQ i will provide you, DO NOT ANSWEAR A QUESTION IF IT'S NOT ANSWERED ON THE FAQ ALREADY. 
    Don't answer any questions that aren't answered in the FAQ.

    IF THE EMPLOYEE QUESTION IS NOT ANSWERED BY THE FAQ PROVIDED YOU MUST RESPOND WITH 'Lo siento, no tengo esa información.'

    Below is the employee question:
    {message}

    Here is the FAQ:
    {faq}

    Plese write an answer for the question
    """

prompt = PromptTemplate(
    input_variables=["message", "faq"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    faq = retrieve_info(message)
    response = chain.run(message=message, faq=faq)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Chatbot Banco Santa Fe")

    st.header("Chatbot Banco Santa Fe")
    message = st.text_area("Problema del cliente")

    if message:
        st.write("Generando una respuesta...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
