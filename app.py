import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
import os

load_dotenv()

# 1. Vectorise the responses csv data
loader = CSVLoader(file_path="dataset.csv")
documents = loader.load()


embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array


# 3. Setup LLMChain & prompts

repo_id = "google/flan-t5-xxl"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

template = """
You will help me provide medical related advice for common diseases 
I will share a user's message with you and you will give me the best answer that 
I should send to this user based on past responses, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past responses

2/ If the responses are irrelevant, then try to mimic the style of the past responses to user's message


Below is a message I received from theuser:
{message}

Here is a list of past responses of how we normally respond to user in similar scenarios:
{past_responses}

Please write the best response that I should send to this user:
"""

prompt = PromptTemplate(
    input_variables=["message", "past_responses"], template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation


def generate_response(message):
    past_responses = retrieve_info(message)
    response = chain.run(message=message, past_responses=past_responses)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon=":books:")

    st.header("Medical Chatbot :book:")
    message = st.text_area("user query")

    if message:
        st.write("Generating best advicee...")

        result = generate_response(message)

        st.info(result)


if __name__ == "__main__":
    main()
