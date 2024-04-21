import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import streamlit as st
import voyageai
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
from langchain_pinecone import PineconeVectorStore 
from io import BytesIO
load_dotenv()
from voyager_embeddings import VoyageEmbeddings
from langchain.document_loaders import PyPDFLoader


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_KEY = SecretStr(CLAUDE_API_KEY) if CLAUDE_API_KEY is not None else None


def read_pdf(pdf):
    if pdf:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        return text



#making chunks
def make_chunks(text,chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc = text_splitter.split_text(text=text)
    return doc


#voyageai embeddings
def voyage_embeddings(chunks):
    vo=voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    texts = [doc.page_content for doc in chunks]
    result = vo.embed(texts, model="voyage-2", input_type="document")
    return result

#streamlit app
st.title("Document Search Engine")


input_pdf = st.file_uploader("Upload a PDF", type="pdf")
if input_pdf:
    
    chat_history = []

    my_document = read_pdf(input_pdf)
    print("PDF uploaded successfully")

    my_chunks = make_chunks(my_document)
    print("Chunks created successfully")

    vector = voyage_embeddings(my_chunks)
    print("Embeddings created successfully")

    voyage_embeddings_wrapper = VoyageEmbeddings(vector.embeddings)
    print("Embeddings wrapper created successfully")

    vector_store = PineconeVectorStore.from_documents(my_document, embedding=voyage_embeddings_wrapper, index_name=INDEX_NAME)
    print("Vector store created successfully")

    retriver = vector_store.as_retriever(search_type="similarity",search_kwargs={'k': 2})
    print("Retriver created successfully")

    llm = ChatAnthropic(anthropic_api_key=CLAUDE_API_KEY,model_name="claude-3-sonnet-20240229")
    print("LLM created successfully")

    qa = RetrievalQA.from_llm(llm=llm,retriever=retriver)
    print("QA created successfully")

    st.write("PDF uploaded successfully")
    st.write("chat with your data, type 'exit' to quit")
    query = st.text_input("Ask a question")
    while True:
        if query == "exit":
            st.write("Goodbye!")
            vector_store.delete()
            break


        response = qa.invoke({"query": query, "chat_history": chat_history})
        chat_history.append({"query": query, "response": response})
        st.write(response)


