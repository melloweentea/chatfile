import streamlit as st 
from pathlib import Path
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os 
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = st.secrets["API_KEY_JO"]

st.set_page_config(page_title="ChatFile", page_icon="🤖")
st.title("ChatFile")

file = st.file_uploader("Upload your file here", type=["pdf", "txt", "md"])

def preprocess_file(file, extension):
    if extension == ".pdf":
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
    else:
        text = file.read().decode()
        # st.write(text)
    return text 

def chunker(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_text(text)
    return chunks 

def generate_response(query, context):
    template = """
    You are a bot answering question based on the uploaded file. Do not attempt to hallucinate a response.
    If the user question is in Thai, reply in Thai. If the user question in English, reply in English.
    Answer the user question in detail using the following context:
    
    Context: {context}
    
    User question: {user_question}
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    return llm.stream(template.format(context=context, user_question=query))

def stream_response(response):
    for chunk in response:
        yield chunk.content
        
if file is not None:
    extension = Path(file.name).suffix
    with st.spinner("Loading..."):
        text = preprocess_file(file, extension)
        chunks = chunker(text)
        vectorstore = FAISS.from_texts(chunks, embedding=OpenAIEmbeddings())
        
        # store_name = file.name[:3]
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         vectorstore = pickle.load(f)
        # else:
        #     vectorstore = FAISS.from_texts(chunks, embedding=OpenAIEmbeddings())
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(vectorstore, f)
            
    query = st.text_input("Ask questions about your file")
    if query:
        docs = vectorstore.similarity_search(query=query, k=3)
        context = "\n---\n".join([doc.page_content for doc in docs])
        st.write_stream(stream_response(generate_response(query, context)))

        