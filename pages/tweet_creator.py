import streamlit as st
import os

st.write("Twitter, let's go!")

from langchain_core.prompts import PromptTemplate
# OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# PDF Processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from io import StringIO
# Memory (Database)
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent # Agent

# PDF Uploader
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1500,
  chunk_overlap=200
)

st.info("The PDF should contain text for the Tweet Creator to work.", icon="ℹ️")

uploaded_file = st.file_uploader("Upload your PDF below")

if uploaded_file is not None:
  # Process uploaded file
  temp_file = "./temp.pdf"
  with open(temp_file, "wb") as file:
    file.write(uploaded_file.getvalue())
    file_name = uploaded_file.name

  raw_documents = PyPDFLoader(temp_file).load()
  try:
    os.remove(temp_file)
    print(f"File '{temp_file}' successfully deleted.")
  except FileNotFoundError:
    print(f"Error: File '{temp_file}' not found.")
  except OSError as e:
    print(f"Error deleting file: {e}")

  documents = text_splitter.split_documents(raw_documents)
  # Process text input
  db = FAISS.from_documents(documents, OpenAIEmbeddings())
  memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
  )
  llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=1, # balance between creative and randomness
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["OPENAI_API_KEY"]
  )
  llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=1, # balance between creative and randomness
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["OPENAI_API_KEY"]
  )

  template = (
      """You are a Twitter Expert.
      1. If you do not know the answer, say you don't know.
      2. You are polite.
      3. You cannot be racist, offensive, derogatory.
      4. Your responses are limited to 280 characters (including spaces).
      5. If there is a chat history, Combine the chat history and follow up question into a stand alone question. See (6).
      6. Chat History: {chat_history}, Follow up tweet prompt: {question}
    """
  )
  prompt = PromptTemplate.from_template(template)

  qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory, verbose=True,condense_question_prompt=prompt)

  user_input = st.text_input("What would you like to tweet about?")
  generate_tweet = st.button("Generate Tweet")
  if generate_tweet is True:
    output = qa({"question": user_input})
    st.write(f"❄️: {user_input}")
    st.write(f"🐦: {output['answer']}")
