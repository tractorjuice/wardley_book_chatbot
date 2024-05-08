import os, re, uuid
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Pinecone

# gpt-3.5-turbo, gpt-4, and gpt-4-turbo-preview point to the latest model version
#MODEL = "gpt-3.5-turbo" # 4K, Sept 2021. Legacy. Currently points to gpt-3.5-turbo-0613.
#MODEL = "gpt-3.5-turbo-16k" # 16K, Sept 2021. Legacy. Snapshot of gpt-3.5-turbo from June 13th 2023. Will be deprecated on June 13, 2024
MODEL = "gpt-3.5-turbo-1106" # 16K, Sept 2021. New Updated GPT 3.5 Turbo. The latest GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens.
#MODEL = "gpt-4" # 8K, Sept 2021
#MODEL = "gpt-4-32k" # 32K, Sept 2021
#MODEL = "gpt-4-turbo-preview" # 128K, Apr 2023
#MODEL = "gpt-4-1106-preview" # 128K, Apr 2023

DEBUG = True # True to overwrite files that already exist

# Set API keys
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Chat with Simon Wardley's Book")
st.title("Chat with Simon Wardley's Book")
st.sidebar.markdown("# Query Simon's book using AI")
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 1.3.0")
st.sidebar.markdown(st.session_state.session_id)
st.sidebar.markdown("Wardley Mapping is provided courtesy of Simon Wardley and licensed Creative Commons Attribution Share-Alike.")
st.sidebar.divider()

# Set styling for buttons. Full column width, primary colour border.
primaryColor = st.get_option("theme.primaryColor")
custom_css_styling = f"""
<style>
    /* Style for buttons */
    div.stButton > button:first-child, div.stDownloadButton > button:first-child {{
        border: 5px solid {primaryColor};
        border-radius: 20px;
        width: 100%;
    }}
    /* Center align button container */
    div.stButton, div.stDownloadButton {{
        text-align: center;
    }}
    .stButton, .stDownloadButton {{
        width: 100%;
        padding: 0;
    }}
</style>
"""
st.html(custom_css_styling)

tags = [
    "streamlit",
    "WardleyBookChatbot",
    st.session_state.session_id,
]

# Check if the user has provided an API key, otherwise default to the secret
user_openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", placeholder="sk-...", type="password")

if user_openai_api_key:
    os.environ["OPENAI_API_KEY"] = user_openai_api_key

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embedding=OpenAIEmbeddings())

        custom_system_template="""
            You are SimonGPT with the style of a strategy researcher with well over twenty years research in strategy and cloud computing.
            You use complicated examples from Wardley Mapping in your answers.
            Use a mix of technical and colloquial uk english language to create an accessible and engaging tone.
            Your language should be for an 12 year old to understand.
            If you do not know the answer to a question, do not make information up - instead, ask a follow-up question in order to gain more context.
            Your primary objective is to help the user formulate excellent answers by utilizing the context about the book and
            relevant details from your knowledge, along with insights from previous conversations.
            ----------------
            Reference Context and Knowledge from Similar Existing Services: {context}
            Previous Conversations: {chat_history}"""

        custom_user_template = "Question:'''{question}'''"

        prompt_messages = [
            SystemMessagePromptTemplate.from_template(custom_system_template),
            HumanMessagePromptTemplate.from_template(custom_user_template)
            ]
        prompt = ChatPromptTemplate.from_messages(prompt_messages)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model_name=MODEL,
            temperature=0,
            max_tokens=300,
        )  # Modify model_name if you have access to GPT-4

    if "chain" not in st.session_state:
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=st.session_state.vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    #"score_threshold": .95,
                    }
                ),
            chain_type="stuff",
            rephrase_question = True,
            return_source_documents=True,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )

    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if query := st.chat_input("What question do you have for the book?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner():
            with st.chat_message("assistant"):
                response = st.session_state.chain(query)
                st.markdown(response['answer'])

                with st.expander("Source"):
                        source_documents = response['source_documents']
                        for index, document in enumerate(source_documents):
                            # Safely retrieve metadata using `get` to avoid KeyError
                            chapter_details = document.metadata.get('Chapter', 'Not provided')
                            section_details = document.metadata.get('Section', 'Not provided')
                            st.markdown(f":blue[Source {index + 1}]")
                            st.markdown(f"{chapter_details}")
                            st.markdown(f"Section: {section_details}")

                st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
else:
    st.warning("Please enter your OpenAI API key", icon="⚠️")