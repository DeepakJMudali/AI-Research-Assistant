import uuid

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.utilities import (
    ArxivAPIWrapper,
    DuckDuckGoSearchAPIWrapper
)
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchRun
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler
)
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

/* Main content */
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.main-header {
    font-size: 34px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 5px;
}

.sub-header {
    font-size: 15px;
    color: #A0A7B4;
    margin-bottom: 30px;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #202B43, #172033);
    border: 1px solid #344563;
    border-radius: 16px;
    padding: 28px;
    margin: 20px 0;
}

.welcome-title {
    color: #FFFFFF;
    font-size: 25px;
    font-weight: 700;
}

.welcome-description {
    color: #B8C4D9;
    font-size: 15px;
    line-height: 1.7;
}

/* Tool cards */
.tool-card {
    background-color: #171D29;
    border: 1px solid #30394B;
    border-radius: 12px;
    padding: 15px;
    min-height: 105px;
}

.tool-title {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 600;
}

.tool-description {
    color: #A0A7B4;
    font-size: 12px;
    margin-top: 6px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111722;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF;
}

/* Chat input */
.stChatInput {
    border-radius: 12px;
}

/* Buttons */
.stButton button {
    border-radius: 9px;
    border: 1px solid #344563;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

system_prompt = """
You are a helpful AI research assistant.

You have access to four tools:


2. Arxiv:
   Search academic research papers.

3. DuckDuckGo:
   Search the web for current information.

4. AI-wiki-search:
   Retrieve information from the custom AI knowledge base.

Instructions:
- Select the appropriate tool based on the user's query.
- Use Arxiv for academic research.
- Use DuckDuckGo for current web information.
- Use AI-wiki-search for information in the indexed knowledge base.
- You may use multiple tools when necessary.
- Do not invent retrieved information.
- Provide clear and accurate answers.
"""


# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

@st.cache_resource
def get_vectorStore():

    loader = WebBaseLoader(
        "https://en.wikipedia.org/wiki/Artificial_intelligence"
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks = splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorStore = FAISS.from_documents(
        chunks,
        embedder
    )

    return vectorStore


# --------------------------------------------------
# TOOLS
# --------------------------------------------------



api_arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=250
)

arxiv = ArxivQueryRun(
    api_wrapper=api_arxiv_wrapper
)


search_wrapper = DuckDuckGoSearchAPIWrapper(
    max_results=2
)

search = DuckDuckGoSearchRun(
    name="search",
    api_wrapper=search_wrapper
)


retriever = get_vectorStore().as_retriever(
    search_kwargs={"k": 2}
)

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="AI-wiki-search",
    description=(
        "Search the custom FAISS knowledge base "
        "containing information about Artificial Intelligence."
    )
)

tools = [search, arxiv, retriever_tool]


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "memory" not in st.session_state:
    st.session_state["memory"] = InMemorySaver()
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🔎 AI Research Assistant")

    st.caption("Powered by LangChain + LangGraph")

    st.divider()

    st.subheader("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your API key"
    )

    st.caption("Model: GPT-4.1-mini")

    st.divider()

    st.subheader("🧰 Available Tools")

    st.markdown("""
    **🌐 DuckDuckGo**

    Search current information on the web.

    **📄 Arxiv**

    Search academic research papers.

    **🧠 Custom RAG**

    Retrieve information from your FAISS knowledge base.
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state["messages"] = []
        st.rerun()

    st.caption("AI Research Assistant | Built with Streamlit")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-header">🔎 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'Research, explore, and get answers using multiple AI-powered tools.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# WELCOME SCREEN
# --------------------------------------------------

st.subheader("💡 Try asking")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

if not api_key:

    st.info("🔑 Enter your OpenAI API key in the sidebar to start chatting.")

prompt = st.chat_input(
    "Ask anything."
    ,
    disabled=not bool(api_key)
)


# --------------------------------------------------
# AGENT EXECUTION
# --------------------------------------------------

if prompt:

  

    with st.chat_message("user"):
        st.markdown(prompt)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=api_key
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer = st.session_state["memory"],
        system_prompt=system_prompt
    )

    with st.chat_message("assistant"):

        try:
           
            response = agent.invoke(
                {
                    "messages": [HumanMessage(content=prompt)]
                },
                config = {"configurable":{"thread_id": st.session_state["thread_id"]}}
            )

            answer = response["messages"][-1].content

            st.markdown(answer)

            

        except Exception as e:

            st.error(f"Something went wrong: {str(e)}")

