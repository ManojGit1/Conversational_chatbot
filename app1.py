# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.messages import HumanMessage, AIMessage
# import streamlit as st

# # Initialize OpenAI components
# OPENAI_API_KEY = "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"  # 🔑 Replace with your key

# def get_vector_store(pdf_path):
#     """Create vector store from PDF"""
#     loader = PyPDFLoader(pdf_path)
#     pages = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2000,
#         chunk_overlap=200,
#         add_start_index=True
#     )
#     chunks = text_splitter.split_documents(pages)
#     return FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

# def get_retriever_chain(vector_store):
#     """Create history-aware retriever chain"""
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
#     retriever = vector_store.as_retriever()
    
#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         ("user", "Given the conversation, generate a search query to find relevant information")
#     ])
    
#     return create_history_aware_retriever(llm, retriever, prompt)


# def get_conversational_rag(retriever_chain):
#     """Create full conversational RAG chain"""
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True)
    
#     # Modified system prompt
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """Answer the question using this context. Include page citations like [Page X] and inline citations for each fact.


#         Context:
#         {context}"""),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}")
#     ])
    
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     return create_retrieval_chain(retriever_chain, document_chain)

# def get_response(user_input):
#     """Get response from RAG chain with sources"""
#     response = st.session_state.conversation_chain.invoke({
#         "chat_history": st.session_state.chat_history,
#         "input": user_input
#     })
    
#     # Extract source pages from documents
#     source_pages = list({doc.metadata['page'] + 1 for doc in response["context"]})  # +1 for 1-based numbering
    
#     # Append sources to answer
#     return f"{response['answer']}\n\n**Sources:** Pages {', '.join(map(str, source_pages))}"

# # Streamlit UI
# st.header("Chat with PDF")

# # Initialize session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="I'm a PDF chatbot. Ask me about the document!")
#     ]

# if "vector_store" not in st.session_state:
#     # Load PDF (Update path to your PDF)
#     pdf_path = '/Users/manojkumarpotnuru/Documents/Big Data Infrastructure/Rioux J. Data Analysis with Python and PySpark 2022.pdf'
#     st.session_state.vector_store = get_vector_store(pdf_path)

# if "conversation_chain" not in st.session_state:
#     retriever_chain = get_retriever_chain(st.session_state.vector_store)
#     st.session_state.conversation_chain = get_conversational_rag(retriever_chain)

# # Display previous messages first
# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message("assistant"):
#             st.markdown(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message("user"):
#             st.markdown(message.content)

# # Handle new input after displaying history
# if user_input := st.chat_input("Ask about the PDF..."):
#     # Add user message to history FIRST
#     st.session_state.chat_history.append(HumanMessage(content=user_input))
    
#     # Immediately display user message through rerun
#     st.experimental_rerun()

# # Handle streaming response after rerun
# if len(st.session_state.chat_history) > 0 and isinstance(st.session_state.chat_history[-1], HumanMessage):
#     last_user_input = st.session_state.chat_history[-1].content
    
#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
#         full_response = ""
#         source_docs = []
        
#         # # Stream the response
#         # for chunk in st.session_state.conversation_chain.stream({
#         #     "chat_history": st.session_state.chat_history[:-1],  # Exclude current input
#         #     "input": last_user_input
#         # }):
#         #     full_response += chunk.get("answer", "")
#         #     response_placeholder.markdown(full_response + "▌")
        
#         # # Final update without cursor
#         # response_placeholder.markdown(full_response)
        

#         # Stream the response and collect sources
#         for chunk in st.session_state.conversation_chain.stream({
#             "chat_history": st.session_state.chat_history[:-1],
#             "input": last_user_input
#         }):
#             if "answer" in chunk:
#                 full_response += chunk["answer"]
#                 response_placeholder.markdown(full_response + "▌")
#             if "context" in chunk:
#                 source_docs = chunk["context"]  # Capture source documents
        
#         # Display final answer
#         response_placeholder.markdown(full_response)
        
#         # Display source documents directly below answer
#         if source_docs:
#             st.markdown("**Source Documents Used:**")
#             for doc in source_docs:
#                 page_num = doc.metadata.get('page', 0) + 1  # Convert to human-readable page number
#                 st.markdown(f"""
#                 📄 **Page {page_num}**  
#                 {doc.page_content[:250]}...  
#                 """)
        
#         # Add final response to history
#         st.session_state.chat_history.append(AIMessage(content=full_response))
        
#     # Rerun to persist in message history
#     st.experimental_rerun()




from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
import streamlit as st
import re

# --- Pydantic Models ---
class Citation(BaseModel):
    page: int
    text: str

class VerifiedResponse(BaseModel):
    answer: str
    citations: list[Citation]

# --- Core Components ---
OPENAI_API_KEY = "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"  # Replace with your key

def get_vector_store(pdf_path):
    """Create vector store with page metadata"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(pages)
    return FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

def get_retriever_chain(vector_store):
    """Create citation-aware retriever"""
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate search terms considering needed citations")
    ])
    
    return create_history_aware_retriever(llm, retriever, prompt)

# Add this function
def extract_verified_sources(response_text, source_docs):
    """Extract and validate cited pages with text snippets"""
    cited_pages = set(map(int, re.findall(r'\[Page (\d+)\]', response_text)))
    sources = []
    
    for doc in source_docs:
        page_num = doc.metadata.get('page', 0) + 1
        if page_num in cited_pages:
            sources.append({
                "page": page_num,
                "text": doc.page_content[:300] + "..."
            })
    
    return sources

# --- Streamlit UI ---
st.set_page_config(page_title="DocChat", layout="wide")
st.header("Document Chat with Verified Citations")

# Session State Management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store("/Users/manojkumarpotnuru/Documents/Big Data Infrastructure/Rioux J. Data Analysis with Python and PySpark 2022.pdf")
if "conversation_chain" not in st.session_state:
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    st.session_state.conversation_chain = create_retrieval_chain(
        retriever_chain,
        create_stuff_documents_chain(
            ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True),
            ChatPromptTemplate.from_messages([
                ("system", """Answer with inline [Page X] citations. Rules:
                1. Cite AFTER each fact
                2. Different facts get different citations
                3. Use only pages from context
                4. Each sentence should end with it's citation
                
                Context: {context}"""),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")
            ])
        )
    )

# Chat Display
for msg in st.session_state.chat_history:
    with st.chat_message("assistant" if isinstance(msg, AIMessage) else "user"):
        st.markdown(msg.content)

# Input Handling
if prompt := st.chat_input("Ask about the document..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.rerun()

# Modify the response handling section
if len(st.session_state.chat_history) > 0 and isinstance(st.session_state.chat_history[-1], HumanMessage):
    last_user_input = st.session_state.chat_history[-1].content
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        source_docs = []
        
        # Stream response
        for chunk in st.session_state.conversation_chain.stream({
            "chat_history": st.session_state.chat_history[:-1],
            "input": last_user_input
        }):
            if "answer" in chunk:
                full_response += chunk["answer"]
                response_placeholder.markdown(full_response + "▌")
            if "context" in chunk:
                source_docs = chunk["context"]
        
        # Display final answer
        response_placeholder.markdown(full_response)
        print("Generated response:", full_response) 
        # Add source verification section
        verified_sources = extract_verified_sources(full_response, source_docs)
        if verified_sources:
            with st.expander("📄 Source Documents (Click to Verify)"):
                for source in verified_sources:
                    st.markdown(f"""
                    **Page {source['page']}**
                    ```text
                    {source['text']}
                    ```
                    """)
        
        st.session_state.chat_history.append(AIMessage(content=full_response))
        st.experimental_rerun()