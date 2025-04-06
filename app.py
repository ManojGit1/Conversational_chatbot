from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

OPENAI_API_KEY = "sk-proj-t_EmPuLMXNrZ0MlxzPxuaOe1IM05GRr4CyCbBRRkyD_n1uqUAVds0ahLETT3BlbkFJHvX8mVID2pXkywAs-USPVnfautRNkJOPCNTkv02PnUr4WqvX45shgXWNMAbcd"  # 🔑 Replace with your key

def get_vector_store(pdf_path):
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
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query for relevant document pages")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag(retriever_chain):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, streaming=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer using ONLY the context below. For every factual claim, 
        include an INLINE CITATION with the relevant page number like [Page X]. 
        Never mention you're citing sources. Context:
        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, document_chain)

# Streamlit UI
st.header("PDF Chat with Inline Citations")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Ask about the document - I'll answer with inline citations!")
    ]

if "vector_store" not in st.session_state:
    pdf_path = '/path/to/your/document.pdf'
    st.session_state.vector_store = get_vector_store(pdf_path)

if "conversation_chain" not in st.session_state:
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    st.session_state.conversation_chain = get_conversational_rag(retriever_chain)

# Display history
for msg in st.session_state.chat_history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle input
if user_input := st.chat_input("Ask about the PDF..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.rerun()

# Generate response
if st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], HumanMessage):
    query = st.session_state.chat_history[-1].content
    
    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain.invoke({
            "chat_history": st.session_state.chat_history[:-1],
            "input": query
        })
        
        # Directly use the LLM's response with inline citations
        final_answer = response["answer"]
        st.markdown(final_answer)
        st.session_state.chat_history.append(AIMessage(content=final_answer))
        st.rerun()
