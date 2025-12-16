import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.embeddings import get_embedding_model
from src.chunker import chunk_documents
from src.vectorstore import create_vectorstore
from src.retriever import retrieve_context
from src.generator import get_llm, generate_answer

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Healthcare Knowledge Chatbot")
st.caption("Upload healthcare documents (PDF or TXT) and ask questions")

# ---------------- Session state ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar: Upload ----------------
with st.sidebar:
    st.header("📄 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Build Knowledge Base"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Processing documents..."):
                documents = []

                for uploaded_file in uploaded_files:
                    suffix = uploaded_file.name.split(".")[-1].lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=f".{suffix}"
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name

                    try:
                        if suffix == "pdf":
                            loader = PyPDFLoader(temp_path)
                        else:
                            loader = TextLoader(temp_path, encoding="utf-8")

                        docs = loader.load()
                        if docs:
                            documents.extend(docs)

                    except Exception as e:
                        st.warning(
                            f"Could not read {uploaded_file.name}: {e}"
                        )

                    finally:
                        os.remove(temp_path)

                if not documents:
                    st.error("No readable text found in uploaded documents.")
                else:
                    embedding_model = get_embedding_model()
                    chunks = chunk_documents(documents)
                    st.session_state.vectorstore = create_vectorstore(
                        chunks, embedding_model
                    )

                    # Reset chat when KB changes
                    st.session_state.messages = []

                    st.success("Knowledge base created successfully!")

# ---------------- Chat history ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat input ----------------
user_input = st.chat_input(
    "Ask a question based on the uploaded documents"
)

if user_input:
    if st.session_state.vectorstore is None:
        st.error("Upload documents and build the knowledge base first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        context_docs = retrieve_context(
            user_input,
            st.session_state.vectorstore
        )

        llm = get_llm()
        answer = generate_answer(
            user_input,
            context_docs,
            llm
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
