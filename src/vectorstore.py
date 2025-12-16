from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embedding_model):
    if not chunks:
        raise ValueError("No document chunks to embed.")
    return FAISS.from_documents(chunks, embedding_model)
