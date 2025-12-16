def retrieve_context(query, vectorstore, k=4):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)
