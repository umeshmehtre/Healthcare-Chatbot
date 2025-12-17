from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace
import os


def get_llm():
    return ChatHuggingFace(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    )


def generate_answer(query, context_docs, llm):
    context_text = "\n\n".join(
        doc.page_content for doc in context_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a healthcare assistant. "
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, say you don't know."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "context": context_text,
            "question": query,
        }
    )
