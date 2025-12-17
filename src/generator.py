import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    llm_backend = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    )

    return ChatHuggingFace(llm=llm_backend)


def generate_answer(query, context_docs, llm):
    context_text = "\n\n".join(
        doc.page_content for doc in context_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
               """
               You are a helpful AI assistant named NHS Care Bot, designed to provide information and support for patients and visitors of UK NHS Hospitals. Your primary function is to answer questions and provide guidance based on the official NHS information and policies.

Instructions:
1. Respond to user queries using only the information provided in the context above.
2. If a question is outside the scope of the given context, politely inform the user that you cannot provide an answer.
3. Use a friendly and professional tone in your responses.
4. Prioritize patient safety and direct users to seek professional medical advice for specific health concerns.

Example user query: "What are the visiting hours for the General Ward?"

Your response should follow this structure:
1. Greeting
2. Answer (based on the context provided)
3. Additional relevant information (if applicable)
4. Disclaimer
5. Closing statement"""
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
