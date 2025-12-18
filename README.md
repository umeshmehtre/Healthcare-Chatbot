# 🩺 Healthcare Document Chatbot (RAG using LLaMA-3)

## What is this project?

This project is a **document-based healthcare chatbot**.

Instead of answering questions from general internet knowledge, the chatbot:
- reads healthcare documents (PDF or TXT)
- understands their content
- answers questions **only from those documents**

If the information is not present in the uploaded documents, the chatbot clearly says that it does not know.

This approach is called **Retrieval-Augmented Generation (RAG)**.

---

## Why healthcare?

Healthcare is one of the most information-heavy and error-sensitive domains.

Patients, doctors, and hospital staff deal with:
- long guidelines
- policy documents
- discharge summaries
- treatment protocols
- patient history records

Most of this information already exists — the problem is **finding the right part quickly**.

This project focuses on solving that problem safely.

---

## How is this useful for patients?

Patients often:
- do not understand long medical documents
- struggle to find relevant information in discharge summaries
- forget instructions after hospital visits

With this chatbot:
- hospitals can upload official healthcare documents
- patients can ask simple questions in natural language
- answers are based only on approved documents

This reduces confusion and improves understanding without replacing doctors.

---

## How is this useful for doctors and hospital staff?

Doctors and staff spend a lot of time:
- searching through patient records
- reading long medical histories
- checking previous treatments or test results

In the future, this system can be extended so that:
- patient history documents are uploaded securely
- doctors can ask questions like:
  - “What medications was this patient prescribed last year?”
  - “Does the patient have a history of diabetes?”
- the system retrieves answers directly from the patient’s records

This **does not replace clinical judgment** — it saves time and reduces manual searching.

---

## How does the system work? (Simple explanation)

1. Documents are uploaded (PDF or TXT)
2. The text is split into small chunks
3. Each chunk is converted into numerical embeddings
4. Embeddings are stored in a vector database (FAISS)
5. When a question is asked:
   - the most relevant chunks are retrieved
   - those chunks are sent to the language model
6. The model generates an answer strictly from that context

```
## Architechture
![Architechture](assets\image.png)

```

```
healthcare-chatbot/
│
├── app.py
│   Main Streamlit application
│
├── requirements.txt
│   Python dependencies
│
├── src/
│   ├── embeddings.py
│   ├── chunker.py
│   ├── vectorstore.py
│   ├── retriever.py
│   └── generator.py
|
├── assests
│   └── image.png 
```
---

## How to run the project locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Set your Hugging Face API token:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN=hf_your_token_here

3. Start the application:
   ```bash   
   streamlit run app.py

## Deployment

The application is deployed using **Streamlit Cloud.**

Steps:

1. Push the repository to GitHub
2. Create a Streamlit Cloud app
3. Select app.py as the entry point
4. Add HUGGINGFACEHUB_API_TOKEN as a secret
5. Deploy

## Future work and extensions

This project is intentionally designed to be extendable.

Possible future improvements:

- Secure patient-specific document storage
- Role-based access (doctor vs patient)
- Search across patient history records
- Integration with hospital information systems
- Audit logs for medical compliance
- Support for structured medical data

These features would require strong security, privacy, and regulatory controls.

## Important disclaimer

This chatbot is **not a medical professional.**

It provides information strictly based on uploaded documents and should **never replace medical advice.**
All medical decisions must be made by qualified healthcare professionals.