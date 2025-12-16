from pathlib import Path

def load_documents(folder_path: str) -> list[str]:
    documents = []

    for file in Path(folder_path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            documents.append(f.read())

    return documents
