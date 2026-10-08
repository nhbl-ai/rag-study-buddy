import requests
from langchain_core.documents import Document

source_urls = [
    "https://docs.langchain.com/oss/python/deepagents/overview",
    "https://docs.langchain.com/oss/python/deepagents/subagents",
    "https://docs.langchain.com/oss/python/deepagents/streaming",
]

documents = []

for source_url in source_urls:
    response = requests.get(f"{source_url}.md", timeout=30)
    response.raise_for_status()

    documents.append(
        Document(
            page_content=response.text,
            metadata={"source": source_url},
        )
    )

print(f"Downloaded {len(documents)} documents.")
