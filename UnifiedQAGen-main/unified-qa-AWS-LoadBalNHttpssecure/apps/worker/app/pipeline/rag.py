import os
import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    return _embeddings


def chunk_documents(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def build_vectorstore(job_id: str) -> Chroma:
    path = os.path.join("/tmp", f"vs_{job_id}")
    os.makedirs(path, exist_ok=True)

    return Chroma(
        collection_name=f"job_{job_id}",
        embedding_function=get_embeddings(),
        persist_directory=path,
    )


def add_to_rag(vs: Chroma, url: str, article_text: str) -> int:
    chunks = chunk_documents(article_text)
    doc_id = hashlib.sha1(url.encode("utf-8")).hexdigest()

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source_url": url,
            "chunk_index": i,
            "doc_id": doc_id,
        }
        for i in range(len(chunks))
    ]

    if chunks:
        vs.add_texts(chunks, metadatas=metadatas, ids=ids)

    return len(chunks)


def similarity_docs(vs: Chroma, query: str, k: int = 5) -> list[dict]:
    docs = vs.similarity_search(query, k=k)

    return [
        {
            "text": doc.page_content,
            "source_url": doc.metadata.get("source_url", ""),
            "chunk_index": doc.metadata.get("chunk_index", -1),
        }
        for doc in docs
    ]