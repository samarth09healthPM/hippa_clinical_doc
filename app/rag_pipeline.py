# app/rag_pipeline.py
# Day 7: Retriever + RAG baseline (retrieval only; generation comes on Day 8)
# Example usage:
#   python app/rag_pipeline.py --db_type chroma --persist_dir ./data/vector_store --collection notes --query "Summarize into HPI/Assessment/Plan" --top_k 5
#   python app/rag_pipeline.py --db_type faiss  --persist_dir ./data/vector_store_faiss --query "Extract Assessment and Plan" --top_k 5

import os
import argparse
import pickle
from typing import List, Dict
import uuid
import datetime
import shutil

from sentence_transformers import SentenceTransformer
import numpy as np

# LangChain vector store wrappers
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document

# For FAISS manual load if using custom persisted index
import faiss
from chromadb.config import Settings as ChromaSettings

def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    def embed_f(texts: List[str]) -> List[List[float]]:
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.tolist()
    return model, embed_f

def load_chroma(persist_dir: str, collection: str, embed_f):
    from langchain.embeddings.base import Embeddings
    class STEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embed_f(texts)
        def embed_query(self, text: str) -> List[float]:
            return embed_f([text])[0]

    embeddings = STEmbeddings()
    vectordb = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectordb

def load_faiss_langchain(persist_dir: str, embed_f):
    # If Day 6 saved FAISS with LangChain’s FAISS.save_local, we can do:
    # return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    # But Day 6 saved raw FAISS + meta.pkl; handle that manually and wrap.
    from langchain.embeddings.base import Embeddings
    class STEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embed_f(texts)
        def embed_query(self, text: str) -> List[float]:
            return embed_f([text])[0]
    embeddings = STEmbeddings()

    index_path = os.path.join(persist_dir, "index.faiss")
    meta_path = os.path.join(persist_dir, "meta.pkl")
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"FAISS files not found in {persist_dir}")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Build FAISS VectorStore from texts + metadata to leverage LC retriever
    texts = [m["text"] for m in meta]
    metadatas = [m["meta"] | {"id": m["id"]} for m in meta]
    vectordb = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    # Replace the underlying index with prebuilt (saves re-embedding cost when querying)
    vectordb.index = index
    return vectordb

def retrieve(vdb, query: str, top_k: int = 5):
    retriever = vdb.as_retriever(search_kwargs={"k": top_k})
    docs: List[Document] = retriever.invoke(query)
    return docs

def format_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        parts.append(f"[{i}] note_id={md.get('note_id')} section={md.get('section')} chunk_idx={md.get('chunk_index')}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Day 7: Retriever + RAG baseline (retrieval only).")
    parser.add_argument("--db_type", choices=["chroma", "faiss"], default="chroma")
    parser.add_argument("--persist_dir", default="./data/vector_store")
    parser.add_argument("--collection", default="notes")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Sure shot fix: Remove existing persist_dir if it exists
    if args.db_type == "chroma" and os.path.exists(args.persist_dir):
        shutil.rmtree(args.persist_dir)

    _, embed_f = load_embedder(args.model_name)

    if args.db_type == "chroma":
        vectordb = load_chroma(args.persist_dir, args.collection, embed_f)
    else:
        vectordb = load_faiss_langchain(args.persist_dir, embed_f)

    docs = retrieve(vectordb, args.query, args.top_k)
    context = format_context(docs)
    print("\n=== Retrieved Context (to feed Day 8 summarizer) ===\n")
    print(context)

if __name__ == "__main__":
    main()
