# quick_check_chroma.py
import chromadb
from chromadb.config import Settings as ChromaSettings

persist_dir = "./data/vector_store"
collection_name = "notes"

client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings())
coll = client.get_collection(collection_name)

query = "Type 2 diabetes management plan with metformin"
res = coll.query(
    query_texts=[query],
    n_results=3,
)

for i, doc in enumerate(res["documents"][0]):
    print(f"\nTop {i+1} doc:")
    print(doc)
    print("Meta:", res["metadatas"][0][i])