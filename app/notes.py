from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# 1. Wrap SentenceTransformer in a LangChain-compatible class
class STEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

# 2. Instantiate embeddings
embeddings = STEmbeddings()

# 3. Create or load Chroma collection
db = Chroma(
    collection_name="notes",
    persist_directory="./data/vector_store",
    embedding_function=embeddings
)

# 4. Add some sample texts
texts = [
    "Patient presents with chest pain for 2 days.",
    "History of hypertension and diabetes.",
    "Currently taking metformin and lisinopril.",
    "No known drug allergies.",
    "Plan: schedule ECG and follow-up in 1 week."
]

metadatas = [
    {"note_id": "1", "section": "HPI", "chunk_index": 0},
    {"note_id": "1", "section": "PMH", "chunk_index": 0},
    {"note_id": "1", "section": "Medications", "chunk_index": 0},
    {"note_id": "1", "section": "Allergies", "chunk_index": 0},
    {"note_id": "1", "section": "Plan", "chunk_index": 0},
]

db.add_texts(texts=texts, metadatas=metadatas)

# 5. Persist to disk

print("Ingestion complete. Collection 'notes' is ready.")