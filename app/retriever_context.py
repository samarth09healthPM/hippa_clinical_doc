from rag_pipeline import retrieve_context
from summarizer import generate_summary

query = "Summarize into HPI/Assessment/Plan"
retrieved_text = retrieve_context(query, top_k=5)
summary = generate_summary(retrieved_text)
print(summary)