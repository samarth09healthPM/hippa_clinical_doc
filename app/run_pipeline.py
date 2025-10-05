# run_pipeline.py
from rag_pipeline import retrieve_context   # <-- your Day 7 retriever
from transformers import pipeline

# 1. Load a summarization model
# Option A: summarization-tuned model (recommended for clean summaries)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Option B: instruction-tuned model (if you want to experiment with prompts)
# summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

# 2. Define a function to generate structured summary
def generate_summary(retrieved_text: str):
    # For BART summarizer (Option A)
    result = summarizer(retrieved_text, max_length=250, min_length=80, do_sample=False)
    return result[0]['summary_text']

    # If using Flan-T5 (Option B), uncomment this instead:
    """
    prompt = f'''
    You are a clinical summarization assistant.
    Use ONLY the provided context to create a structured summary.
    Do not invent information.

    Context:
    {retrieved_text}

    Write the output in this exact format:
    Chief Complaint: ...
    HPI: ...
    PMH: ...
    Medications: ...
    Allergies: ...
    Assessment: ...
    Plan: ...
    '''
    result = summarizer(prompt, max_new_tokens=300, do_sample=False)
    return result[0]['generated_text']
    """

# 3. Main execution
if __name__ == "__main__":
    query = "Summarize into HPI/Assessment/Plan"
    # Get top 5 relevant chunks from your vector store
    retrieved_text = retrieve_context(query, top_k=5)

    print("=== Retrieved Context ===")
    print(retrieved_text)
    print("\n=== Structured Clinical Summary ===")
    summary = generate_summary(retrieved_text)
    print(summary)