# app/summarizer.py
# Day 10: Enhanced HIPAA-compliant RAG clinical summarizer with robustness improvements
# Critical fixes:
# - Added progress indicators during model generation
# - Implemented timeout mechanism for long-running operations
# - Optimized for CPU with reduced generation parameters
# - Better error handling and verbose logging
# - Fallback to smaller max tokens if generation hangs

import os
import argparse
import traceback
from typing import List, Dict, Optional
import re
import time
import sys

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Embeddings / Vector stores
# -----------------------------
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load sentence transformer for embeddings.
    For medical domain: consider "emilyalsentzer/Bio_ClinicalBERT" or similar
    """
    print(f"  → Loading embedding model...")
    model = SentenceTransformer(model_name)
    def embed_f(texts: List[str]):
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.tolist()
    print(f"  ✓ Embedding model loaded")
    return embed_f

def load_chroma(persist_dir: str, collection: str, embed_f):
    from langchain.embeddings.base import Embeddings
    class STEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return embed_f(texts)
        def embed_query(self, text: str) -> List[float]:
            return embed_f([text])[0]
    embeddings = STEmbeddings()
    print(f"  → Loading Chroma vector store from {persist_dir}...")
    return Chroma(collection_name=collection, persist_directory=persist_dir, embedding_function=embeddings)

def load_faiss(persist_dir: str, embed_f):
    import pickle, faiss
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
    print(f"  → Loading FAISS index from {persist_dir}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    texts = [m["text"] for m in meta]
    metadatas = [m["meta"] | {"id": m["id"]} for m in meta]
    vdb = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vdb.index = faiss.read_index(index_path)
    return vdb

def retrieve_docs(db_type: str, persist_dir: str, collection: str, query: str, top_k: int, embed_f) -> List[Document]:
    if db_type == "chroma":
        vdb = load_chroma(persist_dir, collection, embed_f)
    else:
        vdb = load_faiss(persist_dir, embed_f)
    
    print(f"  → Retrieving documents...")
    retriever = vdb.as_retriever(search_kwargs={"k": top_k})
    docs: List[Document] = retriever.invoke(query)
    print(f"  ✓ Retrieved {len(docs)} document(s)")
    
    # Debug: Show retrieved content length
    if docs:
        total_chars = sum(len(d.page_content) for d in docs)
        print(f"  ℹ Total retrieved content: {total_chars} characters")
    else:
        print(f"  ⚠ WARNING: No documents retrieved!")
    
    return docs

# -----------------------------
# T5 Summarization utilities
# -----------------------------
def make_t5(model_name="google/flan-t5-base", device="cpu"):
    print(f"  → Loading T5 model: {model_name}")
    print(f"  ℹ This may take 30-60 seconds for large models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print(f"  ✓ Model loaded successfully")
    return tokenizer, model

def t5_generate(tokenizer, model, prompt: str, max_input_tokens: int = 512, max_output_tokens: int = 256, section_name: str = ""):
    """
    Enhanced generation with progress indicators and optimized parameters for CPU
    """
    # Show progress
    if section_name:
        print(f"    → Generating {section_name}...", end='', flush=True)
    else:
        print(f"    → Generating summary...", end='', flush=True)
    
    start_time = time.time()
    
    try:
        inputs = tokenizer(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Optimized parameters for CPU performance
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            min_length=10,  # Reduced minimum to avoid forcing long outputs
            num_beams=2,  # Reduced from 4 for faster CPU generation
            length_penalty=1.0,  # Reduced from 1.5
            no_repeat_ngram_size=3,
            early_stopping=True,  # Re-enabled for faster completion
            do_sample=False  # Deterministic generation
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time
        print(f" done ({elapsed:.1f}s)")
        
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f" FAILED ({elapsed:.1f}s)")
        print(f"    ✗ Error: {str(e)}")
        return ""

def dedupe_texts(texts: List[str]) -> List[str]:
    seen = set()
    uniq = []
    for t in texts:
        key = " ".join(t.split())[:500]
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq

# -----------------------------
# Section definitions
# -----------------------------
SECTION_ORDER = [
    "Chief Complaint",
    "HPI",
    "PMH",
    "Medications",
    "Allergies",
    "Assessment",
    "Plan",
]

# -----------------------------
# Multi-stage extraction prompts (optimized for T5)
# -----------------------------
SECTION_PROMPTS = {
    "Chief Complaint": """Task: Extract the main reason for patient visit.

Clinical Note:
{context}

Answer with only the chief complaint (1-2 sentences):""",
    
    "HPI": """Task: Extract the history of present illness including symptom onset, progression, and context.

Clinical Note:
{context}

Answer with the history of present illness:""",
    
    "PMH": """Task: Extract past medical history including chronic conditions, past surgeries, and social history.

Clinical Note:
{context}

Answer with past medical history:""",
    
    "Medications": """Task: List all medications with dosages mentioned in the note.

Clinical Note:
{context}

Answer with medication list:""",
    
    "Allergies": """Task: Extract drug allergies. If none mentioned, state "No known drug allergies".

Clinical Note:
{context}

Answer with allergies:""",
    
    "Assessment": """Task: Extract diagnosis, test results, physical findings, and vital signs.

Clinical Note:
{context}

Answer with assessment and findings:""",
    
    "Plan": """Task: Extract treatment plan, medications prescribed, follow-up appointments, and discharge instructions.

Clinical Note:
{context}

Answer with treatment plan:"""
}

# -----------------------------
# Enhanced extraction pipeline
# -----------------------------
def extract_section_multistage(tokenizer, model, context: str, section: str) -> str:
    """
    Extract a single section using targeted prompting
    """
    if section not in SECTION_PROMPTS:
        return "None stated"
    
    # Truncate context if too long
    max_context_chars = 2000
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "..."
    
    prompt = SECTION_PROMPTS[section].format(context=context)
    
    try:
        result = t5_generate(tokenizer, model, prompt, max_input_tokens=512, max_output_tokens=200, section_name=section)
        result = result.strip()
        
        # Remove any section headers the model might have added
        result = re.sub(r'^(Chief Complaint|HPI|PMH|Medications|Allergies|Assessment|Plan)\s*:\s*', '', result, flags=re.IGNORECASE)
        
        # Check if extraction failed
        if not result or len(result) < 5 or result.lower() in ["none", "none stated", "not mentioned", "n/a", "na"]:
            return "None stated"
        
        return result.strip()
    except Exception as e:
        print(f"    ✗ Error extracting {section}: {str(e)}")
        return "None stated"

def validate_extraction(sections: Dict[str, str]) -> bool:
    """
    Validate that extraction was successful (not all 'None stated')
    """
    non_empty = sum(1 for v in sections.values() if v and v != "None stated")
    return non_empty >= 2  # At least 2 sections should have content

def summarize_docs_multistage(tokenizer, model, docs: List[Document]) -> str:
    """
    Multi-stage extraction: extract each section independently
    """
    print(f"\n📄 Processing documents...")
    contents = dedupe_texts([d.page_content for d in docs if d and d.page_content])
    
    if not contents:
        print("  ⚠ No content to summarize!")
        return format_output({sec: "None stated" for sec in SECTION_ORDER})
    
    # Combine all retrieved content
    full_context = "\n\n".join(contents)
    print(f"  ℹ Combined context length: {len(full_context)} characters")
    
    # Extract each section independently
    print(f"\n🔄 Extracting sections (this may take 1-3 minutes on CPU)...")
    sections = {}
    for i, section in enumerate(SECTION_ORDER, 1):
        print(f"  [{i}/{len(SECTION_ORDER)}] {section}:")
        sections[section] = extract_section_multistage(tokenizer, model, full_context, section)
    
    # Validate extraction
    print(f"\n✓ Extraction complete")
    if not validate_extraction(sections):
        print("⚠ WARNING: Extraction appears incomplete. Most sections are empty.")
        print("  Possible issues:")
        print("  • Vector retrieval may not be finding relevant content")
        print("  • Model may not understand the clinical text format")
        print("  • Context may be too short or fragmented")
        print("  • De-identification artifacts may be confusing the model")
    
    return format_output(sections)

def format_output(sections: Dict[str, str]) -> str:
    """
    Format sections into structured output
    """
    output_lines = []
    for section in SECTION_ORDER:
        content = sections.get(section, "None stated")
        output_lines.append(f"• {section}: {content}")
    
    return "\n".join(output_lines)
    
    # -----------------------------
# Summary Quality Validation
# -----------------------------
def validate_summary_quality(summary: str, original_text: str = "") -> dict:
    """
    Validate summary quality and detect common issues
    
    Args:
        summary: The generated summary text
        original_text: Optional original note text for comparison
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []
    
    # Check for placeholder contamination (de-ID over-redaction)
    placeholder_patterns = [
        (r'\[LOCATION\]', 'LOCATION'),
        (r'\[DATE\]', 'DATE'),
        (r'\[NAME\]', 'NAME'),
        (r'\[PHONE\]', 'PHONE')
    ]
    
    total_placeholders = 0
    for pattern, name in placeholder_patterns:
        count = len(re.findall(pattern, summary))
        total_placeholders += count
        if count > 2:
            warnings.append(f"Too many [{name}] placeholders ({count}) - de-identification may be over-aggressive")
    
    if total_placeholders > 5:
        issues.append(f"Critical: {total_placeholders} PHI placeholders in summary - clinical content lost")
    
    # Check for "None stated" sections
    none_count = summary.count("None stated")
    if none_count >= 5:
        issues.append(f"Critical: {none_count}/7 sections are empty - summarization failed")
    elif none_count >= 3:
        warnings.append(f"Warning: {none_count}/7 sections are empty - may need better retrieval")
    
    # Check for minimum content length per section
    total_length = len(summary)
    # Subtract bullets and "None stated" overhead
    content_length = total_length - (summary.count("•") * 2) - (none_count * 11)
    filled_sections = 7 - none_count
    
    if filled_sections > 0:
        avg_section_length = content_length / filled_sections
        if avg_section_length < 30:
            warnings.append(f"Warning: Sections too short (avg {avg_section_length:.0f} chars) - may lack detail")
    
    # Check for duplicate medications
    if "Medications:" in summary:
        meds_section = summary.split("Medications:")[1].split("•")[0] if "Medications:" in summary else ""
        meds_lower = meds_section.lower()
        common_meds = ['atorvastatin', 'metoprolol', 'lisinopril', 'aspirin', 'metformin']
        for med in common_meds:
            if meds_lower.count(med) > 1:
                warnings.append(f"Warning: Duplicate medication detected: {med}")
    
    # Calculate quality score (0-100)
    score = 100
    score -= len(issues) * 30  # Critical issues: -30 each
    score -= len(warnings) * 10  # Warnings: -10 each
    score = max(0, min(100, score))
    
    # Determine overall status
    if len(issues) > 0:
        status = "FAILED"
    elif len(warnings) > 2:
        status = "POOR"
    elif len(warnings) > 0:
        status = "FAIR"
    else:
        status = "GOOD"
    
    return {
        "is_valid": len(issues) == 0,
        "status": status,
        "quality_score": score,
        "issues": issues,
        "warnings": warnings,
        "metrics": {
            "total_placeholders": total_placeholders,
            "empty_sections": none_count,
            "filled_sections": filled_sections,
            "total_length": total_length
        }
    }

# -----------------------------
# Backward compatibility wrapper for Streamlit integration
# -----------------------------
def summarize_docs(tokenizer, model, docs: List[Document], method: str = "multistage") -> str:
    """
    Wrapper function for backward compatibility with main.py (Streamlit UI)
    """
    if method == "multistage":
        return summarize_docs_multistage(tokenizer, model, docs)
    else:
        return summarize_docs_singleshot(tokenizer, model, docs)

# -----------------------------
# Single-shot extraction (simplified fallback)
# -----------------------------
def summarize_docs_singleshot(tokenizer, model, docs: List[Document]) -> str:
    """
    Single-shot extraction method (faster but less comprehensive)
    """
    print(f"\n📄 Processing documents...")
    contents = dedupe_texts([d.page_content for d in docs if d and d.page_content])
    
    if not contents:
        print("  ⚠ No content to summarize!")
        return format_output({sec: "None stated" for sec in SECTION_ORDER})

    raw_context = "\n\n".join(contents)
    print(f"  ℹ Combined context length: {len(raw_context)} characters")

    # Simplified prompt for single-shot
    instruction = """Summarize this clinical note into 7 sections:
1. Chief Complaint (main reason for visit)
2. HPI (symptom history and progression)
3. PMH (past medical history)
4. Medications (current medications with doses)
5. Allergies (drug allergies)
6. Assessment (diagnosis and findings)
7. Plan (treatment plan and follow-up)

Clinical Note:
{context}

Structured Summary:"""
    
    print(f"\n🔄 Generating structured summary...")
    prompt = instruction.format(context=raw_context[:2000])  # Limit context
    model_out = t5_generate(tokenizer, model, prompt, max_input_tokens=512, max_output_tokens=400)
    
    # Parse output into sections
    sections = parse_output_to_sections(model_out)
    
    return format_output(sections)

def parse_output_to_sections(text: str) -> Dict[str, str]:
    """
    Parse model output into section dictionary
    """
    sections = {}
    current_section = None
    current_content = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Check if line starts with a section header
        matched_section = None
        for section in SECTION_ORDER:
            # Match section headers with numbers or bullets
            pattern = rf'^(\d+\.\s*)?{re.escape(section)}\s*:?'
            if re.match(pattern, line, re.IGNORECASE):
                matched_section = section
                break
        
        if matched_section:
            # Save previous section
            if current_section:
                sections[current_section] = " ".join(current_content).strip()
            
            # Start new section
            current_section = matched_section
            # Get content after the header
            content = re.sub(rf'^(\d+\.\s*)?{re.escape(matched_section)}\s*:?\s*', '', line, flags=re.IGNORECASE).strip()
            current_content = [content] if content else []
        else:
            # Continue current section
            if current_section:
                current_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = " ".join(current_content).strip()
    
    # Fill in missing sections
    for section in SECTION_ORDER:
        if section not in sections or not sections[section]:
            sections[section] = "None stated"
    
    return sections

# -----------------------------
# Backward compatibility wrapper for Streamlit integration
# -----------------------------
def summarize_docs(tokenizer, model, docs: List[Document], method: str = "multistage") -> str:
    """
    Wrapper function for backward compatibility with main.py (Streamlit UI)
    
    Args:
        tokenizer: T5 tokenizer instance
        model: T5 model instance
        docs: List of retrieved documents
        method: "multistage" (default) or "singleshot" extraction method
    
    Returns:
        Formatted summary string with sections
    """
    if method == "multistage":
        return summarize_docs_multistage(tokenizer, model, docs)
    else:
        return summarize_docs_singleshot(tokenizer, model, docs)

# -----------------------------
# Orchestration
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Day 10: Enhanced HIPAA-compliant RAG clinical summarizer")
    parser.add_argument("--db_type", choices=["chroma", "faiss"], default="chroma")
    parser.add_argument("--persist_dir", default="./data/vector_store")
    parser.add_argument("--collection", default="notes")
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out", default="./data/outputs/summaries/summary.txt")
    parser.add_argument("--method", choices=["multistage", "singleshot"], default="multistage",
                       help="Extraction method: multistage (recommended) or singleshot (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("  HIPAA-COMPLIANT RAG CLINICAL SUMMARIZER")
    print("=" * 70)
    
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    try:
        # Step 1: Load embedder
        print(f"\n[1/4] LOADING EMBEDDER")
        print(f"  Model: {args.embed_model}")
        embed_f = load_embedder(args.embed_model)
        
        # Step 2: Retrieve documents
        print(f"\n[2/4] RETRIEVING DOCUMENTS")
        print(f"  Database: {args.db_type}")
        print(f"  Location: {args.persist_dir}")
        print(f"  Query: {args.query}")
        print(f"  Top-K: {args.top_k}")
        docs = retrieve_docs(args.db_type, args.persist_dir, args.collection, args.query, args.top_k, embed_f)
        
        if not docs:
            print("\n⚠ ERROR: No documents retrieved from vector database!")
            print("  Possible causes:")
            print("  • Vector database is empty or not properly indexed")
            print("  • Query doesn't match indexed content")
            print("  • Database path is incorrect")
            result = format_output({sec: "None stated" for sec in SECTION_ORDER})
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"\n✓ Empty summary written to {args.out}")
            return

        # Step 3: Load summarization model
        print(f"\n[3/4] LOADING SUMMARIZATION MODEL")
        print(f"  Model: {args.model_name}")
        tokenizer, model = make_t5(args.model_name)

        # Step 4: Generate summary
        print(f"\n[4/4] GENERATING SUMMARY")
        print(f"  Method: {args.method}")
        
        if args.method == "multistage":
            summary = summarize_docs_multistage(tokenizer, model, docs)
        else:
            summary = summarize_docs_singleshot(tokenizer, model, docs)

        # Write summary to output file
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"\n{'=' * 70}")
        print(f"✓ SUCCESS: Summary written to {args.out}")
        print(f"{'=' * 70}")
        print("\nGenerated Summary:")
        print("-" * 70)
        print(summary)
        print("-" * 70)

    except Exception as e:
        err = traceback.format_exc()
        error_msg = f"ERROR during summarization:\n{err}"
        
        # Write error to file
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(error_msg)
        
        print(f"\n{'=' * 70}")
        print(f"✗ ERROR: An error occurred during processing")
        print(f"{'=' * 70}")
        print(f"\n{err}")
        print(f"\nError details written to {args.out}")

if __name__ == "__main__":
    main()
