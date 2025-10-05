import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Presidio
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import PatternRecognizer

# Define medical terms that should NOT be redacted
medical_terms_allowlist = [
    "substernal", "exertional", "pressure-like", "diaphoresis",
    "chest pain", "nausea", "radiation", "murmurs", "ischemia"
]

# Configure analyzer to ignore these terms
analyzer_config = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    "allow_list": medical_terms_allowlist  # Don't redact these
}

# NLP for optional section detection
import spacy

# If using medspacy, uncomment (preferred for clinical):
# import medspacy
# from medspacy.sectionizer import Sectionizer

# If not using medspacy, optional lightweight section tagging:
# We'll use regex on common headers as a fallback
import re

# Encryption
from cryptography.fernet import Fernet

@dataclass
class PHISpan:
    entity_type: str
    start: int
    end: int
    text: str
    section: str

SECTION_HEADERS = [
    # Common clinical sections; customize as needed
    "HPI", "History of Present Illness",
    "PMH", "Past Medical History",
    "Medications", "Allergies",
    "Assessment and Plan", "Assessment & Plan", "Assessment",
    "Plan", "ROS", "Review of Systems",
    "Physical Exam"
]

SECTION_PATTERN = re.compile(
    r"^(?P<header>(" + "|".join([re.escape(h) for h in SECTION_HEADERS]) + r"))\s*:\s*$",
    re.IGNORECASE | re.MULTILINE
)

TAG_MAP = {
    "PERSON": "[NAME]",
    "PHONE_NUMBER": "[PHONE]",
    "DATE_TIME": "[DATE]",
    "DATE": "[DATE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "US_SSN": "[SSN]"
}

class DeidPipeline:
    def __init__(self, fernet_key_path="secure_store/fernet.key"):
    """
    Initialize de-identification pipeline with Presidio
    """
    # For Streamlit Cloud: Generate key if not exists
    import os
    from cryptography.fernet import Fernet
    
    # Try to load existing key or generate new one
    try:
        if os.path.exists(fernet_key_path):
            with open(fernet_key_path, "rb") as f:
                key = f.read()
        else:
            # Generate new key for cloud deployment
            key = Fernet.generate_key()
            # Try to save it (might fail on read-only filesystems)
            try:
                os.makedirs(os.path.dirname(fernet_key_path), exist_ok=True)
                with open(fernet_key_path, "wb") as f:
                    f.write(key)
            except (PermissionError, OSError):
                # Cloud filesystem is read-only, just use the generated key
                pass
        
        self.cipher_suite = Fernet(key)
    except Exception as e:
        # Fallback: Generate temporary key for this session
        key = Fernet.generate_key()
        self.cipher_suite = Fernet(key)
    
    # Initialize Presidio components
    self.analyzer = AnalyzerEngine()
    self.anonymizer = AnonymizerEngine()
    
    # Load spaCy model
    try:
        self.nlp = spacy.load("en_core_web_lg")
    except OSError:
        # If model not found, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
        self.nlp = spacy.load("en_core_web_lg")

    def _detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Lightweight section finder:
        Return list of (section_title, start_idx, end_idx_of_section_block)
        """
        # Find headers by regex, map their start positions
        headers = []
        for m in SECTION_PATTERN.finditer(text):
            headers.append((m.group("header"), m.start()))
        # Add end sentinel
        headers.append(("[END]", len(text)))

        sections = []
        for i in range(len(headers) - 1):
            title, start_pos = headers[i]
            next_title, next_pos = headers[i+1]
            sections.append((title.strip(), start_pos, next_pos))
        if not sections:
            # Single default section if none found
            sections = [("DOCUMENT", 0, len(text))]
        return sections

    def _find_section_for_span(self, sections, start_idx) -> str:
        for title, s, e in sections:
            if s <= start_idx < e:
                return title
        return "DOCUMENT"

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        # Detect entities
        results = self.analyzer.analyze(text=text, language="en")
        # Convert to dict for consistency
        detections = []
        for r in results:
            detections.append({
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score
            })
        return detections

    def mask(self, text: str, detections: List[Dict[str, Any]]) -> Tuple[str, List[PHISpan]]:
        """
        Replace spans with tags safely (right-to-left to maintain indices).
        """
        # Determine sections for context
        sections = self._detect_sections(text)

        # Build PHI span records
        spans: List[PHISpan] = []
        for d in detections:
            entity = d["entity_type"]
            start = d["start"]
            end = d["end"]
            original = text[start:end]
            section = self._find_section_for_span(sections, start)
            spans.append(PHISpan(entity_type=entity, start=start, end=end, text=original, section=section))

        # Replace from the end to avoid index shifting
        masked = text
        for d in sorted(detections, key=lambda x: x["start"], reverse=True):
            entity = d["entity_type"]
            start = d["start"]
            end = d["end"]
            tag = TAG_MAP.get(entity, f"[{entity}]")
            masked = masked[:start] + tag + masked[end:]

        return masked, spans

    def encrypt_span_map(self, spans: List[PHISpan], meta: Dict[str, Any]) -> bytes:
        payload = {
            "meta": meta,
            "spans": [s.__dict__ for s in spans]
        }
        blob = json.dumps(payload).encode("utf-8")
        token = self.fernet.encrypt(blob)
        return token

    def run_on_text(self, text: str, note_id: str) -> Dict[str, Any]:
        detections = self.analyze(text)
        masked, spans = self.mask(text, detections)

        # Encrypt span map
        token = self.encrypt_span_map(
            spans=spans,
            meta={"note_id": note_id}
        )

        return {
            "masked_text": masked,
            "encrypted_span_map": token
        }

def _read_text_with_fallback(path: str) -> str:
    # 1) Try UTF-8 (preferred for cross-platform)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    # 2) Try Windows-1252 (common for Notepad/docx copy-paste on Windows)
    try:
        with open(path, "r", encoding="cp1252") as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    # 3) Last resort: decode with replacement to avoid crashing; preserves structure
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def run_file(input_path: str, outputs_dir: str = "data/outputs", secure_dir: str = "secure_store"):
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(secure_dir, exist_ok=True)

    note_id = os.path.splitext(os.path.basename(input_path))[0]
    text = _read_text_with_fallback(input_path)

    pipeline = DeidPipeline()
    result = pipeline.run_on_text(text, note_id=note_id)

    # Save masked text normalized to UTF-8
    out_txt = os.path.join(outputs_dir, f"{note_id}.deid.txt")
    with open(out_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write(result["masked_text"])

    # Save encrypted span map (binary)
    out_bin = os.path.join(secure_dir, f"{note_id}.spanmap.enc")
    with open(out_bin, "wb") as f:
        f.write(result["encrypted_span_map"])

    print(f"De-identified text -> {out_txt}")
    print(f"Encrypted span map -> {out_bin}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="De-identify a clinical note and save encrypted span map.")
    parser.add_argument("--input", required=True, help="Path to input .txt note")
    parser.add_argument("--outputs_dir", default="data/outputs", help="Output folder for masked text")
    parser.add_argument("--secure_dir", default="secure_store", help="Folder for encrypted span maps")
    args = parser.parse_args()
    run_file(args.input, args.outputs_dir, args.secure_dir)
