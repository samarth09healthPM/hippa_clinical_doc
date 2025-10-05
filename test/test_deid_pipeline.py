import os
import json
from cryptography.fernet import Fernet
from app.deid_pipeline import DeidPipeline, run_file

def test_masking_basic(tmp_path):
    # Arrange: create a temp note with clear PHI and sections
    note = """HPI: John Smith presented on 03/15/2024.
Contact: (212) 555-7890.
Assessment and Plan: Start aspirin."""
    note_path = tmp_path / "temp_note.txt"
    note_path.write_text(note, encoding="utf-8")

    # Copy key into temp secure dir
    os.makedirs(tmp_path / "secure_store", exist_ok=True)
    with open("secure_store/fernet.key","rb") as f:
        (tmp_path / "secure_store" / "fernet.key").write_bytes(f.read())

    # Act
    os.makedirs(tmp_path / "data" / "outputs", exist_ok=True)
    os.makedirs(tmp_path / "secure_store", exist_ok=True)
    # Instantiate pipeline with temp key
    pipeline = DeidPipeline(fernet_key_path=str(tmp_path / "secure_store" / "fernet.key"))
    result = pipeline.run_on_text(note, note_id="temp_note")

    # Assert masked tags present
    masked = result["masked_text"]
    assert "[NAME]" in masked or "[PERSON]" in masked
    assert "[PHONE]" in masked
    assert "[DATE]" in masked

    # Assert sections preserved (headers remain)
    assert "HPI:" in masked
    assert "Assessment and Plan:" in masked

    # Decrypt span map and sanity check fields
    token = result["encrypted_span_map"]
    key = (tmp_path / "secure_store" / "fernet.key").read_bytes()
    f = Fernet(key)
    payload = json.loads(f.decrypt(token).decode("utf-8"))
    assert payload["meta"]["note_id"] == "temp_note"
    assert len(payload["spans"]) > 0
