import os
import json
import base64
import csv

# Input: your Synthea FHIR output folder
FHIR_DIR = "output/fhir"
# Output: consolidated notes file
OUT_FILE = "data/synthetic_notes/notes.csv"

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

rows = []

for fname in os.listdir(FHIR_DIR):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(FHIR_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        try:
            bundle = json.load(f)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

    # FHIR bundles contain multiple resources
    if "entry" not in bundle:
        continue

    for entry in bundle["entry"]:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "DocumentReference":
            # Each DocumentReference may have multiple contents
            for content in resource.get("content", []):
                attachment = content.get("attachment", {})
                data = attachment.get("data")
                if data:
                    try:
                        note_text = base64.b64decode(data).decode("utf-8", errors="ignore")
                    except Exception:
                        note_text = "[decode error]"
                    rows.append({
                        "patient_id": resource.get("subject", {}).get("reference", ""),
                        "note": note_text.strip()
                    })

# Write to CSV
with open(OUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["patient_id", "note"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Extracted {len(rows)} notes into {OUT_FILE}")