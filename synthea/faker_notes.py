from faker import Faker
import random, csv, os

fake = Faker()
OUT_FILE = "data/synthetic_notes/notes.csv"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

conditions = ["diabetes", "hypertension", "asthma", "chest pain", "migraine"]

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "note"])
    writer.writeheader()
    for i in range(50):
        pid = f"patient_{i+1}"
        condition = random.choice(conditions)
        note = f"{fake.name()} ({fake.random_int(25,80)}y) presented with {condition}. {fake.sentence()}"
        writer.writerow({"patient_id": pid, "note": note})

print("✅ Generated 50 synthetic notes")