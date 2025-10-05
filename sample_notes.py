# sample_notes.py - Create synthetic clinical notes for testing

import os
from datetime import datetime, timedelta
import random

# Create sample clinical notes (completely fake)
sample_notes = [
    """
PROGRESS NOTE
Date: 2024-01-15
Patient: John Smith
DOB: 1985-03-22
MRN: 12345

CHIEF COMPLAINT: Follow-up for Type 2 diabetes

HISTORY OF PRESENT ILLNESS:
Patient is a 39-year-old male with a history of Type 2 diabetes mellitus 
diagnosed 3 years ago. He reports good adherence to metformin 1000mg twice 
daily. Home glucose readings have been ranging 110-140 mg/dL fasting. 
Denies polyuria, polydipsia, or blurred vision. No recent hypoglycemic episodes.

PHYSICAL EXAMINATION:
Vital Signs: BP 128/82, HR 76, Temp 98.2°F
General: Well-appearing male in no acute distress
HEENT: PERRL, EOM intact, no retinopathy noted

ASSESSMENT AND PLAN:
1. Type 2 Diabetes Mellitus - well controlled
   - Continue metformin 1000mg BID
   - HbA1c due, will order today
   - Return in 3 months
2. Hypertension - mild elevation
   - Start lifestyle modifications
   - Return in 6 weeks for BP recheck

Provider: Dr. Sarah Johnson, MD
Contact: (555) 123-4567
    """,
    
    """
DISCHARGE SUMMARY
Admission Date: 2024-01-10
Discharge Date: 2024-01-12

PATIENT: Maria Garcia
DOB: 1978-11-05  
MRN: 67890
Address: 123 Main Street, Anytown, CA 90210

ADMISSION DIAGNOSIS: Pneumonia

HISTORY:
Patient is a 46-year-old female who presented to ED with 3-day history 
of productive cough, fever, and shortness of breath. Initial chest X-ray 
showed right lower lobe infiltrate consistent with pneumonia.

HOSPITAL COURSE:
Started on IV antibiotics (ceftriaxone 1g daily). Patient showed good 
clinical response with resolution of fever by day 2. Repeat chest X-ray 
on day 2 showed improvement of infiltrate.

DISCHARGE MEDICATIONS:
1. Azithromycin 500mg daily x 5 days
2. Albuterol inhaler as needed for cough
3. Return to work note provided

FOLLOW-UP: Primary care in 1 week
Emergency contact: Spouse - Carlos Garcia (555) 987-6543

Dr. Michael Chen, MD
Internal Medicine
    """,
    
    """
SOAP NOTE - Cardiology Consultation
Date: 2024-01-20

PATIENT: Robert Johnson  
DOB: 1965-07-15
SSN: 123-45-6789
Phone: (555) 333-2222

SUBJECTIVE:
Patient reports new onset chest pain for past 2 weeks. Pain is substernal, 
described as pressure-like, occurs with exertion, relieved by rest. 
Denies radiation to arms or jaw. No associated nausea or diaphoresis.

OBJECTIVE:
Vitals: BP 145/90, HR 88, RR 16, O2 Sat 98%
Cardiac exam: Regular rate and rhythm, no murmurs
ECG: Normal sinus rhythm, no ST changes
Stress test: Positive for inducible ischemia

ASSESSMENT:
Coronary artery disease, likely stable angina

PLAN:
1. Start atorvastatin 40mg daily
2. Metoprolol 50mg BID
3. Cardiac catheterization scheduled
4. Dietary consultation
5. Return in 2 weeks

Dr. Lisa Wang, MD, FACC
Cardiology Associates
Email: lwang@cardiology.com
    """
]

def create_synthetic_notes():
    """Create synthetic clinical notes files"""
    
    # Ensure directory exists
    os.makedirs('data/synthetic_notes', exist_ok=True)
    
    # Save each note as a separate file
    for i, note in enumerate(sample_notes, 1):
        filename = f'data/synthetic_notes/clinical_note_{i:02d}.txt'
        with open(filename, 'w') as f:
            f.write(note.strip())
        print(f"Created: {filename}")
    
    # Create a combined file for easy processing
    with open('data/synthetic_notes/all_notes.txt', 'w') as f:
        for i, note in enumerate(sample_notes, 1):
            f.write(f"=== CLINICAL NOTE {i:02d} ===\n")
            f.write(note.strip())
            f.write(f"\n\n{'='*50}\n\n")
    
    print(f"Created combined file: data/synthetic_notes/all_notes.txt")
    print(f"Total notes created: {len(sample_notes)}")

if __name__ == "__main__":
    create_synthetic_notes()
    print("✅ Synthetic clinical notes created successfully!")
