# test_deidentification.py - Test PII detection and anonymization

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import os

def test_deidentification():
    """Test de-identification pipeline on sample text"""
    
    print("🧪 Testing De-identification Pipeline...")
    
    # Initialize Presidio engines
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    
    # Sample clinical text with PHI
    sample_text = """
    Patient John Doe (DOB: 03/22/1985) was seen today. 
    His phone number is 555-123-4567 and he lives at 123 Main St, Boston MA.
    SSN: 123-45-6789. Email: john.doe@email.com
    """
    
    print(f"Original text:\n{sample_text}")
    print("-" * 50)
    
    # Step 1: Analyze for PII
    analysis_results = analyzer.analyze(
        text=sample_text,
        language='en'
    )
    
    print("Detected PII entities:")
    for result in analysis_results:
        detected_text = sample_text[result.start:result.end]
        print(f"- {result.entity_type}: '{detected_text}' (confidence: {result.score:.2f})")
    
    print("-" * 50)
    
    # Step 2: Anonymize the text
    anonymized_result = anonymizer.anonymize(
        text=sample_text,
        analyzer_results=analysis_results,
        operators={
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
            "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
            "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATE]"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"})
        }
    )
    
    print(f"Anonymized text:\n{anonymized_result.text}")
    print("-" * 50)
    
    # Test on our synthetic notes if they exist
    notes_file = 'data/synthetic_notes/clinical_note_01.txt'
    if os.path.exists(notes_file):
        print(f"\n🏥 Testing on synthetic clinical note...")
        
        with open(notes_file, 'r') as f:
            clinical_note = f.read()
        
        # Analyze clinical note
        clinical_results = analyzer.analyze(text=clinical_note, language='en')
        
        print(f"Found {len(clinical_results)} PII entities in clinical note:")
        for result in clinical_results:
            detected_text = clinical_note[result.start:result.end]
            print(f"- {result.entity_type}: '{detected_text}' (confidence: {result.score:.2f})")
        
        # Anonymize clinical note
        anonymized_clinical = anonymizer.anonymize(
            text=clinical_note,
            analyzer_results=clinical_results,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
        )
        
        # Save anonymized version
        os.makedirs('data/anonymized_notes', exist_ok=True)
        with open('data/anonymized_notes/anonymized_note_01.txt', 'w') as f:
            f.write(anonymized_clinical.text)
        
        print("✅ Anonymized clinical note saved to data/anonymized_notes/")
    
    print("\n🎉 De-identification test completed!")

if __name__ == "__main__":
    test_deidentification()
