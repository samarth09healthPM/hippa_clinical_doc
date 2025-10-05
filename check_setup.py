# check_setup.py - Verify entire Day 4 setup

import os
import sys

def check_setup():
    """Check if Day 4 setup is complete"""
    
    print("🔍 Day 4 Setup Verification")
    print("=" * 40)
    
    checks = []
    
    # 1. Check virtual environment
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    checks.append(("Virtual environment active", venv_active))
    
    # 2. Check key libraries
    try:
        import spacy
        spacy_ok = True
        # Check if model is available
        nlp = spacy.load("en_core_web_sm")
        spacy_model_ok = True
    except:
        spacy_ok = False
        spacy_model_ok = False
    
    checks.append(("spaCy library installed", spacy_ok))
    checks.append(("spaCy English model available", spacy_model_ok))
    
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        presidio_ok = True
    except:
        presidio_ok = False
    
    checks.append(("Presidio libraries installed", presidio_ok))
    
    # 3. Check data folders
    data_structure = [
        "data/synthetic_notes",
        "data/raw_synthea", 
        "data/anonymized_notes"
    ]
    
    for folder in data_structure:
        folder_exists = os.path.exists(folder)
        checks.append((f"Folder {folder}", folder_exists))
    
    # 4. Check synthetic notes
    sample_notes_exist = os.path.exists('data/synthetic_notes/clinical_note_01.txt')
    checks.append(("Sample clinical notes created", sample_notes_exist))
    
    # 5. Check Synthea output (optional)
    synthea_exists = os.path.exists('synthea/synthea-with-dependencies.jar')
    synthea_output = os.path.exists('synthea/output') or os.path.exists('data/raw_synthea/csv')
    
    checks.append(("Synthea JAR downloaded", synthea_exists))
    checks.append(("Synthea data generated", synthea_output))
    
    # Print results
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    # Summary
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    print(f"\n📊 Setup Status: {passed}/{total} checks passed")
    
    if passed >= total - 2:  # Allow 2 optional items to fail
        print("🎉 Day 4 setup is COMPLETE!")
        print("\n📝 Next steps:")
        print("- Day 5: Build de-identification pipeline")
        print("- Week 2: Vector database and RAG implementation")
    else:
        print("⚠️  Some setup issues need attention")
        print("Review the failed checks above")

if __name__ == "__main__":
    check_setup()
