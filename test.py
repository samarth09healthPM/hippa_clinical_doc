# test_setup.py - Quick installation test

print("Testing library imports...")

try:
    import spacy
    print("✅ spaCy imported successfully")
    
    # Test loading English model
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy English model loaded")
    
except Exception as e:
    print(f"❌ spaCy error: {e}")

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    print("✅ Presidio imported successfully")
except Exception as e:
    print(f"❌ Presidio error: {e}")

try:
    import medspacy
    print("✅ medSpaCy imported successfully")
except Exception as e:
    print(f"❌ medSpaCy error: {e}")

try:
    import chromadb
    print("✅ ChromaDB imported successfully")
except Exception as e:
    print(f"❌ ChromaDB error: {e}")

try:
    import streamlit
    print("✅ Streamlit imported successfully")
except Exception as e:
    print(f"❌ Streamlit error: {e}")

print("\n🎉 Setup test complete!")
