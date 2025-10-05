# HIPAA-Compliant RAG Clinical Summarizer

🏥 An AI-powered clinical note summarization system that ensures HIPAA compliance through de-identification, retrieval-augmented generation (RAG), and secure audit logging.

## 🎯 Project Overview

This project demonstrates a production-ready clinical documentation tool built for healthcare AI applications. It showcases:

- **De-identification**: Automated PHI removal using Microsoft Presidio
- **RAG Architecture**: Vector-based retrieval with Chroma/FAISS
- **Clinical Summarization**: Structured note generation using Flan-T5
- **Security**: Encryption, audit logging, and RBAC authentication
- **Quality Validation**: Automated quality checks with Ragas metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git installed
- 8GB RAM minimum

### Installation

1. **Clone the repository**
git clone https://github.com/samarth09healthPM/hipaa-compliant-rag-clinical
cd hipaa-compliant-rag-clinical

2. **Create virtual environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt
python -m spacy download en_core_web_lg

4. **Create configuration file**
Create `app/streamlit_config.yaml`:
credentials:
usernames:
admin1:
email: admin@example.com
failed_login_attempts: 0
logged_in: false
name: Admin User
password: $2b$12$... # Use streamlit-authenticator to generate
role: admin
user1:
email: user@example.com
failed_login_attempts: 0
logged_in: false
name: Regular User
password: $2b$12$... # Use streamlit-authenticator to generate
role: user
cookie:
expiry_days: 30
key: random_signature_key_here
name: hipaa_rag_cookie

5. **Run the application**
streamlit run app/main.py

## 📁 Project Structure
hipaa-compliant-rag-clinical/
├── app/
│ ├── main.py # Streamlit UI
│ ├── deid_pipeline.py # De-identification
│ ├── indexer.py # Vector database indexing
│ ├── rag_pipeline.py # RAG retrieval logic
│ ├── summarizer.py # Clinical summarization
│ ├── audit.py # Audit logging
│ └── streamlit_config.yaml # Authentication config
├── data/
│ ├── raw/ # Sample clinical notes
│ ├── deidentified/ # De-identified outputs
│ ├── vector_store/ # Vector databases
│ └── outputs/ # Generated summaries
├── logs/ # Audit logs
├── docs/ # Documentation
├── tests/ # Unit tests
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
└── README.md # This file

text

## 🔐 Security Features

- ✅ PHI de-identification (HIPAA Safe Harbor)
- ✅ Role-based access control (RBAC)
- ✅ Encrypted audit logging
- ✅ Session-specific vector stores
- ✅ No PHI in logs or database


## 🎓 Use Cases

This project is designed for:
- Senior Product Manager portfolio (AI/ML in healthcare)
- Healthcare AI startup demonstrations
- Clinical documentation automation research
- HIPAA compliance learning


## 👤 Author

Built by Samarth Sharma as a portfolio project for healthcare AI product management.

Connect with me on [LinkedIn](https://www.linkedin.com/in/i-am-samarth-sharma/)

## ⚠️ Disclaimer

This is a demonstration project using synthetic clinical data. **NOT approved for use with real patient data.**
