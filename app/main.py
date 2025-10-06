# --- Imports and page setup ---
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import uuid
import datetime
from audit import AuditLogger

st.set_page_config(page_title="Clinical Summarizer", layout="wide")
st.title("HIPAA-compliant Clinical RAG Summarizer (MVP)")

# --- Authentication setup ---
def load_config():
    """Load configuration from Streamlit secrets or local YAML"""
    try:
        # Check if running on Streamlit Cloud (secrets available)
        if "credentials" in st.secrets:
            # Convert immutable Streamlit secrets to mutable dict
            config = {
                "credentials": {
                    "usernames": {}
                },
                "cookie": {
                    "name": str(st.secrets["cookie"]["name"]),
                    "key": str(st.secrets["cookie"]["key"]),
                    "expiry_days": int(st.secrets["cookie"]["expiry_days"])
                }
            }
            
            # Convert each user to mutable dict
            for username, user_data in st.secrets["credentials"]["usernames"].items():
                config["credentials"]["usernames"][str(username)] = {
                    "email": str(user_data["email"]),
                    "failed_login_attempts": int(user_data.get("failed_login_attempts", 0)),
                    "logged_in": bool(user_data.get("logged_in", False)),
                    "name": str(user_data["name"]),
                    "password": str(user_data["password"]),
                    "role": str(user_data["role"])
                }
            
            return config
        else:
            # Local development: Load from YAML file
            with open("app/streamlit_config.yaml") as f:
                return yaml.load(f, Loader=SafeLoader)
                
    except FileNotFoundError:
        st.error("⚠️ Configuration file not found. Please set up authentication.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Configuration error: {e}")
        st.info("Make sure secrets are configured in Streamlit Cloud settings.")
        st.stop()

# Load config
config = load_config()

# Create authenticator with mutable config
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# Render the login widget
authenticator.login(location="sidebar")

# Read values from session_state
auth_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")
name = st.session_state.get("name")

if auth_status is False:
    st.error("Invalid username or password")
    st.stop()
elif auth_status is None:
    st.info("Please log in")
    st.stop()
else:
    role = config["credentials"]["usernames"][username]["role"]
    st.session_state["role"] = role
    with st.sidebar:
        st.header("Clinical RAG Summarizer")
        st.markdown("HIPAA-compliant, secure, and easy to use.")
        st.markdown("---")
        st.success(f"Logged in as {name}")
        st.markdown(f"**Role:** {role}")
        authenticator.logout("Logout", location="sidebar")
        st.markdown("---")
        st.info("Use the tabs above to upload notes, generate summaries, and view logs.")

# Clear ChromaDB cache to prevent singleton conflicts
try:
    from chromadb.api.client import SharedSystemClient
    SharedSystemClient.clear_system_cache()
except:
    pass

# Generate a unique persist_dir for each session if not already set
if "persist_dir" not in st.session_state:
    if st.session_state.get("username"):
        st.session_state["persist_dir"] = f"./data/vector_store_{st.session_state['username']}"
    else:
        unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        st.session_state["persist_dir"] = f"./data/vector_store_{unique_id}"

# Initialize the audit logger
audit_logger = AuditLogger()

# Initialize model cache in session state
if "t5_model" not in st.session_state:
    st.session_state["t5_model"] = None
if "t5_tokenizer" not in st.session_state:
    st.session_state["t5_tokenizer"] = None

# --- Tabs ---
upload_tab, summarize_tab, logs_tab = st.tabs(["Upload/Enter Note", "Summarize", "Logs"])

# --- Upload/Enter Note tab ---
with upload_tab:
    st.subheader("Enter or Upload Note")
    st.caption("Paste a synthetic note or upload a .txt file, then de-identify and index.")

    col_upload, col_text = st.columns([1, 2])
    with col_upload:
        file = st.file_uploader("Upload .txt file", type=["txt"])
    with col_text:
        note_text = st.text_area("Paste note text", height=200, placeholder="Paste clinical note text here...")

    col1, col2 = st.columns(2)
    with col1:
        deid_index_clicked = st.button("De-identify & Index", use_container_width=True)
    with col2:
        skip_index_clicked = st.button("Skip (already indexed)", use_container_width=True)

    if file and not note_text:
        note_text = file.read().decode("utf-8", errors="ignore")

    if deid_index_clicked and note_text:
        try:
            with st.spinner("De-identifying and indexing..."):
                from deid_pipeline import DeidPipeline
                pipeline = DeidPipeline()
                result = pipeline.run_on_text(text=note_text, note_id="temp_note")
                deid_text = result["masked_text"]
                st.success("De-identified.")
                st.text_area("De-identified preview", deid_text, height=160)

                from indexer import index_note
                # Use session-specific persist_dir
                note_id = index_note(
                    text=deid_text,
                    note_id=f"note_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    persist_dir=st.session_state["persist_dir"],
                    db_type="chroma",
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    collection="notes"
                )
                st.session_state["last_note_id"] = note_id
                st.session_state["last_deid_text"] = deid_text
                st.session_state["last_note_indexed"] = True
                st.success(f"✓ Indexed note_id: {note_id}")
                st.info(f"📁 Stored in: {st.session_state['persist_dir']}")

                # Audit log for indexing
                audit_logger.log_action(
                    user=st.session_state.get('username', 'anonymous'),
                    action="INDEX_NOTE",
                    resource=note_id,
                    additional_info={
                        "text_length": len(deid_text),
                        "persist_dir": st.session_state["persist_dir"]
                    }
                )

                # Audit log for de-identification
                audit_logger.log_action(
                    user=st.session_state.get('username', 'anonymous'),
                    action="DEID_PROCESS",
                    resource="temp_note",
                    additional_info={"original_length": len(note_text), "deid_length": len(deid_text)}
                )
                st.toast("Note indexed and de-identified!", icon="✅")
        except Exception as e:
            st.error(f"De-identification error: {e}")
            import traceback
            st.code(traceback.format_exc())
    elif skip_index_clicked and note_text:
        st.session_state["last_deid_text"] = note_text
        st.session_state["last_note_indexed"] = False
        st.info("Skipped indexing; text saved for summarization.")

        # Audit log for skipping
        audit_logger.log_action(
            user=st.session_state.get('username', 'anonymous'),
            action="SKIP_INDEX",
            resource="temp_note",
            additional_info={"text_length": len(note_text)}
        )

    if "last_deid_text" not in st.session_state:
        st.caption("Tip: click 'De-identify & Index' or 'Skip' to carry text into the Summarize tab.")
    else:
        st.write(f"✓ Note text ready: {len(st.session_state['last_deid_text'])} characters")
        st.write(f"Preview: {st.session_state['last_deid_text'][:100]}...")

# --- Summarize tab ---
with summarize_tab:
    st.subheader("Summarize")
    st.caption("Retrieves context and generates a structured clinical summary.")

    from rag_pipeline import load_embedder, load_chroma, load_faiss_langchain, retrieve
    from summarizer import make_t5, summarize_docs, validate_summary_quality
    
    # Environment detection
    import os
    IS_CLOUD = os.path.exists('/mount/src')
    if IS_CLOUD:
        st.info("🌐 Cloud Mode: Using optimized model (flan-t5-base)")
    
    # Clear ChromaDB system cache to avoid singleton conflicts
    try:
        import chromadb
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception as e:
        st.warning(f"Could not clear ChromaDB cache: {e}")

    # Show current vector store location
    st.info(f"📁 Using vector store: {st.session_state['persist_dir']}")

    source_choice = st.radio("Use source:", ["Last de-identified text", "Note ID"], horizontal=True)
    default_note_id = st.session_state.get("last_note_id", "")
    user_note_id = st.text_input("Note ID (optional)", value=str(default_note_id))

    # Add method selection
    method_choice = st.radio("Extraction method:", ["multistage", "singleshot"], horizontal=True, 
                            help="Multistage: Better quality, slower. Singleshot: Faster, may miss details.")

    generate_clicked = st.button("Generate Summary", type="primary", use_container_width=True)

    if generate_clicked:
        try:
            with st.spinner("Retrieving context..."):
                embed_model = "sentence-transformers/all-MiniLM-L6-v2"
                db_type = "chroma"
                persist_dir = st.session_state["persist_dir"]
                collection = "notes"
                top_k = 5

                # Cache vector database in session state to avoid recreating
                cache_key = f"vdb_{persist_dir}_{collection}"
                
                if cache_key not in st.session_state:
                    st.info("⏳ Loading vector database (first time)...")
                    _, embeddings = load_embedder(embed_model)
                    
                    # Clear cache before creating new instance
                    try:
                        SharedSystemClient.clear_system_cache()
                    except:
                        pass
                    
                    if db_type == "chroma":
                        vdb = load_chroma(persist_dir, collection, embeddings)
                    else:
                        vdb = load_faiss_langchain(persist_dir, embeddings)
                    
                    st.session_state[cache_key] = vdb
                    st.success("✓ Vector database loaded")
                else:
                    vdb = st.session_state[cache_key]
                    st.info("✓ Using cached vector database")

                # Use actual note content for retrieval
                if source_choice == "Note ID" and user_note_id:
                    query_text = user_note_id
                    st.info(f"🔍 Retrieving by Note ID: {user_note_id}")
                else:
                    deid_text = st.session_state.get("last_deid_text", "")
                    if not deid_text:
                        st.warning("No de-identified text available. Please use the Upload tab first.")
                        st.stop()
                    query_text = deid_text[:500]
                    st.info(f"🔍 Retrieving using note content ({len(deid_text)} chars)")

                docs = retrieve(vdb, query_text, top_k)
                
                if not docs:
                    st.error("⚠ No documents retrieved from vector database!")
                    st.warning("This usually means:")
                    st.write("• The vector database is empty")
                    st.write("• The note wasn't properly indexed")
                    st.write(f"• Check if files exist in: {persist_dir}")
                    
                    if st.button("🔄 Clear cache and retry"):
                        if cache_key in st.session_state:
                            del st.session_state[cache_key]
                        SharedSystemClient.clear_system_cache()
                        st.rerun()
                    st.stop()
                
                st.success(f"✓ Retrieved {len(docs)} document(s)")
                
                # Show preview of retrieved content
                with st.expander("View retrieved content"):
                    for i, doc in enumerate(docs, 1):
                        st.write(f"**Document {i}:**")
                        st.code(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

            with st.spinner("Generating summary... (this may take 1-2 minutes on CPU)"):
                # Cache model loading in session state
                if st.session_state["t5_model"] is None or st.session_state["t5_tokenizer"] is None:
                    st.info("⏳ Loading T5 model (first time only)...")
                    tokenizer, model = make_t5("google/flan-t5-base")
                    st.session_state["t5_tokenizer"] = tokenizer
                    st.session_state["t5_model"] = model
                else:
                    tokenizer = st.session_state["t5_tokenizer"]
                    model = st.session_state["t5_model"]
                    st.info("✓ Using cached model")

                # Generate summary
                summary = summarize_docs(tokenizer, model, docs, method=method_choice)
                
                # Store summary in session state
                summary_key = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state["last_summary"] = summary
                st.session_state["last_summary_key"] = summary_key

            # Validation
            original_text = st.session_state.get("last_deid_text", "")
            validation = validate_summary_quality(summary, original_text)
            
            # Display validation results
            status_color = {
                "GOOD": "🟢",
                "FAIR": "🟡", 
                "POOR": "🟠",
                "FAILED": "🔴"
            }
            
            st.success("✓ Summary generated successfully")
            
            # Show quality assessment in two columns
            col_status, col_score = st.columns([3, 1])
            with col_status:
                st.markdown(f"### {status_color.get(validation['status'], '⚪')} Quality Status: **{validation['status']}**")
            with col_score:
                st.metric("Quality Score", f"{validation['quality_score']}/100")
            
            # Display critical issues if any
            if validation['issues']:
                st.error("**❌ Critical Issues Detected:**")
                for issue in validation['issues']:
                    st.markdown(f"- {issue}")
                st.markdown("**Recommendation:** Review de-identification settings and retrieval quality")
            
            # Display warnings if any
            if validation['warnings']:
                st.warning("**⚠️ Quality Warnings:**")
                for warning in validation['warnings']:
                    st.markdown(f"- {warning}")
            
            # Show detailed quality metrics in expandable section
            with st.expander("📊 Detailed Quality Metrics"):
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("PHI Placeholders", validation['metrics']['total_placeholders'])
                with metric_col2:
                    st.metric("Empty Sections", validation['metrics']['empty_sections'])
                with metric_col3:
                    st.metric("Filled Sections", f"{validation['metrics']['filled_sections']}/7")
                with metric_col4:
                    st.metric("Total Length", f"{validation['metrics']['total_length']} chars")
            
            # Show warning banner if quality is poor
            if validation['status'] in ['POOR', 'FAILED']:
                st.warning("⚠️ **Quality Alert:** The summary below has significant quality issues. Review carefully before clinical use.")
            elif validation['status'] == 'FAIR':
                st.info("ℹ️ The summary has minor quality issues. Review the warnings above.")
            else:
                st.success("✅ Summary quality is acceptable.")
            
            # Display the summary
            st.text_area("Structured Summary", summary, height=400, key=f"summary_display_{summary_key}")
            st.download_button("Download .txt", data=summary, file_name=f"summary_{user_note_id or 'latest'}.txt")

            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Summary Length", f"{len(summary)} chars")
            with col2:
                st.metric("Documents Retrieved", len(docs))
            with col3:
                sections_filled = 7 - summary.count("None stated")
                st.metric("Sections Filled", f"{sections_filled}/7")

            # Audit log for summary generation with validation results
            audit_logger.log_action(
                user=st.session_state.get('username', 'anonymous'),
                action="GENERATE_SUMMARY",
                resource=user_note_id or "temp_note",
                additional_info={
                    "retrieved_docs": len(docs),
                    "method": method_choice,
                    "summary_length": len(summary),
                    "persist_dir": persist_dir,
                    "sections_filled": sections_filled,
                    "quality_status": validation['status'],
                    "quality_score": validation['quality_score'],
                    "validation_issues": len(validation['issues']),
                    "validation_warnings": len(validation['warnings']),
                    "phi_placeholders": validation['metrics']['total_placeholders']
                }
            )
            
        except ValueError as ve:
            if "already exists" in str(ve):
                st.error("❌ ChromaDB instance conflict detected!")
                st.warning("This happens when the vector database is accessed with different settings.")
                st.info("**Solution:** Click the button below to clear the cache and retry.")
                
                if st.button("🔄 Clear ChromaDB cache and retry", type="primary"):
                    try:
                        SharedSystemClient.clear_system_cache()
                    except:
                        pass
                    
                    keys_to_delete = [k for k in st.session_state.keys() if k.startswith("vdb_")]
                    for key in keys_to_delete:
                        del st.session_state[key]
                    
                    st.success("✓ Cache cleared! Click 'Generate Summary' again.")
                    st.rerun()
            else:
                st.error(f"❌ Error during summarization: {ve}")
                import traceback
                st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error during summarization: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Show last summary if available (when button not clicked)
    elif "last_summary" in st.session_state:
        st.info("Showing last generated summary:")
        st.text_area("Last Summary", st.session_state["last_summary"], height=400)
        st.download_button("Download Last Summary", 
                          data=st.session_state["last_summary"], 
                          file_name="last_summary.txt")

# --- Logs tab ---
with logs_tab:
    st.subheader("Logs")
    if st.session_state.get("role") != "admin":
        st.info("Admins only.")
    else:
        st.caption("Audit logs for all user actions.")
        
        # Audit log for viewing logs
        audit_logger.log_action(
            user=st.session_state.get('username', 'anonymous'),
            action="VIEW_LOGS",
            resource="app_audit.jsonl"
        )
        
        # Add log filtering
        col1, col2 = st.columns([3, 1])
        with col1:
            filter_action = st.selectbox("Filter by action:", 
                                        ["All", "INDEX_NOTE", "GENERATE_SUMMARY", "DEID_PROCESS", "VIEW_LOGS"])
        with col2:
            num_lines = st.number_input("Show last N lines:", min_value=10, max_value=500, value=50)
        
        try:
            import json
            with open("logs/app_audit.jsonl") as f:
                lines = f.readlines()[-num_lines:]
            
            st.write(f"Showing {len(lines)} most recent log entries:")
            
            for line in lines:
                try:
                    log_entry = json.loads(line.strip())
                    if filter_action == "All" or log_entry.get("action") == filter_action:
                        with st.expander(f"{log_entry.get('timestamp', 'N/A')} - {log_entry.get('action', 'N/A')}"):
                            st.json(log_entry)
                except json.JSONDecodeError:
                    st.code(line.strip())
        except FileNotFoundError:
            st.warning("No logs found yet. Logs will appear after you perform actions.")
