import json
import hashlib
from datetime import datetime
from pathlib import Path
import uuid
import pytz

class AuditLogger:
    def __init__(self, log_file_path="logs/app_audit.jsonl"):
        self.log_file = Path(log_file_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Create file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.touch()
    
    def _get_last_hash(self):
        """Read the last log entry and return its hash"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get('sha256_curr', '')
        except:
            pass
        return ''  # First entry has no previous hash
    
    def _compute_hash(self, log_entry):
        """Create a hash fingerprint of the log entry"""
        # Convert the log entry to a string and hash it
        entry_string = json.dumps(log_entry, sort_keys=True)
        return hashlib.sha256(entry_string.encode()).hexdigest()
    
    def log_action(self, user, action, resource, additional_info=None):
        """
        Main logging function - call this whenever a user does something
        
        Args:
            user: username (e.g., 'dr_smith')
            action: what they did (e.g., 'UPLOAD_NOTE', 'GENERATE_SUMMARY', 'VIEW_LOGS')
            resource: what they acted on (e.g., 'note_12345.txt', 'patient_record')
            additional_info: any extra details (dictionary)
        """
        # Get the hash of the previous log entry
        previous_hash = self._get_last_hash()
        
        # Generate unique IDs for tracing
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]  # Shorter ID for span
        
        # India timezone
        india = pytz.timezone('Asia/Kolkata')
        local_time = datetime.now(india).isoformat()
        
        # Create the new log entry
        log_entry = {
            "timestamp": local_time + "Z",
            "user": user,
            "action": action,
            "resource": resource,
            "sha256_prev": previous_hash,
            "additional_info": additional_info or {},
            
             # OpenTelemetry attributes
            "otel_trace_id": trace_id,
            "otel_span_id": span_id,
            "otel_service_name": "clinical-rag-app",
            "severity": "INFO"  # Can be DEBUG, INFO, WARN, ERROR
        }
        
        # Compute hash of THIS entry
        current_hash = self._compute_hash(log_entry)
        log_entry["sha256_curr"] = current_hash
        
        # Append to log file (append-only = cannot change old entries)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return log_entry
