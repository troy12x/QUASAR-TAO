# The MIT License (MIT)
# Copyright © 2024 HFA Research Team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import json
import hashlib
import hmac
import threading
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import os

import bittensor as bt


@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str
    timestamp: float
    event_type: str
    component: str
    actor: str  # miner_uid, validator_uid, system, etc.
    action: str
    resource: str
    details: Dict[str, Any] = field(default_factory=dict)
    integrity_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def compute_integrity_hash(self, secret_key: str) -> str:
        """Compute HMAC integrity hash for tamper detection"""
        # Create deterministic string representation
        data_str = f"{self.event_id}:{self.timestamp}:{self.event_type}:{self.component}:{self.actor}:{self.action}:{self.resource}:{json.dumps(self.details, sort_keys=True)}"
        
        # Compute HMAC-SHA256
        return hmac.new(
            secret_key.encode('utf-8'),
            data_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def verify_integrity(self, secret_key: str) -> bool:
        """Verify integrity hash"""
        if not self.integrity_hash:
            return False
        
        expected_hash = self.compute_integrity_hash(secret_key)
        return hmac.compare_digest(self.integrity_hash, expected_hash)


@dataclass
class AuditQuery:
    """Audit trail query parameters"""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    event_types: Optional[List[str]] = None
    components: Optional[List[str]] = None
    actors: Optional[List[str]] = None
    actions: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0
    
    def to_sql_where_clause(self) -> tuple[str, List[Any]]:
        """Convert query to SQL WHERE clause"""
        conditions = []
        params = []
        
        if self.start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(self.start_time)
        
        if self.end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(self.end_time)
        
        if self.event_types:
            placeholders = ",".join("?" * len(self.event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(self.event_types)
        
        if self.components:
            placeholders = ",".join("?" * len(self.components))
            conditions.append(f"component IN ({placeholders})")
            params.extend(self.components)
        
        if self.actors:
            placeholders = ",".join("?" * len(self.actors))
            conditions.append(f"actor IN ({placeholders})")
            params.extend(self.actors)
        
        if self.actions:
            placeholders = ",".join("?" * len(self.actions))
            conditions.append(f"action IN ({placeholders})")
            params.extend(self.actions)
        
        if self.resources:
            placeholders = ",".join("?" * len(self.resources))
            conditions.append(f"resource IN ({placeholders})")
            params.extend(self.resources)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params


class AuditTrailManager:
    """
    Comprehensive audit trail management system for the unified HFA-SimpleMind subnet.
    
    Features:
    - Tamper-evident audit logging with integrity verification
    - Structured event storage with efficient querying
    - Compliance reporting and audit trail export
    - Automated retention and archival policies
    - Real-time audit monitoring and alerting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize audit trail manager.
        
        Args:
            config: Configuration dictionary with audit settings
        """
        self.config = config or {}
        
        # Storage configuration
        self.db_path = self.config.get("db_path", "audit_trail.db")
        self.retention_days = self.config.get("retention_days", 365)
        self.archive_enabled = self.config.get("archive_enabled", True)
        self.archive_path = self.config.get("archive_path", "audit_archives/")
        
        # Security configuration
        self.secret_key = self.config.get("secret_key", self._generate_secret_key())
        self.integrity_verification_enabled = self.config.get("integrity_verification", True)
        
        # Performance configuration
        self.batch_size = self.config.get("batch_size", 100)
        self.flush_interval = self.config.get("flush_interval", 60)  # seconds
        
        # Event buffering
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = threading.Lock()
        self.flush_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_verified": 0,
            "integrity_failures": 0,
            "last_flush": None
        }
        
        # Initialize database
        self._initialize_database()
        
        # Start background flush thread
        self.start()
        
        bt.logging.info(f"📋 AuditTrailManager initialized (db: {self.db_path})")
    
    def _generate_secret_key(self) -> str:
        """Generate a secret key for integrity verification"""
        import secrets
        return secrets.token_hex(32)
    
    def _initialize_database(self):
        """Initialize SQLite database for audit storage"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    details TEXT NOT NULL,
                    integrity_hash TEXT,
                    created_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_component ON audit_events(component)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actor ON audit_events(actor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_action ON audit_events(action)")
            
            conn.commit()
    
    def start(self):
        """Start background processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.flush_thread.start()
        
        bt.logging.info("📋 Audit trail background processing started")
    
    def stop(self):
        """Stop background processing and flush remaining events"""
        self.is_running = False
        
        if self.flush_thread:
            self.flush_thread.join(timeout=10)
        
        # Flush any remaining events
        self._flush_events()
        
        bt.logging.info("📋 Audit trail background processing stopped")
    
    def _flush_loop(self):
        """Background thread for flushing events to database"""
        while self.is_running:
            try:
                self._flush_events()
                time.sleep(self.flush_interval)
            except Exception as e:
                bt.logging.error(f"Error in audit trail flush loop: {e}")
    
    def _flush_events(self):
        """Flush buffered events to database"""
        if not self.event_buffer:
            return
        
        with self.buffer_lock:
            events_to_flush = self.event_buffer.copy()
            self.event_buffer.clear()
        
        if not events_to_flush:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for event in events_to_flush:
                    conn.execute("""
                        INSERT OR REPLACE INTO audit_events 
                        (event_id, timestamp, event_type, component, actor, action, resource, details, integrity_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.timestamp,
                        event.event_type,
                        event.component,
                        event.actor,
                        event.action,
                        event.resource,
                        json.dumps(event.details),
                        event.integrity_hash
                    ))
                
                conn.commit()
            
            self.stats["events_logged"] += len(events_to_flush)
            self.stats["last_flush"] = time.time()
            
            bt.logging.debug(f"📋 Flushed {len(events_to_flush)} audit events to database")
            
        except Exception as e:
            bt.logging.error(f"Error flushing audit events: {e}")
            
            # Put events back in buffer for retry
            with self.buffer_lock:
                self.event_buffer.extend(events_to_flush)
    
    def log_event(self, event_type: str, component: str, actor: str, 
                  action: str, resource: str, details: Optional[Dict[str, Any]] = None):
        """Log an audit event"""
        event_id = f"{component}_{actor}_{int(time.time() * 1000000)}"
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            component=component,
            actor=actor,
            action=action,
            resource=resource,
            details=details or {}
        )
        
        # Compute integrity hash if enabled
        if self.integrity_verification_enabled:
            event.integrity_hash = event.compute_integrity_hash(self.secret_key)
        
        # Add to buffer
        with self.buffer_lock:
            self.event_buffer.append(event)
        
        bt.logging.debug(f"📋 Logged audit event: {event_type} - {component} - {action}")
    
    def log_miner_event(self, miner_uid: int, action: str, resource: str, 
                       details: Optional[Dict[str, Any]] = None):
        """Log miner-related audit event"""
        self.log_event(
            event_type="miner_activity",
            component="miner",
            actor=f"miner_{miner_uid}",
            action=action,
            resource=resource,
            details=details
        )
    
    def log_validator_event(self, validator_uid: int, action: str, resource: str,
                           details: Optional[Dict[str, Any]] = None):
        """Log validator-related audit event"""
        self.log_event(
            event_type="validator_activity",
            component="validator",
            actor=f"validator_{validator_uid}",
            action=action,
            resource=resource,
            details=details
        )
    
    def log_scoring_event(self, miner_uid: int, validator_uid: int, task_id: str,
                         score: float, details: Optional[Dict[str, Any]] = None):
        """Log scoring-related audit event"""
        scoring_details = {
            "miner_uid": miner_uid,
            "validator_uid": validator_uid,
            "task_id": task_id,
            "score": score
        }
        if details:
            scoring_details.update(details)
        
        self.log_event(
            event_type="scoring",
            component="scoring_harness",
            actor=f"validator_{validator_uid}",
            action="score_response",
            resource=f"miner_{miner_uid}",
            details=scoring_details
        )
    
    def log_system_event(self, action: str, resource: str, 
                        details: Optional[Dict[str, Any]] = None):
        """Log system-related audit event"""
        self.log_event(
            event_type="system",
            component="system",
            actor="system",
            action=action,
            resource=resource,
            details=details
        )
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        where_clause, params = query.to_sql_where_clause()
        
        sql = f"""
            SELECT event_id, timestamp, event_type, component, actor, action, resource, details, integrity_hash
            FROM audit_events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        
        params.extend([query.limit, query.offset])
        
        events = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql, params)
                
                for row in cursor.fetchall():
                    event = AuditEvent(
                        event_id=row[0],
                        timestamp=row[1],
                        event_type=row[2],
                        component=row[3],
                        actor=row[4],
                        action=row[5],
                        resource=row[6],
                        details=json.loads(row[7]) if row[7] else {},
                        integrity_hash=row[8]
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            bt.logging.error(f"Error querying audit events: {e}")
            return []
    
    def verify_event_integrity(self, event: AuditEvent) -> bool:
        """Verify the integrity of an audit event"""
        if not self.integrity_verification_enabled or not event.integrity_hash:
            return True
        
        is_valid = event.verify_integrity(self.secret_key)
        
        if is_valid:
            self.stats["events_verified"] += 1
        else:
            self.stats["integrity_failures"] += 1
            bt.logging.warning(f"📋 Integrity verification failed for event: {event.event_id}")
        
        return is_valid
    
    def verify_trail_integrity(self, start_time: Optional[float] = None, 
                              end_time: Optional[float] = None) -> Dict[str, Any]:
        """Verify integrity of audit trail within time range"""
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Process in batches
        )
        
        total_events = 0
        verified_events = 0
        failed_events = 0
        failed_event_ids = []
        
        offset = 0
        batch_size = 1000
        
        while True:
            query.offset = offset
            query.limit = batch_size
            
            events = self.query_events(query)
            if not events:
                break
            
            for event in events:
                total_events += 1
                if self.verify_event_integrity(event):
                    verified_events += 1
                else:
                    failed_events += 1
                    failed_event_ids.append(event.event_id)
            
            offset += batch_size
        
        integrity_report = {
            "total_events": total_events,
            "verified_events": verified_events,
            "failed_events": failed_events,
            "integrity_percentage": (verified_events / total_events * 100) if total_events > 0 else 100,
            "failed_event_ids": failed_event_ids,
            "verification_timestamp": time.time()
        }
        
        bt.logging.info(f"📋 Integrity verification completed: {integrity_report['integrity_percentage']:.2f}% verified")
        
        return integrity_report
    
    def export_audit_trail(self, query: AuditQuery, format: str = "json") -> str:
        """Export audit trail in specified format"""
        events = self.query_events(query)
        
        if format == "json":
            export_data = {
                "export_timestamp": time.time(),
                "query_parameters": asdict(query),
                "event_count": len(events),
                "events": [event.to_dict() for event in events]
            }
            return json.dumps(export_data, indent=2, default=str)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "event_id", "timestamp", "event_type", "component", 
                "actor", "action", "resource", "details", "integrity_hash"
            ])
            
            # Write events
            for event in events:
                writer.writerow([
                    event.event_id,
                    event.timestamp,
                    event.event_type,
                    event.component,
                    event.actor,
                    event.action,
                    event.resource,
                    json.dumps(event.details),
                    event.integrity_hash or ""
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_compliance_report(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Generate compliance report for specified time period"""
        query = AuditQuery(start_time=start_time, end_time=end_time, limit=100000)
        events = self.query_events(query)
        
        # Analyze events
        event_type_counts = {}
        component_counts = {}
        actor_counts = {}
        action_counts = {}
        
        for event in events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
            component_counts[event.component] = component_counts.get(event.component, 0) + 1
            actor_counts[event.actor] = actor_counts.get(event.actor, 0) + 1
            action_counts[event.action] = action_counts.get(event.action, 0) + 1
        
        # Verify integrity
        integrity_report = self.verify_trail_integrity(start_time, end_time)
        
        compliance_report = {
            "report_period": {
                "start_time": start_time,
                "end_time": end_time,
                "start_date": datetime.fromtimestamp(start_time).isoformat(),
                "end_date": datetime.fromtimestamp(end_time).isoformat()
            },
            "summary": {
                "total_events": len(events),
                "unique_event_types": len(event_type_counts),
                "unique_components": len(component_counts),
                "unique_actors": len(actor_counts),
                "unique_actions": len(action_counts)
            },
            "breakdown": {
                "event_types": event_type_counts,
                "components": component_counts,
                "actors": actor_counts,
                "actions": action_counts
            },
            "integrity": integrity_report,
            "generated_at": time.time()
        }
        
        return compliance_report
    
    def cleanup_old_events(self):
        """Clean up old audit events based on retention policy"""
        cutoff_time = time.time() - (self.retention_days * 24 * 3600)
        
        try:
            # Archive old events if enabled
            if self.archive_enabled:
                self._archive_old_events(cutoff_time)
            
            # Delete old events
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                conn.commit()
            
            if deleted_count > 0:
                bt.logging.info(f"📋 Cleaned up {deleted_count} old audit events")
            
        except Exception as e:
            bt.logging.error(f"Error cleaning up old audit events: {e}")
    
    def _archive_old_events(self, cutoff_time: float):
        """Archive old events before deletion"""
        try:
            # Create archive directory
            os.makedirs(self.archive_path, exist_ok=True)
            
            # Query old events
            query = AuditQuery(end_time=cutoff_time, limit=100000)
            old_events = self.query_events(query)
            
            if not old_events:
                return
            
            # Create archive file
            archive_filename = f"audit_archive_{datetime.fromtimestamp(cutoff_time).strftime('%Y%m%d')}.json"
            archive_filepath = os.path.join(self.archive_path, archive_filename)
            
            # Export to archive
            archive_data = {
                "archive_timestamp": time.time(),
                "cutoff_time": cutoff_time,
                "event_count": len(old_events),
                "events": [event.to_dict() for event in old_events]
            }
            
            with open(archive_filepath, 'w') as f:
                json.dump(archive_data, f, indent=2, default=str)
            
            bt.logging.info(f"📋 Archived {len(old_events)} old audit events to {archive_filepath}")
            
        except Exception as e:
            bt.logging.error(f"Error archiving old audit events: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total events
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
                total_events = cursor.fetchone()[0]
                
                # Events by type
                cursor = conn.execute("SELECT event_type, COUNT(*) FROM audit_events GROUP BY event_type")
                events_by_type = dict(cursor.fetchall())
                
                # Events by component
                cursor = conn.execute("SELECT component, COUNT(*) FROM audit_events GROUP BY component")
                events_by_component = dict(cursor.fetchall())
                
                # Recent activity (last 24 hours)
                last_24h = time.time() - (24 * 3600)
                cursor = conn.execute("SELECT COUNT(*) FROM audit_events WHERE timestamp > ?", (last_24h,))
                recent_events = cursor.fetchone()[0]
            
            stats = {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "events_by_component": events_by_component,
                "recent_events_24h": recent_events,
                "buffer_size": len(self.event_buffer),
                "runtime_stats": self.stats.copy()
            }
            
            return stats
            
        except Exception as e:
            bt.logging.error(f"Error getting audit trail statistics: {e}")
            return {"error": str(e)}


def create_audit_trail_manager(config: Optional[Dict[str, Any]] = None) -> AuditTrailManager:
    """Factory function to create audit trail manager"""
    return AuditTrailManager(config)