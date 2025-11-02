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
import smtplib
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import requests

import bittensor as bt
from .health_monitor import Alert, AlertSeverity


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 15
    enabled: bool = True
    last_triggered: Optional[float] = None
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if alert should trigger"""
        if not self.enabled:
            return False
        
        # Check cooldown
        if (self.last_triggered and 
            time.time() - self.last_triggered < self.cooldown_minutes * 60):
            return False
        
        return self.condition(data)
    
    def trigger(self) -> str:
        """Trigger the alert and return formatted message"""
        self.last_triggered = time.time()
        return self.message_template


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    channel_type: str  # email, webhook, slack, discord
    config: Dict[str, Any]
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.WARNING
    
    def should_notify(self, alert: Alert) -> bool:
        """Check if this channel should be notified for the alert"""
        if not self.enabled:
            return False
        
        # Check severity threshold
        severity_levels = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2
        }
        
        return severity_levels[alert.severity] >= severity_levels[self.min_severity]


class AlertManager:
    """
    Comprehensive alerting system for the unified HFA-SimpleMind subnet.
    
    Features:
    - Rule-based alerting with customizable conditions
    - Multiple notification channels (email, webhook, Slack, Discord)
    - Alert deduplication and cooldown periods
    - Alert escalation and acknowledgment
    - Integration with health monitoring system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration dictionary with alerting settings
        """
        self.config = config or {}
        
        # Alert rules and channels
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.acknowledged_alerts: Set[str] = set()
        
        # Processing settings
        self.max_alerts_per_minute = self.config.get("max_alerts_per_minute", 10)
        self.alert_retention_hours = self.config.get("alert_retention_hours", 168)  # 1 week
        
        # Rate limiting
        self.alert_timestamps: List[float] = []
        
        # Initialize default alert rules
        self._register_default_alert_rules()
        
        # Initialize notification channels from config
        self._initialize_notification_channels()
        
        bt.logging.info("🚨 AlertManager initialized")
    
    def _register_default_alert_rules(self):
        """Register default alert rules"""
        
        # High CPU usage alert
        self.register_alert_rule(
            "high_cpu_usage",
            lambda data: data.get("cpu_percent", 0) > 90,
            AlertSeverity.CRITICAL,
            "🔥 CRITICAL: High CPU usage detected: {cpu_percent:.1f}%",
            cooldown_minutes=5
        )
        
        # High memory usage alert
        self.register_alert_rule(
            "high_memory_usage",
            lambda data: data.get("memory_percent", 0) > 95,
            AlertSeverity.CRITICAL,
            "🔥 CRITICAL: High memory usage detected: {memory_percent:.1f}%",
            cooldown_minutes=5
        )
        
        # Low disk space alert
        self.register_alert_rule(
            "low_disk_space",
            lambda data: data.get("disk_percent", 0) > 95,
            AlertSeverity.CRITICAL,
            "💾 CRITICAL: Low disk space: {disk_percent:.1f}% used",
            cooldown_minutes=30
        )
        
        # GPU overheating alert
        self.register_alert_rule(
            "gpu_overheating",
            lambda data: any(gpu.get("temperature", 0) > 85 for gpu in data.get("gpu_info", [])),
            AlertSeverity.CRITICAL,
            "🌡️ CRITICAL: GPU overheating detected",
            cooldown_minutes=10
        )
        
        # High monoculture risk alert
        self.register_alert_rule(
            "high_monoculture_risk",
            lambda data: data.get("monoculture_risk_level") == "high",
            AlertSeverity.WARNING,
            "🎯 WARNING: High monoculture risk detected in subnet",
            cooldown_minutes=60
        )
        
        # Low benchmark success rate alert
        self.register_alert_rule(
            "low_benchmark_success",
            lambda data: data.get("benchmark_success_rate", 1.0) < 0.3,
            AlertSeverity.WARNING,
            "📊 WARNING: Low benchmark success rate: {benchmark_success_rate:.1%}",
            cooldown_minutes=30
        )
        
        # Miner connectivity issues
        self.register_alert_rule(
            "low_miner_count",
            lambda data: data.get("active_miners", 0) < 2,
            AlertSeverity.WARNING,
            "⛏️ WARNING: Low number of active miners: {active_miners}",
            cooldown_minutes=15
        )
    
    def _initialize_notification_channels(self):
        """Initialize notification channels from configuration"""
        channels_config = self.config.get("notification_channels", {})
        
        for channel_name, channel_config in channels_config.items():
            try:
                channel = NotificationChannel(
                    name=channel_name,
                    channel_type=channel_config["type"],
                    config=channel_config.get("config", {}),
                    enabled=channel_config.get("enabled", True),
                    min_severity=AlertSeverity(channel_config.get("min_severity", "warning"))
                )
                
                self.notification_channels[channel_name] = channel
                bt.logging.info(f"🚨 Initialized notification channel: {channel_name} ({channel.channel_type})")
                
            except Exception as e:
                bt.logging.error(f"Failed to initialize notification channel {channel_name}: {e}")
    
    def register_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                          severity: AlertSeverity, message_template: str, 
                          cooldown_minutes: int = 15):
        """Register a new alert rule"""
        rule = AlertRule(
            name=name,
            condition=condition,
            severity=severity,
            message_template=message_template,
            cooldown_minutes=cooldown_minutes
        )
        
        self.alert_rules[name] = rule
        bt.logging.info(f"🚨 Registered alert rule: {name}")
    
    def add_notification_channel(self, name: str, channel_type: str, 
                               config: Dict[str, Any], min_severity: AlertSeverity = AlertSeverity.WARNING):
        """Add a notification channel"""
        channel = NotificationChannel(
            name=name,
            channel_type=channel_type,
            config=config,
            min_severity=min_severity
        )
        
        self.notification_channels[name] = channel
        bt.logging.info(f"🚨 Added notification channel: {name} ({channel_type})")
    
    def process_data(self, data: Dict[str, Any]):
        """Process data and check for alert conditions"""
        if not self._check_rate_limit():
            bt.logging.warning("Alert rate limit exceeded, skipping processing")
            return
        
        triggered_alerts = []
        
        # Check each alert rule
        for rule_name, rule in self.alert_rules.items():
            try:
                if rule.should_trigger(data):
                    # Create alert
                    alert_id = f"{rule_name}_{int(time.time())}"
                    message = rule.trigger().format(**data)
                    
                    alert = Alert(
                        id=alert_id,
                        severity=rule.severity,
                        title=f"Alert: {rule_name}",
                        description=message,
                        component="alert_manager",
                        metadata={"rule_name": rule_name, "data": data}
                    )
                    
                    triggered_alerts.append(alert)
                    
            except Exception as e:
                bt.logging.error(f"Error processing alert rule {rule_name}: {e}")
        
        # Send notifications for triggered alerts
        for alert in triggered_alerts:
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: Alert):
        """Handle a triggered alert"""
        # Add to active alerts and history
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        bt.logging.warning(f"🚨 Alert triggered: {alert.title} - {alert.description}")
        
        # Send notifications
        self._send_notifications(alert)
        
        # Clean up old alerts
        self._cleanup_old_alerts()
    
    def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        for channel_name, channel in self.notification_channels.items():
            if channel.should_notify(alert):
                try:
                    self._send_notification(channel, alert)
                except Exception as e:
                    bt.logging.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _send_notification(self, channel: NotificationChannel, alert: Alert):
        """Send notification through specific channel"""
        if channel.channel_type == "email":
            self._send_email_notification(channel, alert)
        elif channel.channel_type == "webhook":
            self._send_webhook_notification(channel, alert)
        elif channel.channel_type == "slack":
            self._send_slack_notification(channel, alert)
        elif channel.channel_type == "discord":
            self._send_discord_notification(channel, alert)
        else:
            bt.logging.warning(f"Unknown notification channel type: {channel.channel_type}")
    
    def _send_email_notification(self, channel: NotificationChannel, alert: Alert):
        """Send email notification"""
        config = channel.config
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Email body
        body = f"""
Alert Details:
- Title: {alert.title}
- Severity: {alert.severity.value.upper()}
- Component: {alert.component}
- Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}
- Description: {alert.description}

Metadata: {json.dumps(alert.metadata, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587))
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
        server.quit()
        
        bt.logging.info(f"📧 Email notification sent for alert: {alert.title}")
    
    def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert):
        """Send webhook notification"""
        config = channel.config
        
        payload = {
            "alert": alert.to_dict(),
            "timestamp": time.time()
        }
        
        response = requests.post(
            config['url'],
            json=payload,
            headers=config.get('headers', {}),
            timeout=config.get('timeout', 10)
        )
        
        response.raise_for_status()
        bt.logging.info(f"🔗 Webhook notification sent for alert: {alert.title}")
    
    def _send_slack_notification(self, channel: NotificationChannel, alert: Alert):
        """Send Slack notification"""
        config = channel.config
        
        # Determine color based on severity
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "text": f"Alert: {alert.title}",
            "attachments": [{
                "color": color_map.get(alert.severity, "warning"),
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Component", "value": alert.component, "short": True},
                    {"title": "Time", "value": datetime.fromtimestamp(alert.timestamp).isoformat(), "short": True},
                    {"title": "Description", "value": alert.description, "short": False}
                ]
            }]
        }
        
        response = requests.post(
            config['webhook_url'],
            json=payload,
            timeout=10
        )
        
        response.raise_for_status()
        bt.logging.info(f"💬 Slack notification sent for alert: {alert.title}")
    
    def _send_discord_notification(self, channel: NotificationChannel, alert: Alert):
        """Send Discord notification"""
        config = channel.config
        
        # Determine color based on severity
        color_map = {
            AlertSeverity.INFO: 0x00ff00,      # Green
            AlertSeverity.WARNING: 0xffff00,   # Yellow
            AlertSeverity.CRITICAL: 0xff0000   # Red
        }
        
        embed = {
            "title": alert.title,
            "description": alert.description,
            "color": color_map.get(alert.severity, 0xffff00),
            "fields": [
                {"name": "Severity", "value": alert.severity.value.upper(), "inline": True},
                {"name": "Component", "value": alert.component, "inline": True},
                {"name": "Time", "value": datetime.fromtimestamp(alert.timestamp).isoformat(), "inline": False}
            ],
            "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat()
        }
        
        payload = {"embeds": [embed]}
        
        response = requests.post(
            config['webhook_url'],
            json=payload,
            timeout=10
        )
        
        response.raise_for_status()
        bt.logging.info(f"🎮 Discord notification sent for alert: {alert.title}")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.alert_timestamps = [
            ts for ts in self.alert_timestamps
            if current_time - ts < 60
        ]
        
        # Check if we're under the limit
        if len(self.alert_timestamps) >= self.max_alerts_per_minute:
            return False
        
        # Add current timestamp
        self.alert_timestamps.append(current_time)
        return True
    
    def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        cutoff_time = time.time() - (self.alert_retention_hours * 3600)
        
        # Remove old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        # Remove resolved alerts from active alerts
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and alert.resolved_timestamp and 
            time.time() - alert.resolved_timestamp > 3600  # 1 hour after resolution
        ]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.acknowledged_alerts.add(alert_id)
            alert = self.active_alerts[alert_id]
            alert.metadata["acknowledged_by"] = user
            alert.metadata["acknowledged_at"] = time.time()
            
            bt.logging.info(f"✅ Alert acknowledged by {user}: {alert.title}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_timestamp = time.time()
            alert.metadata["resolved_by"] = user
            
            bt.logging.info(f"✅ Alert resolved by {user}: {alert.title}")
            return True
        
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert.to_dict() for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        now = time.time()
        last_24h = now - (24 * 3600)
        last_7d = now - (7 * 24 * 3600)
        
        # Count alerts by time period
        alerts_24h = [a for a in self.alert_history if a.timestamp > last_24h]
        alerts_7d = [a for a in self.alert_history if a.timestamp > last_7d]
        
        # Count by severity
        severity_counts_24h = {}
        severity_counts_7d = {}
        
        for alert in alerts_24h:
            severity = alert.severity.value
            severity_counts_24h[severity] = severity_counts_24h.get(severity, 0) + 1
        
        for alert in alerts_7d:
            severity = alert.severity.value
            severity_counts_7d[severity] = severity_counts_7d.get(severity, 0) + 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "acknowledged_alerts": len(self.acknowledged_alerts),
            "alerts_last_24h": len(alerts_24h),
            "alerts_last_7d": len(alerts_7d),
            "severity_breakdown_24h": severity_counts_24h,
            "severity_breakdown_7d": severity_counts_7d,
            "alert_rules_count": len(self.alert_rules),
            "notification_channels_count": len(self.notification_channels)
        }


def create_alert_manager(config: Optional[Dict[str, Any]] = None) -> AlertManager:
    """Factory function to create alert manager"""
    return AlertManager(config)