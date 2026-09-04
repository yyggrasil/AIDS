"""
AIDS Raspberry Pi Edge Intrusion Detection Package.
Provides live packet capturing, flow aggregation, ML model inference,
and automated email alerting with anti-flood mechanisms.
"""

from .flow_aggregator import FlowAggregator, Flow, FEATURE_NAMES
from .email_alert import EmailAlertManager
from .detection_logger import DetectionLogger
from .rpi_detector import RPIDetector

__all__ = ["FlowAggregator", "Flow", "FEATURE_NAMES", "EmailAlertManager", "DetectionLogger", "RPIDetector"]



