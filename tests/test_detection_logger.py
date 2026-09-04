"""
Unit Tests for DetectionLogger and Intrusion Logging System.
Tests JSONL, CSV, Text formatting, log rotation, concurrent logging,
and integration with RPIDetector.
"""

import os
import tempfile
import shutil
import json
import csv
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from raspberry_pi.detection_logger import DetectionLogger, CSV_FIELDNAMES
from raspberry_pi.flow_aggregator import Flow
from raspberry_pi.rpi_detector import RPIDetector
from scapy.layers.inet import IP, TCP


class TestDetectionLogger(unittest.TestCase):
    """Test suite for the DetectionLogger component."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _sample_flow_and_result(self, is_attack=True, attack_type="DoS", prob=0.985):
        flow = Flow("192.168.1.150", 55432, "192.168.1.1", 80, "6", start_time=100.0)
        flow.add_packet(is_fwd=True, pkt_len=60, header_len=20, timestamp=100.0, tcp_flags={'SYN': True})
        flow.add_packet(is_fwd=False, pkt_len=60, header_len=20, timestamp=100.01, tcp_flags={'SYN': True, 'ACK': True})
        flow.add_packet(is_fwd=True, pkt_len=1200, header_len=20, timestamp=100.05, tcp_flags={'ACK': True, 'PSH': True}, payload_len=1160)
        flow.add_packet(is_fwd=True, pkt_len=60, header_len=20, timestamp=100.10, tcp_flags={'FIN': True, 'ACK': True})

        res = {
            'is_attack': is_attack,
            'attack_type': attack_type,
            'probability': prob,
            'all_probabilities': {attack_type: prob, 'Benign': 1.0 - prob},
            'flow_summary': flow.get_summary()
        }
        return flow, res

    def test_jsonl_logging(self):
        """Verifies JSON Lines format and schema of detected intrusions."""
        log_path = os.path.join(self.test_dir, "detections.jsonl")
        logger = DetectionLogger(log_file=log_path, log_format="jsonl", enabled=True)

        flow, res = self._sample_flow_and_result(is_attack=True, attack_type="DoS", prob=0.975)
        logged = logger.log_detection(res, flow=flow, interface="eth0")
        self.assertTrue(logged)
        self.assertEqual(logger.total_logged, 1)
        self.assertEqual(logger.total_attacks_logged, 1)

        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        entry = json.loads(lines[0])
        self.assertEqual(entry["event"], "INTRUSION_DETECTED")
        self.assertEqual(entry["attack_type"], "DoS")
        self.assertTrue(entry["is_attack"])
        self.assertAlmostEqual(entry["confidence"], 0.975, places=3)
        self.assertEqual(entry["source"]["ip"], "192.168.1.150")
        self.assertEqual(entry["source"]["port"], 55432)
        self.assertEqual(entry["destination"]["ip"], "192.168.1.1")
        self.assertEqual(entry["destination"]["port"], 80)
        self.assertEqual(entry["protocol"], "TCP")
        self.assertEqual(entry["interface"], "eth0")
        self.assertEqual(entry["flow_metrics"]["total_packets"], 4)
        self.assertIn("iptables", entry["mitigation_rule"])

        recent = logger.get_recent_logs(limit=5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["attack_type"], "DoS")

    def test_csv_logging(self):
        """Verifies CSV format, header row, and content mapping."""
        log_path = os.path.join(self.test_dir, "detections.csv")
        logger = DetectionLogger(log_file=log_path, log_format="csv", enabled=True)

        flow, res = self._sample_flow_and_result(is_attack=True, attack_type="Exploits", prob=0.92)
        logged = logger.log_detection(res, flow=flow, interface="wlan0")
        self.assertTrue(logged)

        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["event"], "INTRUSION_DETECTED")
        self.assertEqual(row["attack_type"], "Exploits")
        self.assertEqual(row["src_ip"], "192.168.1.150")
        self.assertEqual(row["dst_ip"], "192.168.1.1")
        self.assertEqual(row["protocol"], "TCP")
        self.assertEqual(row["interface"], "wlan0")
        self.assertEqual(row["total_packets"], "4")
        self.assertIn("iptables", row["mitigation_rule"])

    def test_text_logging(self):
        """Verifies human-readable text format."""
        log_path = os.path.join(self.test_dir, "detections.log")
        logger = DetectionLogger(log_file=log_path, log_format="text", enabled=True)

        flow, res = self._sample_flow_and_result(is_attack=True, attack_type="Reconnaissance", prob=0.88)
        logged = logger.log_detection(res, flow=flow, interface="eth0")
        self.assertTrue(logged)

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("INTRUSION_DETECTED", content)
        self.assertIn("Reconnaissance", content)
        self.assertIn("88.00%", content)
        self.assertIn("192.168.1.150:55432", content)
        self.assertIn("sudo iptables", content)

    def test_both_formats_logging(self):
        """Verifies simultaneous logging to JSONL and CSV."""
        log_path = os.path.join(self.test_dir, "detections.jsonl")
        logger = DetectionLogger(log_file=log_path, log_format="both", enabled=True)

        flow, res = self._sample_flow_and_result(is_attack=True, attack_type="DoS", prob=0.99)
        logged = logger.log_detection(res, flow=flow, interface="eth0")
        self.assertTrue(logged)

        csv_path = os.path.join(self.test_dir, "detections.csv")
        self.assertTrue(os.path.exists(log_path))
        self.assertTrue(os.path.exists(csv_path))

    def test_log_all_flows_filtering(self):
        """Verifies that benign traffic is suppressed by default, but logged if log_all_flows=True."""
        log_path = os.path.join(self.test_dir, "detections.jsonl")
        # 1. Default: only attacks
        logger = DetectionLogger(log_file=log_path, log_all_flows=False, enabled=True)
        _, benign_res = self._sample_flow_and_result(is_attack=False, attack_type="Benign", prob=0.05)
        logged = logger.log_detection(benign_res)
        self.assertFalse(logged)
        self.assertEqual(logger.total_logged, 0)

        # 2. log_all_flows = True
        logger_all = DetectionLogger(log_file=log_path, log_all_flows=True, enabled=True)
        logged_all = logger_all.log_detection(benign_res)
        self.assertTrue(logged_all)
        self.assertEqual(logger_all.total_logged, 1)
        self.assertEqual(logger_all.total_benign_logged, 1)

    def test_log_rotation(self):
        """Verifies automatic file rollover when exceeding max_bytes limit."""
        log_path = os.path.join(self.test_dir, "detections.jsonl")
        # Very small max_bytes to force rotation after a few entries
        logger = DetectionLogger(
            log_file=log_path,
            log_format="jsonl",
            max_bytes=250,
            backup_count=3,
            enabled=True
        )

        flow, res = self._sample_flow_and_result(is_attack=True, attack_type="DoS", prob=0.99)
        for _ in range(5):
            logger.log_detection(res, flow=flow)

        # Backup file .1 should have been created
        backup_path = f"{log_path}.1"
        self.assertTrue(os.path.exists(backup_path), "Arquivo rotacionado .1 não foi criado.")

    def test_thread_safety(self):
        """Verifies thread-safe concurrent logging from multiple worker threads."""
        log_path = os.path.join(self.test_dir, "concurrent.jsonl")
        logger = DetectionLogger(log_file=log_path, log_format="jsonl", enabled=True)

        num_threads = 5
        logs_per_thread = 20

        def worker(thread_id):
            for i in range(logs_per_thread):
                _, res = self._sample_flow_and_result(
                    is_attack=True,
                    attack_type=f"Attack_{thread_id}",
                    prob=0.90 + (i * 0.001)
                )
                logger.log_detection(res)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_total = num_threads * logs_per_thread
        self.assertEqual(logger.total_logged, expected_total)

        with open(log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        self.assertEqual(len(lines), expected_total)

    def test_stats_reporting(self):
        """Verifies get_stats output."""
        log_path = os.path.join(self.test_dir, "stats_test.jsonl")
        logger = DetectionLogger(log_file=log_path, enabled=True)
        stats = logger.get_stats()
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["total_logged"], 0)
        self.assertEqual(stats["log_file"], log_path)


class TestRPIDetectorLoggingIntegration(unittest.TestCase):
    """Tests integration between RPIDetector and DetectionLogger."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_detector_writes_to_detection_logger(self, mock_resolve, mock_load):
        """Verifies that an attack detected by RPIDetector is recorded by DetectionLogger."""
        # Mock ML Pipeline predicting Malicious (class 1 prob = 0.95)
        mock_clf = MagicMock()
        mock_clf.classes_ = np.array([0, 1])
        mock_pipeline = MagicMock()
        mock_pipeline.named_steps = {'classifier': mock_clf}
        mock_pipeline.predict_proba.return_value = np.array([[0.05, 0.95]])
        mock_load.return_value = mock_pipeline

        log_path = os.path.join(self.test_dir, "detector_detections.jsonl")
        detection_logger = DetectionLogger(
            log_file=log_path,
            log_format="jsonl",
            enabled=True
        )

        email_mgr = MagicMock()
        detector = RPIDetector(
            mode="binary",
            dry_run=False,
            email_manager=email_mgr,
            detection_logger=detection_logger
        )

        t0 = time.time()
        p1 = IP(src="192.168.10.50", dst="192.168.10.1")/TCP(sport=44444, dport=80, flags="S")
        p1.time = t0
        detector.process_packet(p1)

        p2 = IP(src="192.168.10.50", dst="192.168.10.1")/TCP(sport=44444, dport=80, flags="FA")
        p2.time = t0 + 0.05
        detector.process_packet(p2)

        # Verify email was called
        email_mgr.send_alert.assert_called()

        # Verify detection log was written
        self.assertEqual(detection_logger.total_logged, 1)
        self.assertEqual(detection_logger.total_attacks_logged, 1)
        self.assertTrue(os.path.exists(log_path))

        with open(log_path, "r", encoding="utf-8") as f:
            record = json.loads(f.readline().strip())

        self.assertEqual(record["event"], "INTRUSION_DETECTED")
        self.assertEqual(record["source"]["ip"], "192.168.10.50")
        self.assertEqual(record["destination"]["ip"], "192.168.10.1")
        self.assertAlmostEqual(record["confidence"], 0.95, places=2)

        # Verify get_stats reports detections_logged
        stats = detector.get_stats()
        self.assertEqual(stats["detections_logged"], 1)
        self.assertEqual(stats["detection_log_file"], log_path)


if __name__ == '__main__':
    unittest.main()
