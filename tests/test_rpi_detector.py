"""
Unit Tests for Raspberry Pi Edge Intrusion Detection and Email Alert System.
Tests flow aggregation, feature extraction, ML inference, email formatting,
and anti-flood cooldown mechanisms using Scapy & smtplib mocks.
"""

import unittest
from unittest.mock import MagicMock, patch
import time
import os
import pandas as pd
import numpy as np

from scapy.layers.inet import IP, TCP, UDP
from raspberry_pi.flow_aggregator import FlowAggregator, Flow, FEATURE_NAMES
from raspberry_pi.email_alert import EmailAlertManager
from raspberry_pi.rpi_detector import RPIDetector


class TestFlowAggregation(unittest.TestCase):
    """Tests for packet aggregation and feature extraction."""

    def setUp(self):
        self.aggregator = FlowAggregator(inactivity_timeout=2.0, active_timeout=10.0, max_flows=100)

    def test_bidirectional_tcp_flow(self):
        """Verifies proper tracking of a bidirectional TCP handshake and data exchange."""
        t0 = 1000.0
        # 1. SYN (Client -> Server)
        p1 = IP(src="192.168.1.50", dst="10.0.0.1")/TCP(sport=50000, dport=80, flags="S", window=65535)
        p1.time = t0
        self.aggregator.process_packet(p1)

        # 2. SYN-ACK (Server -> Client)
        p2 = IP(src="10.0.0.1", dst="192.168.1.50")/TCP(sport=80, dport=50000, flags="SA", window=32768)
        p2.time = t0 + 0.01
        self.aggregator.process_packet(p2)

        # 3. ACK (Client -> Server)
        p3 = IP(src="192.168.1.50", dst="10.0.0.1")/TCP(sport=50000, dport=80, flags="A")
        p3.time = t0 + 0.02
        self.aggregator.process_packet(p3)

        # 4. PSH-ACK with data (Client -> Server)
        p4 = IP(src="192.168.1.50", dst="10.0.0.1")/TCP(sport=50000, dport=80, flags="PA")/b"GET / HTTP/1.1\r\n\r\n"
        p4.time = t0 + 0.05
        self.aggregator.process_packet(p4)

        # 5. FIN-ACK (Client -> Server)
        p5 = IP(src="192.168.1.50", dst="10.0.0.1")/TCP(sport=50000, dport=80, flags="FA")
        p5.time = t0 + 0.10
        self.aggregator.process_packet(p5)

        # Flush completed/expired flow
        expired = self.aggregator.flush_expired(current_time=t0 + 0.15)
        self.assertEqual(len(expired), 1)

        flow = expired[0]
        self.assertEqual(flow.src_ip, "192.168.1.50")
        self.assertEqual(flow.dst_ip, "10.0.0.1")
        self.assertEqual(flow.src_port, 50000)
        self.assertEqual(flow.dst_port, 80)
        self.assertEqual(flow.protocol, "6")
        self.assertEqual(len(flow.fwd_packet_lengths), 4)
        self.assertEqual(len(flow.bwd_packet_lengths), 1)
        self.assertEqual(flow.fwd_init_win, 65535)
        self.assertEqual(flow.bwd_init_win, 32768)

    def test_feature_extraction_schema(self):
        """Verifies extracted features exactly match the 70 CICFlowMeter features."""
        flow = Flow(
            src_ip="192.168.1.10",
            src_port=12345,
            dst_ip="10.0.0.5",
            dst_port=443,
            protocol="6",
            start_time=100.0
        )
        flow.add_packet(is_fwd=True, pkt_len=60, header_len=20, timestamp=100.0, tcp_flags={'SYN': True}, tcp_window=1024)
        flow.add_packet(is_fwd=False, pkt_len=60, header_len=20, timestamp=100.02, tcp_flags={'SYN': True, 'ACK': True}, tcp_window=2048)
        flow.add_packet(is_fwd=True, pkt_len=1500, header_len=20, timestamp=100.05, tcp_flags={'ACK': True, 'PSH': True}, payload_len=1460)

        feats = flow.extract_features()
        self.assertEqual(len(feats), 70)
        for expected_col in FEATURE_NAMES:
            self.assertIn(expected_col, feats, f"Feature {expected_col} missing in extracted features")

        df = flow.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (1, 70))
        self.assertEqual(list(df.columns), FEATURE_NAMES)

    def test_inactivity_timeout_and_eviction(self):
        """Verifies flow expiration after inactivity timeout and memory eviction."""
        agg = FlowAggregator(inactivity_timeout=5.0, max_flows=2)
        p1 = IP(src="1.1.1.1", dst="2.2.2.2")/TCP(sport=100, dport=200, flags="S")
        p1.time = 10.0
        agg.process_packet(p1)

        # Before timeout
        expired = agg.flush_expired(current_time=12.0)
        self.assertEqual(len(expired), 0)

        # After timeout
        expired = agg.flush_expired(current_time=16.0)
        self.assertEqual(len(expired), 1)

        # Test capacity eviction
        p2 = IP(src="3.3.3.3", dst="4.4.4.4")/TCP(sport=300, dport=400, flags="S")
        p2.time = 20.0
        agg.process_packet(p2)

        p3 = IP(src="5.5.5.5", dst="6.6.6.6")/TCP(sport=500, dport=600, flags="S")
        p3.time = 21.0
        agg.process_packet(p3)

        p4 = IP(src="7.7.7.7", dst="8.8.8.8")/TCP(sport=700, dport=800, flags="S")
        p4.time = 22.0
        agg.process_packet(p4)

        # Capacity was 2, so oldest flow should have been evicted
        self.assertLessEqual(len(agg.flows), 2)

    def test_udp_flow_aggregation_and_categorization(self):
        """Verifies UDP packet parsing, flow aggregation, 1-pkt rate protection, and model categorization."""
        p_udp = IP(src="192.168.1.100", dst="8.8.8.8")/UDP(sport=53000, dport=53)/(b"X" * 64)
        p_udp.time = 100.0
        flow = self.aggregator.process_packet(p_udp)

        self.assertIsNotNone(flow)
        self.assertEqual(flow.protocol, "17")
        self.assertEqual(flow.src_port, 53000)
        self.assertEqual(flow.dst_port, 53)
        self.assertEqual(flow.fwd_seg_size_min, 8)
        self.assertEqual(flow.fin_flags, 0)
        self.assertEqual(flow.syn_flags, 0)
        self.assertEqual(flow.rst_flags, 0)
        self.assertEqual(flow.ack_flags, 0)
        self.assertEqual(flow.psh_flags, 0)

        # Verify rate protection on single UDP packet (prevents false positive flood detection)
        feats = flow.extract_features()
        self.assertEqual(feats['Flow Packets/s'], 0.0)
        self.assertEqual(feats['Flow Bytes/s'], 0.0)
        self.assertEqual(feats['Fwd Packets/s'], 0.0)
        self.assertEqual(feats['Bwd Packets/s'], 0.0)
        self.assertEqual(feats['Protocol'], "17")

        # Verify DataFrame categorization for model pipeline
        df = flow.to_dataframe()
        self.assertEqual(df['Protocol'].iloc[0], "17")
        self.assertTrue(pd.api.types.is_string_dtype(df['Protocol']) or isinstance(df['Protocol'].iloc[0], str))
        self.assertEqual(flow.get_summary()['protocol_name'], "UDP")

    def test_tcp_single_packet_rate_preservation(self):
        """Verifies that TCP SYN probes retain packet rate signals while conforming to 0 payload byte rate."""
        p_tcp = IP(src="192.168.1.100", dst="10.0.0.1")/TCP(sport=54321, dport=80, flags="S")
        p_tcp.time = 100.0
        flow = self.aggregator.process_packet(p_tcp)

        self.assertIsNotNone(flow)
        self.assertEqual(flow.protocol, "6")
        feats = flow.extract_features()
        self.assertGreater(feats['Flow Packets/s'], 0.0)
        # In CICFlowMeter, flows without payload bytes have 0.0 Flow Bytes/s
        self.assertEqual(feats['Flow Bytes/s'], 0.0)
        self.assertEqual(feats['Flow Duration'], 1.0)

        # A packet with payload retains positive Flow Bytes/s
        p_data = IP(src="192.168.1.100", dst="10.0.0.1")/TCP(sport=54322, dport=80, flags="PA")/b"PAYLOAD"
        p_data.time = 101.0
        flow_data = self.aggregator.process_packet(p_data)
        feats_data = flow_data.extract_features()
        self.assertGreater(feats_data['Flow Bytes/s'], 0.0)

    def test_host_port_scan_detection(self):
        """Verifies that scanning multiple distinct destination ports triggers host-level port scan flag."""
        agg = FlowAggregator(scan_threshold=5, scan_window=5.0)
        t0 = 200.0
        last_flow = None
        for port in range(1, 7):
            pkt = IP(src="192.168.1.50", dst="192.168.1.2")/TCP(sport=40000 + port, dport=port, flags="S")
            pkt.time = t0 + (port * 0.01)
            last_flow = agg.process_packet(pkt)

        self.assertIsNotNone(last_flow)
        self.assertTrue(getattr(last_flow, 'is_port_scan', False))


class TestEmailAlertManager(unittest.TestCase):
    """Tests for email formatting, SMTP sending, and anti-flood cooldown."""

    def setUp(self):
        self.manager = EmailAlertManager(
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_user="test@test.com",
            smtp_pass="password",
            sender="aids@test.com",
            recipient="admin@test.com",
            cooldown_seconds=2.0,
            enabled=True
        )

    def test_format_content(self):
        """Verifies formatting of HTML and Plain text email bodies."""
        alert_data = {
            'attack_type': 'DoS',
            'probability': 0.985,
            'src_ip': '192.168.1.150',
            'src_port': 55432,
            'dst_ip': '192.168.1.1',
            'dst_port': 80,
            'protocol_name': 'TCP',
            'duration_sec': 0.85,
            'total_packets': 320,
            'total_bytes': 24500,
            'interface': 'eth0'
        }
        text_body, html_body = self.manager.format_alert_content(alert_data, suppressed_count=5)
        self.assertIn("DoS", text_body)
        self.assertIn("192.168.1.150", text_body)
        self.assertIn("5 evento(s)", text_body)
        self.assertIn("<html", html_body)
        self.assertIn("98.50%", html_body)
        self.assertIn("192.168.1.150", html_body)

    def test_anti_flood_throttling(self):
        """Verifies suppression of duplicate alerts within cooldown window."""
        key = "192.168.1.200_DoS"
        t0 = 100.0

        # First alert: should send
        should_send, count = self.manager.should_alert(key, now=t0)
        self.assertTrue(should_send)
        self.assertEqual(count, 0)

        # Immediate follow-up: should be suppressed
        should_send, count = self.manager.should_alert(key, now=t0 + 0.5)
        self.assertFalse(should_send)
        self.assertEqual(count, 1)

        # Another follow-up: should be suppressed
        should_send, count = self.manager.should_alert(key, now=t0 + 1.0)
        self.assertFalse(should_send)
        self.assertEqual(count, 2)

        # After cooldown (2.0s): should send with aggregated count of 2
        should_send, count = self.manager.should_alert(key, now=t0 + 2.5)
        self.assertTrue(should_send)
        self.assertEqual(count, 2)

    @patch("smtplib.SMTP")
    def test_smtp_send(self, mock_smtp_cls):
        """Verifies SMTP communication flow with TLS."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server

        alert_data = {
            'attack_type': 'Exploits',
            'probability': 0.95,
            'src_ip': '10.10.10.10',
            'src_port': 1234,
            'dst_ip': '10.10.10.1',
            'dst_port': 22,
            'protocol_name': 'TCP',
            'duration_sec': 0.1,
            'total_packets': 5,
            'total_bytes': 300,
            'interface': 'eth0'
        }

        result = self.manager.send_alert(alert_data, async_send=False)
        self.assertTrue(result)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@test.com", "password")
        mock_server.sendmail.assert_called_once()
        mock_server.quit.assert_called_once()


def _create_mock_binary_pipeline():
    mock_clf = MagicMock()
    mock_clf.classes_ = np.array([0, 1])
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'classifier': mock_clf}
    mock_pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
    return mock_pipeline


def _create_mock_multiclass_pipeline():
    classes = ['Benign', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    mock_clf = MagicMock()
    mock_clf.classes_ = np.array(classes)
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'classifier': mock_clf}
    probs = np.array([0.01, 0.01, 0.01, 0.01, 0.95, 0.00, 0.00, 0.00, 0.00, 0.01])
    mock_pipeline.predict_proba.return_value = np.array([probs])
    return mock_pipeline


def _create_mock_dt_pipeline(predicted_class='DoS', class_prob=0.88):
    classes = ['Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    mock_clf = MagicMock()
    mock_clf.classes_ = np.array(classes)
    mock_pipeline = MagicMock()
    mock_pipeline.named_steps = {'classifier': mock_clf}
    probs = np.zeros(len(classes))
    idx = classes.index(predicted_class)
    probs[idx] = class_prob
    mock_pipeline.predict_proba.return_value = np.array([probs])
    return mock_pipeline


class TestRPIDetector(unittest.TestCase):
    """Tests for the RPi detector engine and model inference."""

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_binary_detection_pipeline(self, mock_resolve, mock_load):
        """Tests binary intrusion detection against trained pipeline."""
        mock_load.return_value = _create_mock_binary_pipeline()

        email_manager = EmailAlertManager(enabled=False)
        detector = RPIDetector(
            mode="binary",
            model_path="models/stacking_pipeline_binary.joblib",
            dry_run=True,
            email_manager=email_manager
        )

        flow = Flow("175.45.176.2", 23357, "149.171.126.16", 80, "6", start_time=100.0)
        flow.add_packet(is_fwd=True, pkt_len=120, header_len=20, timestamp=100.0, tcp_flags={'SYN': True})
        flow.add_packet(is_fwd=False, pkt_len=60, header_len=20, timestamp=100.01, tcp_flags={'SYN': True, 'ACK': True})
        flow.add_packet(is_fwd=True, pkt_len=500, header_len=20, timestamp=100.05, tcp_flags={'ACK': True, 'PSH': True}, payload_len=460)
        flow.add_packet(is_fwd=True, pkt_len=60, header_len=20, timestamp=100.10, tcp_flags={'FIN': True, 'ACK': True})

        result = detector.predict_flow(flow)
        self.assertIn('is_attack', result)
        self.assertIn('attack_type', result)
        self.assertIn('probability', result)
        self.assertIsInstance(result['is_attack'], (bool, np.bool_))
        self.assertGreaterEqual(result['probability'], 0.0)
        self.assertLessEqual(result['probability'], 1.0)

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_multiclass.joblib")
    def test_multiclass_detection_pipeline(self, mock_resolve, mock_load):
        """Tests multiclass intrusion detection against trained pipeline."""
        mock_load.return_value = _create_mock_multiclass_pipeline()

        email_manager = EmailAlertManager(enabled=False)
        detector = RPIDetector(
            mode="multiclass",
            model_path="models/stacking_pipeline_multiclass.joblib",
            dry_run=True,
            email_manager=email_manager
        )

        flow = Flow("10.0.0.99", 54321, "10.0.0.1", 8080, "6", start_time=200.0)
        flow.add_packet(is_fwd=True, pkt_len=200, header_len=20, timestamp=200.0, tcp_flags={'SYN': True})
        flow.add_packet(is_fwd=True, pkt_len=1400, header_len=20, timestamp=200.02, tcp_flags={'ACK': True, 'PSH': True}, payload_len=1360)
        flow.add_packet(is_fwd=False, pkt_len=100, header_len=20, timestamp=200.04, tcp_flags={'ACK': True})

        result = detector.predict_flow(flow)
        self.assertIn('is_attack', result)
        self.assertIn('attack_type', result)
        self.assertIn(result['attack_type'], [
            'Benign', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
        ])

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_detector_packet_stream(self, mock_resolve, mock_load):
        """Simulates full packet stream ingestion and evaluation."""
        mock_load.return_value = _create_mock_binary_pipeline()

        email_manager = MagicMock()
        detector = RPIDetector(
            mode="binary",
            dry_run=False,
            email_manager=email_manager
        )

        t0 = time.time()
        p1 = IP(src="192.168.5.10", dst="192.168.5.1")/TCP(sport=33000, dport=445, flags="S")
        p1.time = t0
        detector.process_packet(p1)

        p2 = IP(src="192.168.5.10", dst="192.168.5.1")/TCP(sport=33000, dport=445, flags="FA")
        p2.time = t0 + 0.05
        detector.process_packet(p2)

        stats = detector.get_stats()
        self.assertGreaterEqual(stats['total_flows_evaluated'], 1)
        self.assertIn('detections_logged', stats)
        self.assertGreaterEqual(stats['detections_logged'], 1)
        self.assertGreaterEqual(stats['cpu_percent'], 0.0)
        self.assertGreater(stats['ram_used_mb'], 0.0)

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_dt_multiclass_attack_classification(self, mock_resolve, mock_load):
        """Verifies that when a flow is detected as malicious, DT multiclass identifies the attack type."""
        mock_load.return_value = _create_mock_binary_pipeline()

        detector = RPIDetector(
            mode="binary",
            dry_run=True,
            email_manager=EmailAlertManager(enabled=False),
            use_dt_multiclass=True
        )
        detector.dt_multiclass_pipeline = _create_mock_dt_pipeline(predicted_class='DoS', class_prob=0.92)

        flow = Flow("192.168.1.100", 50000, "10.0.0.1", 80, "6", start_time=100.0)
        flow.add_packet(is_fwd=True, pkt_len=100, header_len=20, timestamp=100.0, tcp_flags={'SYN': True})
        flow.add_packet(is_fwd=False, pkt_len=60, header_len=20, timestamp=100.01, tcp_flags={'SYN': True, 'ACK': True})

        result = detector.predict_flow(flow)
        self.assertTrue(result['is_attack'])
        self.assertEqual(result['attack_type'], 'DoS')
        self.assertEqual(result['dt_attack_type'], 'DoS')
        self.assertAlmostEqual(result['dt_confidence'], 0.92)
        self.assertAlmostEqual(result['probability'], 0.9)

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_dt_multiclass_benign_argmax_fallback(self, mock_resolve, mock_load):
        """When binary detects attack but DT argmax is Benign, selects highest probability attack class."""
        mock_load.return_value = _create_mock_binary_pipeline()

        detector = RPIDetector(
            mode="binary",
            dry_run=True,
            email_manager=EmailAlertManager(enabled=False),
            use_dt_multiclass=True
        )
        # DT mock where Benign is 0.6, but Exploits is 0.35 (highest attack class)
        classes = ['Analysis', 'Backdoor', 'Benign', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
        mock_clf = MagicMock()
        mock_clf.classes_ = np.array(classes)
        mock_pipe = MagicMock()
        mock_pipe.named_steps = {'classifier': mock_clf}
        probs = np.zeros(len(classes))
        probs[classes.index('Benign')] = 0.60
        probs[classes.index('Exploits')] = 0.35
        mock_pipe.predict_proba.return_value = np.array([probs])
        detector.dt_multiclass_pipeline = mock_pipe

        flow = Flow("192.168.1.100", 50000, "10.0.0.1", 80, "6", start_time=100.0)
        flow.add_packet(is_fwd=True, pkt_len=100, header_len=20, timestamp=100.0, tcp_flags={'SYN': True})

        result = detector.predict_flow(flow)
        self.assertTrue(result['is_attack'])
        self.assertEqual(result['attack_type'], 'Exploits')
        self.assertAlmostEqual(result['dt_confidence'], 0.35)

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_dt_multiclass_disabled_fallback(self, mock_resolve, mock_load):
        """When DT multiclass is disabled, falls back to generic label."""
        mock_load.return_value = _create_mock_binary_pipeline()

        detector = RPIDetector(
            mode="binary",
            dry_run=True,
            email_manager=EmailAlertManager(enabled=False),
            use_dt_multiclass=False
        )

        flow = Flow("192.168.1.100", 50000, "10.0.0.1", 80, "6", start_time=100.0)
        flow.add_packet(is_fwd=True, pkt_len=100, header_len=20, timestamp=100.0, tcp_flags={'SYN': True})

        result = detector.predict_flow(flow)
        self.assertTrue(result['is_attack'])
        self.assertEqual(result['attack_type'], 'Maligno (Ataque Detectado)')

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_cascade_mode_execution(self, mock_resolve, mock_load):
        """Verifies cascade mode initializes binary stage 1 and uses DT multiclass."""
        mock_load.return_value = _create_mock_binary_pipeline()

        detector = RPIDetector(
            mode="cascade",
            dry_run=True,
            email_manager=EmailAlertManager(enabled=False),
            use_dt_multiclass=True
        )
        detector.dt_multiclass_pipeline = _create_mock_dt_pipeline(predicted_class='Fuzzers', class_prob=0.85)

        flow = Flow("10.0.0.50", 40000, "10.0.0.1", 22, "6", start_time=150.0)
        flow.add_packet(is_fwd=True, pkt_len=120, header_len=20, timestamp=150.0, tcp_flags={'SYN': True})

        result = detector.predict_flow(flow)
        self.assertTrue(result['is_attack'])
        self.assertEqual(result['attack_type'], 'Fuzzers')

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_port_scan_immediate_classification(self, mock_resolve, mock_load):
        """Verifies that a flow marked as port scan is classified as Reconnaissance."""
        mock_pipeline = MagicMock()
        mock_clf = MagicMock()
        mock_clf.classes_ = np.array([0, 1])
        mock_pipeline.named_steps = {'classifier': mock_clf}
        mock_pipeline.predict_proba.return_value = np.array([[0.85, 0.15]])
        mock_load.return_value = mock_pipeline

        detector = RPIDetector(
            mode="binary",
            dry_run=True,
            email_manager=EmailAlertManager(enabled=False)
        )

        flow = Flow("192.168.1.50", 40001, "192.168.1.2", 22, "6", start_time=100.0)
        flow.is_port_scan = True
        result = detector.predict_flow(flow)

        self.assertTrue(result['is_attack'])
        self.assertEqual(result['attack_type'], 'Reconnaissance')
        self.assertGreaterEqual(result['probability'], 0.99)

    @patch.object(RPIDetector, '_load_pipeline')
    @patch.object(RPIDetector, '_resolve_model_path', return_value="models/stacking_pipeline_binary.joblib")
    def test_no_duplicate_evaluation_on_rst(self, mock_resolve, mock_load):
        """Verifies that flows evaluated upon RST termination are not re-evaluated during periodic flush."""
        mock_load.return_value = _create_mock_binary_pipeline()
        detector = RPIDetector(
            mode="binary",
            dry_run=True,
            email_manager=EmailAlertManager(enabled=False)
        )

        t0 = 100.0
        p1 = IP(src="192.168.1.50", dst="192.168.1.2")/TCP(sport=50001, dport=80, flags="S")
        p1.time = t0
        detector.process_packet(p1)

        # RST received: finishes flow and evaluates once
        p2 = IP(src="192.168.1.2", dst="192.168.1.50")/TCP(sport=80, dport=50001, flags="RA")
        p2.time = t0 + 0.001
        detector.process_packet(p2)

        self.assertEqual(detector.total_flows_evaluated, 1)

        # Periodic flush should not re-evaluate the already-evaluated flow
        flushed = detector.flush_and_detect(current_time=t0 + 5.0)
        self.assertEqual(len(flushed), 0)
        self.assertEqual(detector.total_flows_evaluated, 1)


if __name__ == '__main__':
    unittest.main()

