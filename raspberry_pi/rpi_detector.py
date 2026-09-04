"""
Edge Intrusion Detection Engine for Raspberry Pi.
Integrates real-time packet capture, flow aggregation, Stacking Ensemble inference,
and automated email alerts with anti-flood protection.
"""

import os
import sys
import time
import logging
import threading
import signal
import psutil
import joblib
import pandas as pd
import numpy as np

try:
    from .config import load_rpi_env
    from .flow_aggregator import FlowAggregator, Flow
    from .email_alert import EmailAlertManager
    from .detection_logger import DetectionLogger
except (ImportError, ValueError):
    from config import load_rpi_env
    from flow_aggregator import FlowAggregator, Flow
    from email_alert import EmailAlertManager
    from detection_logger import DetectionLogger

load_rpi_env()





logger = logging.getLogger("AIDS.RPIDetector")


class RPIDetector:
    """
    Real-time Network Intrusion Detection Engine optimized for Raspberry Pi Edge devices.
    """

    def __init__(
        self,
        mode: str = "binary",
        model_path: str = None,
        interface: str = None,
        threshold: float = 0.5,
        inactivity_timeout: float = 10.0,
        active_timeout: float = 60.0,
        max_flows: int = 10000,
        dry_run: bool = False,
        email_manager: EmailAlertManager = None,
        detection_logger: DetectionLogger = None
    ):
        self.mode = mode.lower()
        self.interface = interface or os.getenv("NETWORK_INTERFACE", "eth0")
        self.threshold = float(threshold)
        self.dry_run = dry_run

        # Initialize Flow Aggregator
        self.aggregator = FlowAggregator(
            inactivity_timeout=inactivity_timeout,
            active_timeout=active_timeout,
            max_flows=max_flows
        )

        # Initialize Email Alert Manager
        self.email_manager = email_manager or EmailAlertManager(enabled=not dry_run)

        # Initialize Detection Logger (Persistent Audit Trail)
        self.detection_logger = detection_logger or DetectionLogger()

        # Load Stacking Model & Preprocessor Pipeline
        self.model_path = model_path or self._resolve_model_path()
        self.pipeline = self._load_pipeline(self.model_path)

        # Runtime Stats
        self.is_running = False
        self.stop_event = threading.Event()
        self.total_flows_evaluated = 0
        self.total_attacks_detected = 0
        self.attack_type_counts = {}
        self._stats_lock = threading.Lock()
        self._eval_thread = None

    def _resolve_model_path(self) -> str:
        """Finds the default pipeline model joblib based on detection mode."""
        # Check current working directory or parent directory for models/
        search_dirs = [".", "..", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
        for base in search_dirs:
            p1 = os.path.join(base, f"models/stacking_pipeline_{self.mode}.joblib")
            if os.path.exists(p1):
                return p1
            p2 = os.path.join(base, f"models/Stacking_{self.mode}.joblib")
            if os.path.exists(p2):
                return p2

        raise FileNotFoundError(f"Modelo não encontrado para o modo '{self.mode}'. Execute o treinamento primeiro.")

    def _load_pipeline(self, model_path: str):
        """Loads model or builds pipeline with preprocessor."""
        logger.info("Carregando modelo de detecção de: %s", model_path)
        obj = joblib.load(model_path)

        # If object is already a full scikit-learn Pipeline
        if hasattr(obj, 'named_steps') and 'preprocessor' in obj.named_steps:
            return obj

        # Otherwise, load preprocessor and combine with classifier
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
        scaler_candidates = [
            f"models/scaler_{self.mode}.joblib",
            os.path.join(base_dir, "models", f"scaler_{self.mode}.joblib"),
            os.path.join(os.path.dirname(model_path), f"scaler_{self.mode}.joblib")
        ]
        for scaler_path in scaler_candidates:
            if os.path.exists(scaler_path):
                preprocessor = joblib.load(scaler_path)
                from sklearn.pipeline import Pipeline
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', obj)
                ])
                return pipeline

        raise RuntimeError("Não foi possível carregar o pipeline completo (preprocessor + classificador).")

    def predict_flow(self, flow: Flow) -> dict:
        """
        Extracts features from flow and executes inference with the Stacking Pipeline.
        """
        df = flow.to_dataframe()
        summary = flow.get_summary()

        try:
            proba_arr = self.pipeline.predict_proba(df)[0]
            clf = self.pipeline.named_steps['classifier']
            classes = list(clf.classes_)

            if self.mode == "binary":
                # Class 0: Benign, Class 1: Malicious
                malicious_idx = 1 if 1 in classes else (classes.index('1') if '1' in classes else len(classes) - 1)
                malicious_prob = float(proba_arr[malicious_idx])
                is_attack = malicious_prob >= self.threshold
                attack_type = "Maligno (Ataque Detectado)" if is_attack else "Benign"
                prob = malicious_prob if is_attack else float(proba_arr[0])
            else:
                # Multiclass
                best_idx = int(np.argmax(proba_arr))
                pred_label = str(classes[best_idx])
                prob = float(proba_arr[best_idx])
                is_attack = (pred_label.strip().upper() != "BENIGN") and (prob >= self.threshold)
                attack_type = pred_label if is_attack else "Benign"

            return {
                'is_attack': is_attack,
                'attack_type': attack_type,
                'probability': prob,
                'all_probabilities': {str(c): float(p) for c, p in zip(classes, proba_arr)},
                'flow_summary': summary
            }
        except Exception as ex:
            logger.error("Erro durante inferência do fluxo: %s", str(ex))
            return {
                'is_attack': False,
                'attack_type': "Error",
                'probability': 0.0,
                'flow_summary': summary
            }

    def process_packet(self, packet):
        """Processes an incoming raw packet."""
        flow = self.aggregator.process_packet(packet)
        # If TCP FIN or RST finished the flow immediately, evaluate it
        if flow and flow.is_finished:
            self._evaluate_single_flow(flow)
        return flow

    def _evaluate_single_flow(self, flow: Flow):
        """Classifies a completed flow and dispatches alerts if malicious."""
        res = self.predict_flow(flow)
        with self._stats_lock:
            self.total_flows_evaluated += 1

        if res['is_attack']:
            with self._stats_lock:
                self.total_attacks_detected += 1
                atk_type = res['attack_type']
                self.attack_type_counts[atk_type] = self.attack_type_counts.get(atk_type, 0) + 1

            summary = res['flow_summary']
            logger.warning(
                "🚨 [ALERTA DE INTRUSÃO] Tipo: %s (Confiança: %.2f%%) | %s:%s -> %s:%s | %s",
                res['attack_type'],
                res['probability'] * 100,
                summary['src_ip'],
                summary['src_port'],
                summary['dst_ip'],
                summary['dst_port'],
                summary['protocol_name']
            )

            # Trigger email notification
            alert_payload = {
                'attack_type': res['attack_type'],
                'probability': res['probability'],
                'src_ip': summary['src_ip'],
                'src_port': summary['src_port'],
                'dst_ip': summary['dst_ip'],
                'dst_port': summary['dst_port'],
                'protocol_name': summary['protocol_name'],
                'duration_sec': summary['duration_sec'],
                'total_packets': summary['total_packets'],
                'total_bytes': summary['total_bytes'],
                'interface': self.interface
            }
            if not self.dry_run:
                self.email_manager.send_alert(alert_payload, async_send=True)
            else:
                logger.info("[DRY-RUN] Alerta de e-mail simulado com sucesso.")

        # Record to persistent detection log (JSONL/CSV/Text with rotation)
        if self.detection_logger and (res['is_attack'] or self.detection_logger.log_all_flows):
            self.detection_logger.log_detection(res, flow=flow, interface=self.interface)

        return res

    def flush_and_detect(self, current_time: float = None, force: bool = False) -> list:
        """Flushes expired flows from the aggregator and runs detection."""
        expired_flows = self.aggregator.flush_expired(current_time, force=force)
        results = []
        for flow in expired_flows:
            res = self._evaluate_single_flow(flow)
            results.append(res)
        return results

    def _periodic_evaluation_worker(self, interval: float = 1.0):
        """Background thread worker to flush expired flows at fixed intervals."""
        while not self.stop_event.is_set():
            try:
                self.flush_and_detect()
            except Exception as ex:
                logger.error("Erro no worker de avaliação periódica: %s", str(ex))
            self.stop_event.wait(timeout=interval)

    def start_sniffing(self, iface: str = None, bpf_filter: str = "ip", timeout: int = None):
        """
        Starts live packet capture using Scapy on the specified interface.
        """
        from scapy.sendrecv import sniff

        capture_iface = iface or self.interface
        logger.info("📡 Iniciando captura em tempo real na interface: %s (Filtro: %s)", capture_iface, bpf_filter)
        logger.info("🧠 Modo de Classificação: %s | Limiar de Detecção: %.2f", self.mode.upper(), self.threshold)

        self.is_running = True
        self.stop_event.clear()

        # Start periodic background flush worker
        self._eval_thread = threading.Thread(
            target=self._periodic_evaluation_worker,
            args=(1.0,),
            daemon=True
        )
        self._eval_thread.start()

        def _packet_callback(pkt):
            if self.stop_event.is_set():
                return
            self.process_packet(pkt)

        try:
            sniff(
                iface=capture_iface,
                prn=_packet_callback,
                filter=bpf_filter,
                store=False,
                timeout=timeout,
                stop_filter=lambda _: self.stop_event.is_set()
            )
        except KeyboardInterrupt:
            logger.info("Captura interrompida pelo usuário.")
        finally:
            self.stop()

    def replay_pcap(self, pcap_path: str, show_progress: bool = True) -> dict:
        """
        Replays a PCAP file for offline testing or validation without physical sniffing.
        Uses streaming PcapReader to avoid RAM spikes on Raspberry Pi.
        """
        from scapy.utils import PcapReader

        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"Arquivo PCAP não encontrado: {pcap_path}")

        logger.info("📂 Lendo e reproduzindo PCAP: %s ...", pcap_path)
        start_time = time.time()
        pkt_count = 0

        with PcapReader(pcap_path) as pcap_reader:
            for pkt in pcap_reader:
                self.process_packet(pkt)
                pkt_count += 1
                if show_progress and pkt_count % 1000 == 0:
                    sys.stdout.write(f"\rProcessados {pkt_count} pacotes...")
                    sys.stdout.flush()

        # Flush remaining flows forcibly at the end of the pcap
        self.flush_and_detect(force=True)
        elapsed = time.time() - start_time
        if show_progress:
            sys.stdout.write("\n")

        stats = self.get_stats()
        logger.info(
            "✅ Replay concluído: %d pacotes em %.2fs (%.1f pkts/s). %d fluxos avaliados, %d ataques detectados.",
            pkt_count, elapsed, (pkt_count / elapsed if elapsed > 0 else 0),
            stats['total_flows_evaluated'], stats['total_attacks_detected']
        )
        return stats

    def stop(self):
        """Stops the sniffer and background threads cleanly."""
        self.is_running = False
        self.stop_event.set()
        if self._eval_thread and self._eval_thread.is_alive():
            self._eval_thread.join(timeout=2.0)
        # Final flush
        self.flush_and_detect(current_time=time.time() + 3600)
        logger.info("🛑 Detector de intrusão finalizado com sucesso.")

    def get_stats(self) -> dict:
        """Returns runtime performance and intrusion detection metrics."""
        with self._stats_lock:
            agg_stats = self.aggregator.get_stats()
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            logged_count = self.detection_logger.total_logged if self.detection_logger else 0
            log_path = self.detection_logger.log_file if self.detection_logger else None
            return {
                'active_flows': agg_stats['active_flows'],
                'total_packets_processed': agg_stats['total_packets_processed'],
                'total_flows_evaluated': self.total_flows_evaluated,
                'total_attacks_detected': self.total_attacks_detected,
                'attack_type_counts': dict(self.attack_type_counts),
                'alerts_sent': self.email_manager.total_alerts_sent,
                'alerts_suppressed': self.email_manager.total_alerts_suppressed,
                'detections_logged': logged_count,
                'detection_log_file': log_path,
                'cpu_percent': cpu,
                'ram_percent': mem.percent,
                'ram_used_mb': round(mem.used / (1024 * 1024), 1)
            }
