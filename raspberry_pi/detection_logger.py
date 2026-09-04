"""
Detection Logger Module for Network Intrusion Detection System (AIDS-RPi).
Provides persistent, thread-safe, and size-rotated audit logging for detected malicious
packets and network flows, supporting JSON Lines (JSONL), CSV, and formatted Text.
"""

import os
import sys
import json
import csv
import time
import socket
import logging
import threading
from datetime import datetime
try:
    from .config import load_rpi_env
    load_rpi_env()
except (ImportError, ValueError):
    try:
        from config import load_rpi_env
        load_rpi_env()
    except (ImportError, ValueError):
        from dotenv import load_dotenv
        load_dotenv()

logger = logging.getLogger("AIDS.DetectionLogger")




CSV_FIELDNAMES = [
    "timestamp",
    "epoch",
    "event",
    "attack_type",
    "is_attack",
    "confidence",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "interface",
    "duration_sec",
    "total_packets",
    "total_bytes",
    "fwd_packets",
    "bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "packets_per_sec",
    "bytes_per_sec",
    "mitigation_rule"
]


class DetectionLogger:
    """
    Thread-safe, high-performance security logger for detected malicious network packets and flows.
    Supports automatic file rotation to protect storage space on Edge devices (Raspberry Pi).
    """

    def __init__(
        self,
        log_file: str = None,
        log_format: str = None,
        max_bytes: int = None,
        backup_count: int = None,
        log_all_flows: bool = None,
        enabled: bool = None
    ):
        # Configuration with environment variable fallbacks
        env_enabled = os.getenv("DETECTION_LOG_ENABLED", "True").strip().lower() == "true"
        self.enabled = enabled if enabled is not None else env_enabled

        self.log_file = log_file or os.getenv("DETECTION_LOG_FILE", "logs/detections.jsonl")
        
        # Determine format from argument, env, or file extension
        raw_fmt = log_format or os.getenv("DETECTION_LOG_FORMAT", "")
        if raw_fmt:
            self.log_format = raw_fmt.strip().lower()
        else:
            ext = os.path.splitext(self.log_file)[1].lower()
            if ext in (".csv",):
                self.log_format = "csv"
            elif ext in (".log", ".txt"):
                self.log_format = "text"
            else:
                self.log_format = "jsonl"

        # If format is "both" or "all", designate secondary CSV file
        if self.log_format in ("all", "both"):
            base_dir = os.path.dirname(os.path.abspath(self.log_file))
            base_name = os.path.splitext(os.path.basename(self.log_file))[0]
            self.csv_file = os.path.join(base_dir, f"{base_name}.csv")
        else:
            self.csv_file = None

        env_max_bytes = int(os.getenv("DETECTION_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
        self.max_bytes = int(max_bytes if max_bytes is not None else env_max_bytes)

        env_backup_count = int(os.getenv("DETECTION_LOG_BACKUP_COUNT", "5"))
        self.backup_count = int(backup_count if backup_count is not None else env_backup_count)

        env_all_flows = os.getenv("DETECTION_LOG_ALL_FLOWS", "False").strip().lower() == "true"
        self.log_all_flows = log_all_flows if log_all_flows is not None else env_all_flows

        # Concurrency & Stats
        self._lock = threading.Lock()
        self.total_logged = 0
        self.total_attacks_logged = 0
        self.total_benign_logged = 0
        self.last_log_time = None
        self.hostname = socket.gethostname()

        # Initialize directory and header if needed
        if self.enabled and self.log_file:
            self._ensure_storage_ready()

    def _ensure_storage_ready(self):
        """Creates target directory and initializes CSV header if applicable."""
        try:
            dirname = os.path.dirname(os.path.abspath(self.log_file))
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            if self.log_format == "csv":
                if not os.path.exists(self.log_file) or os.path.getsize(self.log_file) == 0:
                    self._write_csv_header(self.log_file)
            elif self.csv_file:
                if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                    self._write_csv_header(self.csv_file)
        except Exception as ex:
            logger.error("Erro ao preparar armazenamento de logs de detecção: %s", str(ex))

    def _write_csv_header(self, filepath: str):
        """Writes CSV columns header to the given path."""
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
        except Exception as ex:
            logger.error("Falha ao gravar cabeçalho CSV em %s: %s", filepath, str(ex))

    def _rotate_if_needed(self, filepath: str, is_csv: bool = False):
        """
        Rotates log file if it exceeds max_bytes, keeping up to backup_count archives.
        Reinitializes CSV header if rotating a CSV file.
        """
        if not os.path.exists(filepath):
            if is_csv:
                self._write_csv_header(filepath)
            return

        try:
            if os.path.getsize(filepath) >= self.max_bytes:
                if self.backup_count > 0:
                    oldest = f"{filepath}.{self.backup_count}"
                    if os.path.exists(oldest):
                        os.remove(oldest)

                    for idx in range(self.backup_count - 1, 0, -1):
                        cur = f"{filepath}.{idx}"
                        nxt = f"{filepath}.{idx + 1}"
                        if os.path.exists(cur):
                            os.rename(cur, nxt)

                    os.rename(filepath, f"{filepath}.1")

                if is_csv:
                    self._write_csv_header(filepath)
        except Exception as ex:
            logger.error("Erro durante rotação do arquivo de detecção %s: %s", filepath, str(ex))

    def format_records(self, res: dict, flow=None, interface: str = None) -> tuple[dict, dict, str]:
        """
        Builds structured dictionary, flat CSV row, and text log line from detection event.
        """
        summary = res.get('flow_summary', {})
        src_ip = summary.get('src_ip', '0.0.0.0')
        src_port = int(summary.get('src_port', 0))
        dst_ip = summary.get('dst_ip', '0.0.0.0')
        dst_port = int(summary.get('dst_port', 0))
        protocol = summary.get('protocol_name', 'TCP/UDP')
        duration = float(summary.get('duration_sec', 0.0))
        total_pkts = int(summary.get('total_packets', 0))
        total_bytes = int(summary.get('total_bytes', 0))
        fwd_pkts = int(summary.get('fwd_packets', 0))
        bwd_pkts = int(summary.get('bwd_packets', 0))

        is_attack = bool(res.get('is_attack', False))
        attack_type = str(res.get('attack_type', 'Benign' if not is_attack else 'Maligno'))
        prob = float(res.get('probability', 0.0))

        now = time.time()
        iso_ts = datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Extract packet-level statistics if flow object is provided
        if flow is not None:
            fwd_lengths = getattr(flow, 'fwd_packet_lengths', [])
            bwd_lengths = getattr(flow, 'bwd_packet_lengths', [])
            all_lengths = getattr(flow, 'all_packet_lengths', [])
            fwd_bytes = int(sum(fwd_lengths))
            bwd_bytes = int(sum(bwd_lengths))
            min_len = int(min(all_lengths)) if all_lengths else 0
            max_len = int(max(all_lengths)) if all_lengths else 0
            mean_len = float(sum(all_lengths) / len(all_lengths)) if all_lengths else 0.0
            tcp_flags = {
                'FIN': getattr(flow, 'fin_flags', 0),
                'SYN': getattr(flow, 'syn_flags', 0),
                'RST': getattr(flow, 'rst_flags', 0),
                'PSH': getattr(flow, 'psh_flags', 0),
                'ACK': getattr(flow, 'ack_flags', 0)
            }
        else:
            fwd_bytes = int(summary.get('fwd_bytes', 0))
            bwd_bytes = int(summary.get('bwd_bytes', 0))
            min_len = 0
            max_len = 0
            mean_len = round(total_bytes / total_pkts, 2) if total_pkts > 0 else 0.0
            tcp_flags = summary.get('tcp_flags', None)

        pkts_per_sec = round(total_pkts / duration, 2) if duration > 0 else float(total_pkts)
        bytes_per_sec = round(total_bytes / duration, 2) if duration > 0 else float(total_bytes)

        mitigation_cmd = f"sudo iptables -A INPUT -s {src_ip} -j DROP" if is_attack and src_ip != "0.0.0.0" else "N/A"

        # 1. Structured JSON Record
        json_record = {
            "timestamp": iso_ts,
            "epoch": round(now, 4),
            "event": "INTRUSION_DETECTED" if is_attack else "BENIGN_FLOW",
            "attack_type": attack_type,
            "is_attack": is_attack,
            "confidence": round(prob, 4),
            "confidence_pct": f"{prob * 100:.2f}%",
            "source": {
                "ip": src_ip,
                "port": src_port
            },
            "destination": {
                "ip": dst_ip,
                "port": dst_port
            },
            "protocol": protocol,
            "interface": interface or "unknown",
            "flow_metrics": {
                "duration_sec": round(duration, 4),
                "total_packets": total_pkts,
                "total_bytes": total_bytes,
                "fwd_packets": fwd_pkts,
                "bwd_packets": bwd_pkts,
                "fwd_bytes": fwd_bytes,
                "bwd_bytes": bwd_bytes,
                "packets_per_sec": pkts_per_sec,
                "bytes_per_sec": bytes_per_sec
            },
            "packet_stats": {
                "length_min": min_len,
                "length_max": max_len,
                "length_mean": round(mean_len, 2),
                "tcp_flags": tcp_flags
            },
            "mitigation_rule": mitigation_cmd,
            "hostname": self.hostname
        }

        # 2. Flat CSV Record
        csv_row = {
            "timestamp": iso_ts,
            "epoch": round(now, 4),
            "event": "INTRUSION_DETECTED" if is_attack else "BENIGN_FLOW",
            "attack_type": attack_type,
            "is_attack": is_attack,
            "confidence": round(prob, 4),
            "src_ip": src_ip,
            "src_port": src_port,
            "dst_ip": dst_ip,
            "dst_port": dst_port,
            "protocol": protocol,
            "interface": interface or "unknown",
            "duration_sec": round(duration, 4),
            "total_packets": total_pkts,
            "total_bytes": total_bytes,
            "fwd_packets": fwd_pkts,
            "bwd_packets": bwd_pkts,
            "fwd_bytes": fwd_bytes,
            "bwd_bytes": bwd_bytes,
            "packets_per_sec": pkts_per_sec,
            "bytes_per_sec": bytes_per_sec,
            "mitigation_rule": mitigation_cmd
        }

        # 3. Formatted Human-readable Text Line
        event_tag = "INTRUSION_DETECTED" if is_attack else "BENIGN_FLOW"
        text_line = (
            f"[{iso_ts}] [{event_tag}] "
            f"Type: {attack_type} (Conf: {prob * 100:.2f}%) | "
            f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} ({protocol}) | "
            f"Pkts: {total_pkts}, Bytes: {total_bytes:,}, Dur: {duration:.4f}s | "
            f"Action: {mitigation_cmd}"
        )

        return json_record, csv_row, text_line

    def log_detection(self, res: dict, flow=None, interface: str = None) -> bool:
        """
        Logs a detected event to the configured persistent log file.
        If log_all_flows is False and res['is_attack'] is False, the call is ignored.
        """
        if not self.enabled:
            return False

        is_attack = bool(res.get('is_attack', False))
        if not is_attack and not self.log_all_flows:
            return False

        json_rec, csv_rec, text_rec = self.format_records(res, flow=flow, interface=interface)

        with self._lock:
            try:
                if self.log_format in ("json", "jsonl"):
                    self._rotate_if_needed(self.log_file, is_csv=False)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(json_rec, ensure_ascii=False) + "\n")

                elif self.log_format == "csv":
                    self._rotate_if_needed(self.log_file, is_csv=True)
                    with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                        writer.writerow(csv_rec)

                elif self.log_format == "text":
                    self._rotate_if_needed(self.log_file, is_csv=False)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(text_rec + "\n")

                elif self.log_format in ("all", "both"):
                    # Write JSONL to main file
                    self._rotate_if_needed(self.log_file, is_csv=False)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(json_rec, ensure_ascii=False) + "\n")

                    # Write CSV to secondary file
                    if self.csv_file:
                        self._rotate_if_needed(self.csv_file, is_csv=True)
                        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                            writer.writerow(csv_rec)

                self.total_logged += 1
                if is_attack:
                    self.total_attacks_logged += 1
                else:
                    self.total_benign_logged += 1
                self.last_log_time = time.time()
                return True

            except Exception as ex:
                logger.error("❌ Falha ao gravar no log de detecção (%s): %s", self.log_file, str(ex))
                return False

    def log_flow(self, flow, res: dict, interface: str = None) -> bool:
        """Convenience alias for log_detection with flow as primary argument."""
        return self.log_detection(res, flow=flow, interface=interface)

    def get_recent_logs(self, limit: int = 10) -> list:
        """Reads and returns the last N records from the log file."""
        if not os.path.exists(self.log_file):
            return []

        with self._lock:
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                selected = lines[-limit:] if len(lines) > limit else lines

                if self.log_format in ("json", "jsonl", "all", "both"):
                    parsed = []
                    for item in selected:
                        try:
                            parsed.append(json.loads(item))
                        except Exception:
                            parsed.append(item)
                    return parsed

                return selected
            except Exception as ex:
                logger.error("Erro ao ler logs recentes de %s: %s", self.log_file, str(ex))
                return []

    def get_stats(self) -> dict:
        """Returns runtime logger statistics."""
        with self._lock:
            file_size = 0
            if os.path.exists(self.log_file):
                try:
                    file_size = os.path.getsize(self.log_file)
                except OSError:
                    pass

            return {
                "enabled": self.enabled,
                "log_file": self.log_file,
                "log_format": self.log_format,
                "log_all_flows": self.log_all_flows,
                "total_logged": self.total_logged,
                "total_attacks_logged": self.total_attacks_logged,
                "total_benign_logged": self.total_benign_logged,
                "last_log_time": self.last_log_time,
                "file_size_bytes": file_size
            }
