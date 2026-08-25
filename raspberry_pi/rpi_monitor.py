#!/usr/bin/env python3
"""
AIDS-RPi: Real-Time Network Intrusion Detection System for Raspberry Pi.
Captures live network packets, performs Stacking Ensemble inference on flows,
and dispatches real-time security alerts via email.

Usage:
    python3 rpi_monitor.py --interface eth0 --mode binary
    python3 rpi_monitor.py --pcap test.pcap --mode multiclass
    python3 rpi_monitor.py --test-email
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
from dotenv import load_dotenv

try:
    from .rpi_detector import RPIDetector
    from .email_alert import EmailAlertManager
except (ImportError, ValueError):
    from rpi_detector import RPIDetector
    from email_alert import EmailAlertManager

# Load environment configuration
load_dotenv()


def setup_logging(log_file: str = None, verbose: bool = False):
    """Configures console and file logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


def print_banner():
    """Displays startup ASCII banner and hardware information."""
    banner = """
  █████╗ ██╗██████╗ ███████╗      ██████╗ ██████╗ ██╗
 ██╔══██╗██║██╔══██╗██╔════╝      ██╔══██╗██╔══██╗██║
 ███████║██║██║  ██║███████╗█████╗██████╔╝██████╔╝██║
 ██╔══██║██║██║  ██║╚════██║╚════╝██╔══██╗██╔═══╝ ██║
 ██║  ██║██║██████╔╝███████║      ██║  ██║██║     ██║
 ╚═╝  ╚═╝╚═╝╚═════╝ ╚══════╝      ╚═╝  ╚═╝╚═╝     ╚═╝
 Autonomous Network Intrusion Detection System for Edge Devices
================================================================
"""
    print(banner)


def run_stats_reporter(detector: RPIDetector, interval: int, stop_event: threading.Event):
    """Periodically logs system performance and detection statistics."""
    while not stop_event.wait(timeout=interval):
        stats = detector.get_stats()
        logging.info(
            "📊 [STATUS] CPU: %.1f%% | RAM: %.1f%% (%s MB) | Fluxos Ativos: %d | Avaliados: %d | Ataques: %d | Alertas: %d",
            stats['cpu_percent'],
            stats['ram_percent'],
            stats['ram_used_mb'],
            stats['active_flows'],
            stats['total_flows_evaluated'],
            stats['total_attacks_detected'],
            stats['alerts_sent']
        )


def main():
    parser = argparse.ArgumentParser(
        description="AIDS-RPi: Real-Time Network Intrusion Detection System for Raspberry Pi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--interface",
        default=os.getenv("NETWORK_INTERFACE", "eth0"),
        help="Network interface to sniff (e.g., eth0, wlan0, any)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["binary", "multiclass"],
        default=os.getenv("DETECTION_MODE", "binary").lower(),
        help="Classification mode: binary (Benign/Malicious) or multiclass (DoS, Exploits, etc.)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=float(os.getenv("ALERT_THRESHOLD", "0.5")),
        help="Probability threshold for malicious classification"
    )
    parser.add_argument(
        "-r", "--pcap",
        type=str,
        default=None,
        help="Replay a PCAP file for offline validation instead of live capture"
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        default=os.getenv("DRY_RUN", "False").strip().lower() == "true",
        help="Run without sending real emails (simulates alert output)"
    )
    parser.add_argument(
        "-c", "--cooldown",
        type=float,
        default=float(os.getenv("COOLDOWN_SECONDS", "60")),
        help="Anti-flood cooldown time in seconds per attacker/attack type"
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Test SMTP connection and send a test security alert, then exit"
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=15,
        help="Interval in seconds for status reporting"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.getenv("LOG_FILE", None),
        help="Path to write log file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()
    setup_logging(log_file=args.log_file, verbose=args.verbose)
    print_banner()

    # Handle Test Email Mode
    if args.test_email:
        logging.info("📧 Testando configurações do servidor SMTP...")
        email_manager = EmailAlertManager(cooldown_seconds=args.cooldown)
        success, msg = email_manager.test_connection()
        if success:
            logging.info("✅ Conexão SMTP bem-sucedida! Enviando e-mail de teste...")
            test_alert = {
                'attack_type': 'TESTE_SISTEMA',
                'probability': 0.999,
                'src_ip': '192.168.1.100',
                'src_port': 44444,
                'dst_ip': '192.168.1.1',
                'dst_port': 80,
                'protocol_name': 'TCP',
                'duration_sec': 0.05,
                'total_packets': 10,
                'total_bytes': 1024,
                'interface': args.interface
            }
            sent = email_manager.send_alert(test_alert, async_send=False)
            if sent:
                logging.info("✅ E-mail de teste enviado com sucesso para: %s", email_manager.recipient)
                sys.exit(0)
            else:
                logging.error("❌ Falha ao enviar e-mail de teste.")
                sys.exit(1)
        else:
            logging.error("❌ Falha na conexão SMTP: %s", msg)
            sys.exit(1)

    # Initialize Email Alert Manager
    email_manager = EmailAlertManager(
        cooldown_seconds=args.cooldown,
        enabled=not args.dry_run
    )

    # Initialize Edge Intrusion Detector
    try:
        detector = RPIDetector(
            mode=args.mode,
            interface=args.interface,
            threshold=args.threshold,
            dry_run=args.dry_run,
            email_manager=email_manager
        )
    except Exception as ex:
        logging.critical("❌ Falha ao inicializar o detector: %s", str(ex))
        sys.exit(1)

    # Setup Graceful Shutdown on SIGINT / SIGTERM
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        logging.info("🛑 Sinal recebido (%s). Finalizando monitoramento...", signal.Signals(signum).name)
        stop_event.set()
        detector.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Handle PCAP Replay Mode
    if args.pcap:
        try:
            stats = detector.replay_pcap(args.pcap)
            logging.info("🎉 Análise do PCAP concluída com sucesso!")
            sys.exit(0)
        except Exception as ex:
            logging.error("❌ Erro durante análise do PCAP: %s", str(ex))
            sys.exit(1)

    # Live Sniffing Mode
    logging.info("🚀 Iniciando serviço AIDS-RPi no Raspberry Pi...")
    logging.info("⚙️  Interface: %s | Modo: %s | Limiar: %.2f | Anti-Flood Cooldown: %ds",
                 args.interface, args.mode.upper(), args.threshold, int(args.cooldown))
    if args.dry_run:
        logging.warning("⚠️  Modo DRY-RUN ativo. Alertas por e-mail NÃO serão enviados.")

    # Start Stats Thread
    stats_thread = threading.Thread(
        target=run_stats_reporter,
        args=(detector, args.stats_interval, stop_event),
        daemon=True
    )
    stats_thread.start()

    try:
        detector.start_sniffing(iface=args.interface)
    except Exception as ex:
        logging.critical("❌ Erro na captura de pacotes: %s", str(ex))
        detector.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
