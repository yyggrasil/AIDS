"""
Email Alert Module for Network Intrusion Detection on Raspberry Pi.
Provides secure SMTP delivery (TLS/SSL) with rich HTML templates,
asynchronous worker dispatch, and smart cooldown / anti-flood throttling.
"""

import os
import smtplib
import socket
import logging
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("AIDS.EmailAlert")


class EmailAlertManager:
    """
    Manages automated email security alerts with anti-flood throttling,
    modern HTML formatting, and non-blocking background dispatch.
    """

    def __init__(
        self,
        smtp_host: str = None,
        smtp_port: int = None,
        smtp_user: str = None,
        smtp_pass: str = None,
        use_tls: bool = None,
        use_ssl: bool = None,
        sender: str = None,
        recipient: str = None,
        cooldown_seconds: float = None,
        enabled: bool = None
    ):
        # Configuration with environment variable fallbacks
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(smtp_port or os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_pass = smtp_pass or os.getenv("SMTP_PASS", "")
        
        env_tls = os.getenv("SMTP_USE_TLS", "True").strip().lower() == "true"
        self.use_tls = use_tls if use_tls is not None else env_tls
        
        env_ssl = os.getenv("SMTP_USE_SSL", "False").strip().lower() == "true" or self.smtp_port == 465
        self.use_ssl = use_ssl if use_ssl is not None else env_ssl

        self.sender = sender or os.getenv("ALERT_SENDER", self.smtp_user or "aids-rpi@antigravity.local")
        self.recipient = recipient or os.getenv("ALERT_RECIPIENT", "")
        
        env_cooldown = float(os.getenv("COOLDOWN_SECONDS", "60"))
        self.cooldown_seconds = float(cooldown_seconds if cooldown_seconds is not None else env_cooldown)

        env_enabled = os.getenv("ALERT_EMAIL_ENABLED", "True").strip().lower() == "true"
        self.enabled = enabled if enabled is not None else env_enabled

        # Anti-flood tracking: key (e.g. src_ip) -> dict(last_sent=float, suppressed_count=int)
        self._lock = threading.Lock()
        self._flood_history = {}
        self.hostname = socket.gethostname()
        self.total_alerts_sent = 0
        self.total_alerts_suppressed = 0

    def is_configured(self) -> bool:
        """Checks if minimal SMTP configuration is present."""
        return bool(self.smtp_host and self.recipient and (self.smtp_user or self.smtp_port == 25))

    def should_alert(self, key: str, now: float = None) -> tuple[bool, int]:
        """
        Evaluates throttling for a specific key (e.g. attacker IP or attack type).
        Returns (should_send: bool, suppressed_count: int).
        """
        current_time = now if now is not None else time.time()
        with self._lock:
            if key not in self._flood_history:
                self._flood_history[key] = {
                    'last_sent': current_time,
                    'suppressed_count': 0
                }
                return True, 0

            record = self._flood_history[key]
            elapsed = current_time - record['last_sent']
            if elapsed >= self.cooldown_seconds:
                suppressed = record['suppressed_count']
                record['last_sent'] = current_time
                record['suppressed_count'] = 0
                return True, suppressed
            else:
                record['suppressed_count'] += 1
                self.total_alerts_suppressed += 1
                return False, record['suppressed_count']

    def format_alert_content(self, alert_data: dict, suppressed_count: int = 0) -> tuple[str, str]:
        """
        Formats alert data into plain text and modern HTML email bodies.
        """
        attack_type = alert_data.get('attack_type', 'Maligno (Desconhecido)')
        prob = alert_data.get('probability', 1.0) * 100
        src_ip = alert_data.get('src_ip', '0.0.0.0')
        src_port = alert_data.get('src_port', 0)
        dst_ip = alert_data.get('dst_ip', '0.0.0.0')
        dst_port = alert_data.get('dst_port', 0)
        protocol = alert_data.get('protocol_name', 'TCP/UDP')
        duration = alert_data.get('duration_sec', 0.0)
        total_packets = alert_data.get('total_packets', 0)
        total_bytes = alert_data.get('total_bytes', 0)
        interface = alert_data.get('interface', 'eth0/wlan0')
        ts_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        suppression_text = ""
        suppression_html = ""
        if suppressed_count > 0:
            suppression_text = f"\n[AVISO ANTI-FLOOD]: {suppressed_count} evento(s) semelhante(s) foram agregados durante a janela de cooldown de {int(self.cooldown_seconds)}s.\n"
            suppression_html = f"""
            <div style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 6px; padding: 12px; margin: 15px 0;">
                <strong>⚡ Anti-Flood / Throttling Ativo:</strong> {suppressed_count} alerta(s) semelhante(s) desta origem foram suprimidos/agregados durante os últimos {int(self.cooldown_seconds)} segundos.
            </div>
            """

        # Plain text
        text_body = f"""=====================================================
🚨 [AIDS-RPi] ALERTA DE INTRUSÃO DETECTADA
=====================================================
Dispositivo : Raspberry Pi ({self.hostname})
Interface   : {interface}
Timestamp   : {ts_str}

DETALHES DO ATAQUE:
-----------------------------------------------------
Classificação : {attack_type}
Confiança     : {prob:.2f}%
Origem        : {src_ip}:{src_port}
Destino       : {dst_ip}:{dst_port}
Protocolo     : {protocol}

ESTATÍSTICAS DO FLUXO:
-----------------------------------------------------
Duração Total : {duration:.4f} s
Total Pacotes : {total_packets}
Total Volume  : {total_bytes:,} bytes
{suppression_text}
Ação Recomendada: Verificar regras de firewall (iptables/nftables) para o IP {src_ip}.
=====================================================
"""

        # HTML Body
        html_body = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; }}
        .card {{ max-width: 620px; background: #ffffff; margin: 0 auto; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-top: 5px solid #e53e3e; }}
        .header {{ background-color: #1a202c; color: #ffffff; padding: 20px; text-align: center; }}
        .header h2 {{ margin: 0; font-size: 20px; color: #ff4d4f; letter-spacing: 0.5px; }}
        .header p {{ margin: 5px 0 0 0; font-size: 13px; color: #a0aec0; }}
        .content {{ padding: 25px; color: #2d3748; line-height: 1.5; }}
        .badge {{ display: inline-block; background-color: #fee2e2; color: #b91c1c; font-weight: bold; padding: 4px 10px; border-radius: 4px; font-size: 14px; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 15px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 14px; }}
        th {{ background-color: #f7fafc; color: #4a5568; font-weight: 600; width: 38%; }}
        .footer {{ background-color: #f7fafc; padding: 15px; text-align: center; font-size: 12px; color: #718096; border-top: 1px solid #e2e8f0; }}
        .highlight {{ color: #e53e3e; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h2>🚨 AIDS-RPi: Intrusão Detectada</h2>
            <p>Monitoramento Edge em Tempo Real • Raspberry Pi ({self.hostname})</p>
        </div>
        <div class="content">
            <div class="badge">Ameaça Detectada: {attack_type}</div>
            {suppression_html}
            <table>
                <tr>
                    <th>Classificação</th>
                    <td><span class="highlight">{attack_type}</span> (Confiança: <strong>{prob:.2f}%</strong>)</td>
                </tr>
                <tr>
                    <th>IP / Porta Origem</th>
                    <td><strong>{src_ip}</strong> : {src_port}</td>
                </tr>
                <tr>
                    <th>IP / Porta Destino</th>
                    <td><strong>{dst_ip}</strong> : {dst_port}</td>
                </tr>
                <tr>
                    <th>Protocolo</th>
                    <td>{protocol}</td>
                </tr>
                <tr>
                    <th>Duração do Fluxo</th>
                    <td>{duration:.4f} s</td>
                </tr>
                <tr>
                    <th>Pacotes / Volume</th>
                    <td>{total_packets} pacotes ({total_bytes:,} bytes)</td>
                </tr>
                <tr>
                    <th>Interface / Timestamp</th>
                    <td>{interface} • {ts_str}</td>
                </tr>
            </table>
            <p style="font-size: 13px; color: #4a5568;">
                🔒 <strong>Recomendação:</strong> Inspecione logs de rede ou aplique bloqueio no firewall:
                <code style="background: #edf2f7; padding: 2px 6px; border-radius: 4px;">sudo iptables -A INPUT -s {src_ip} -j DROP</code>
            </p>
        </div>
        <div class="footer">
            Sistema Autônomo de Detecção de Intrusão (AIDS) • Stacking Classifier Ensemble
        </div>
    </div>
</body>
</html>
"""
        return text_body, html_body

    def send_alert(self, alert_data: dict, async_send: bool = True) -> bool:
        """
        Sends an email alert. If async_send=True, runs on a separate worker thread
        to prevent blocking network packet capture.
        """
        if not self.enabled:
            logger.debug("Alerta ignorado: envio de e-mails desabilitado (ALERT_EMAIL_ENABLED=False).")
            return False

        src_ip = alert_data.get('src_ip', 'unknown')
        attack_type = alert_data.get('attack_type', 'Malicious')
        throttle_key = f"{src_ip}_{attack_type}"

        should_send, suppressed_count = self.should_alert(throttle_key)
        if not should_send:
            logger.info("Alerta throttled para %s (cooldown %ds ativo).", throttle_key, self.cooldown_seconds)
            return False

        if async_send:
            thread = threading.Thread(
                target=self._send_smtp_payload,
                args=(alert_data, suppressed_count),
                daemon=True
            )
            thread.start()
            return True
        else:
            return self._send_smtp_payload(alert_data, suppressed_count)

    def _send_smtp_payload(self, alert_data: dict, suppressed_count: int = 0) -> bool:
        """Performs actual SMTP connection and email delivery."""
        if not self.recipient or not self.smtp_host:
            logger.warning("Alerta não enviado: Destinatário ou Host SMTP não configurado.")
            return False

        attack_type = alert_data.get('attack_type', 'Intrusão')
        src_ip = alert_data.get('src_ip', '0.0.0.0')
        subject = f"🚨 [AIDS-RPi] Alerta de Segurança: {attack_type} detectado de {src_ip}"

        text_content, html_content = self.format_alert_content(alert_data, suppressed_count)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = self.recipient
        msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S -0000")

        msg.attach(MIMEText(text_content, "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        try:
            logger.info("Conectando ao servidor SMTP %s:%d...", self.smtp_host, self.smtp_port)
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=15)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15)
                if self.use_tls:
                    server.starttls()

            if self.smtp_user and self.smtp_pass:
                server.login(self.smtp_user, self.smtp_pass)

            recipients = [r.strip() for r in self.recipient.split(",") if r.strip()]
            server.sendmail(self.sender, recipients, msg.as_string())
            server.quit()

            with self._lock:
                self.total_alerts_sent += 1

            logger.info("✅ Alerta de e-mail enviado com sucesso para %s!", self.recipient)
            return True
        except Exception as ex:
            logger.error("❌ Falha ao enviar alerta por e-mail: %s", str(ex))
            return False

    def test_connection(self) -> tuple[bool, str]:
        """Tests connection and authentication with the SMTP server."""
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=10)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10)
                if self.use_tls:
                    server.starttls()

            if self.smtp_user and self.smtp_pass:
                server.login(self.smtp_user, self.smtp_pass)

            server.quit()
            return True, "Conexão SMTP e autenticação bem-sucedidas!"
        except Exception as ex:
            return False, f"Erro ao conectar com servidor SMTP: {str(ex)}"
