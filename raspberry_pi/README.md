# 🍓 Guia de Edge Deployment e Monitoramento de Rede em Tempo Real no Raspberry Pi (AIDS-RPi)

Este guia fornece as instruções completas, passo a passo, para implantar e executar em produção o **Sistema Autônomo de Detecção de Intrusão em Redes (AIDS)** diretamente em um **Raspberry Pi** (modelos 3B+, 4B, 5 ou Zero 2 W) utilizando o modelo **Stacking Ensemble Classifier** treinado e alertas automáticos por e-mail via SMTP seguro com proteção anti-flood.

---

## 📋 Arquitetura da Solução Edge

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    Tráfego de Rede (LAN / Wi-Fi)            │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
                     [Scapy Sniffer em Tempo Real]
                      (Interface: eth0 / wlan0)
                                 │
                                 ▼
                    [Flow Aggregator Bidirecional]
                 - Agrupamento por 5-tuple
                 - Timeout de Inatividade & Atividade
                 - Extração de 70 Features (CICFlowMeter)
                 - Gestão de Memória Otimizada para RPi
                                 │
                                 ▼
                   [Stacking Pipeline Classifier]
                 - Preprocessor (RobustScaler + OneHot)
                 - Base Learners: LinearSVC, RF, ET, DT
                 - Meta-Learner: Regressão Logística
                                 │
                                 ▼
                     [Maligno? Probabilidade >= 0.50]
                      /                            \
                    SIM                            NÃO
                   /   \                             \
                  /     \                             \
  [Email Alert Manager]  [Detection Logger]      [Descarte / Log]
  - Throttling/Cooldown  - JSONL / CSV / Text
  - Disparo Assíncrono   - Rotação Automática
  - HTML + Texto Puro    - Auditoria Forense / SIEM
```

---

## 🛠️ 1. Pré-Requisitos e Dependências do Sistema

Recomenda-se o **Raspberry Pi OS (64-bit)** baseado no Debian 12 (Bookworm) ou Debian 11 (Bullseye).

### 1.1 Atualizar o Sistema e Instalar Pacotes Essenciais
Abra o terminal no Raspberry Pi e execute:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv \
                    libpcap-dev tcpdump libcap2-bin \
                    build-essential git
```

---

## 📦 2. Clonagem e Configuração do Ambiente Virtual

### 2.1 Clonar o Repositório
```bash
cd /home/pi
git clone https://github.com/your-org/AIDS.git
cd AIDS
```

### 2.2 Criar e Ativar o Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 Instalar Dependências Python
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔒 3. Execução com Privilégios Mínimos (Segurança Linux Capabilities)

Para capturar pacotes em modo promíscuo ou criar raw sockets **sem precisar rodar o Python como root (`sudo`)**, conceda as capacidades Linux `CAP_NET_RAW` e `CAP_NET_ADMIN` ao binário do Python:

```bash
# Obter o caminho exato do Python no ambiente virtual ou do sistema
PYTHON_PATH=$(readlink -f $(which python3))

# Conceder permissões de captura de pacotes
sudo setcap cap_net_raw,cap_net_admin=eip "$PYTHON_PATH"

# Verificar se a permissão foi atribuída corretamente
getcap "$PYTHON_PATH"
```

---

## ⚙️ 4. Configuração das Variáveis de Ambiente no Raspberry Pi 3 (`raspberry_pi/.env`)

O projeto utiliza **dois arquivos de configuração isolados**:
- **Raiz do projeto (`.env`)**: configura exclusivamente o treinamento dos modelos de Machine Learning (`main.py`).
- **Pasta `raspberry_pi/.env`**: configura exclusivamente o monitoramento em tempo real, engine de detecção, alertas SMTP e logs no Raspberry Pi 3.

Para configurar no Raspberry Pi 3, copie o arquivo de exemplo dedicado dentro de `raspberry_pi/`:

```bash
cd raspberry_pi
cp .env-example .env
nano .env
```

### Exemplo de Configuração para Raspberry Pi 3 (`raspberry_pi/.env`):

```ini
# --- Configuração de Rede no Raspberry Pi ---
NETWORK_INTERFACE=eth0          # ou wlan0 para Wi-Fi, end0 no RPi 5
DETECTION_MODE=binary           # binary (Benign/Malicious) ou multiclass
ALERT_THRESHOLD=0.50            # Limiar de probabilidade de ataque (0.0 a 1.0)
FLOW_INACTIVITY_TIMEOUT=10.0    # Segundos sem pacotes para fechar fluxo
FLOW_ACTIVE_TIMEOUT=60.0        # Duração máxima de fluxo contínuo
DRY_RUN=False                   # False para enviar e-mails reais

# --- Alertas por E-mail (SMTP Seguro) ---
ALERT_EMAIL_ENABLED=True
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=True
SMTP_USE_SSL=False
SMTP_USER=seu_email@gmail.com
SMTP_PASS=sua_senha_de_aplicativo_16_digitos
ALERT_SENDER=aids-rpi@antigravity.local
ALERT_RECIPIENT=soc@empresa.com,admin@empresa.com

# --- Proteção Anti-Flood / Cooldown ---
COOLDOWN_SECONDS=60.0           # Intervalo mínimo entre e-mails do mesmo IP

# --- Log de Pacotes e Intrusões Detectadas (Audit Trail) ---
DETECTION_LOG_ENABLED=True      # Gravação persistente em arquivo
DETECTION_LOG_FILE=logs/detections.jsonl # Caminho do arquivo de log
DETECTION_LOG_FORMAT=jsonl      # jsonl (SIEM), csv (planilhas), text (syslog) ou all
DETECTION_LOG_MAX_BYTES=10485760 # Rotação a cada 10 MB para poupar o cartão SD
DETECTION_LOG_BACKUP_COUNT=5    # Manter até 5 arquivos rotacionados (.1, .2, ...)
DETECTION_LOG_ALL_FLOWS=False   # True para logar todos os fluxos, False apenas ataques
```

> 💡 **Nota para Gmail**: Utilize uma **Senha de Aplicativo (App Password)** de 16 caracteres gerada em *Conta Google > Segurança > Verificação em duas etapas > Senhas de app*.

---

## 🧪 5. Testando a Conexão SMTP e Alertas

Valide se o Raspberry Pi consegue autenticar no servidor SMTP e enviar alertas com o comando:

```bash
python3 rpi_monitor.py --test-email
```

Caso o e-mail seja recebido na caixa de entrada, a integração SMTP está pronta para operação em tempo real.

---

## 🚀 6. Execução Manual e Modos de Uso

### 6.1 Modo Detecção em Tempo Real na Interface Principal
Inicia a captura, classifica com o modelo Stacking, envia alertas de e-mail e grava detecções em `logs/detections.jsonl`:
```bash
python3 rpi_monitor.py --interface eth0 --mode binary
```

### 6.2 Modo Simulação / Teste Seguro (DRY-RUN)
Executa a captura e inferência, **grava todas as detecções no log persistente**, mas simula o envio de e-mails:
```bash
python3 rpi_monitor.py --interface eth0 --dry-run
```

### 6.3 Modo Multiclasse (Identificação de Ataques Específicos)
Identifica categorias como *DoS, Exploits, Reconnaissance, Fuzzers, Worms, etc.* e grava o tipo exato no log:
```bash
python3 rpi_monitor.py --interface eth0 --mode multiclass
```

### 6.4 Validação Offline com Arquivo PCAP
Reproduz um arquivo PCAP capturado previamente:
```bash
python3 rpi_monitor.py --pcap data/sample_traffic.pcap --mode binary
```

### 6.5 Personalização do Log de Detecções (CSV / JSONL / Todos os Fluxos)
Você pode escolher o formato e o arquivo de saída diretamente via linha de comando:

```bash
# Gravar em formato CSV (ideal para Excel, LibreOffice e Pandas):
python3 rpi_monitor.py --interface eth0 --detection-log logs/detections.csv --detection-log-format csv

# Gravar em ambos os formatos simultaneamente (JSONL + CSV):
python3 rpi_monitor.py --interface eth0 --detection-log logs/detections.jsonl --detection-log-format all

# Gravar todos os fluxos avaliados (benignos e ataques) para auditoria completa:
python3 rpi_monitor.py --interface eth0 --log-all-flows

# Desativar a gravação de logs em arquivo se necessário:
python3 rpi_monitor.py --interface eth0 --no-detection-log
```

---

## 🔄 7. Inicialização Automática no Boot (Serviço Systemd)

Para que o monitor de intrusão inicie automaticamente sempre que o Raspberry Pi ligar:

### 7.1 Copiar o Arquivo de Serviço
```bash
sudo cp raspberry_pi/aids-rpi.service /etc/systemd/system/
```

### 7.2 Ajustar Caminhos e Usuário (se necessário)
Verifique se os caminhos no arquivo correspondem ao seu usuário:
```bash
sudo nano /etc/systemd/system/aids-rpi.service
```

### 7.3 Recarregar o Systemd e Iniciar o Serviço
```bash
sudo systemctl daemon-reload
sudo systemctl enable aids-rpi.service
sudo systemctl start aids-rpi.service
```

### 7.4 Verificar Status do Serviço
```bash
sudo systemctl status aids-rpi.service
```

### 7.5 Acompanhar Logs em Tempo Real
```bash
journalctl -u aids-rpi.service -f
```

---

## ⚡ 8. Dicas de Otimização de Performance no Raspberry Pi

1. **Ativar zram (Compressão de RAM)**:
   ```bash
   sudo apt install -y zram-tools
   ```
2. **Governor de CPU para Alta Performance**:
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```
3. **Limitação de Memória no Aggregator**:
   O `FlowAggregator` descarta automaticamente fluxos antigos se atingir `10.000` fluxos simultâneos, garantindo consumo de RAM estável (< 400 MB).

---

## 🧪 9. Execução de Testes Automatizados

Para rodar todos os testes unitários do sistema no Raspberry Pi:

```bash
python3 -m unittest discover tests
```

---

## 🌐 10. Configuração de Roteamento da Rede Doméstica (Método 1 - Gateway Padrão)

Para que todo o tráfego da rede doméstica passe pelo Raspberry Pi antes de sair para a internet:

1. **Fixar IP Estático no Raspberry Pi** (ex: `192.168.1.2`).
2. **Habilitar IP Forwarding:**
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   ```
3. **Configurar NAT no Raspberry Pi:**
   ```bash
   sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
   sudo iptables -A FORWARD -i eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
   sudo iptables -A FORWARD -j ACCEPT
   ```
4. **Atualizar Roteador Doméstico:** No servidor DHCP do seu roteador Wi-Fi, altere o **Default Gateway** para o IP do Raspberry Pi (`192.168.1.2`).

