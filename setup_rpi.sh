#!/usr/bin/env bash
# ==============================================================================
# AIDS-RPi: Setup Automatizado para Raspberry Pi como IDS e Gateway de Rede
# ==============================================================================
# Este script configura o Raspberry Pi para:
#   1. Instalar dependencias de sistema (libpcap, iptables, python3-venv, etc.)
#   2. Criar e configurar o ambiente virtual Python (.venv) com as dependencias
#   3. Conceder permissoes de rede via Linux Capabilities (CAP_NET_RAW / ADMIN)
#   4. Habilitar IP Forwarding e NAT Masquerade (passando todo o trafego por ele)
#   5. Instalar e iniciar o servico systemd (systemctl) para monitoramento no boot
#
# Uso:
#   sudo bash setup_rpi.sh [INTERFACE]
#   Exemplo:
#   sudo bash setup_rpi.sh eth0
#   sudo bash setup_rpi.sh wlan0
# ==============================================================================

set -euo pipefail

# Cores para saida no terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}${BOLD}[SUCESSO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}${BOLD}[AVISO]${NC} $1"
}

error() {
    echo -e "${RED}${BOLD}[ERRO]${NC} $1" >&2
}

# ------------------------------------------------------------------------------
# 1. Verificacao de Privilegios e Identificacao de Diretorios e Usuario
# ------------------------------------------------------------------------------
echo -e "${CYAN}${BOLD}"
echo "=================================================================="
echo "    [AIDS-RPi] Configuracao de Gateway & Servico Systemd          "
echo "        Autonomous Intrusion Detection System for Edge            "
echo "=================================================================="
echo -e "${NC}"

if [ "$(id -u)" -ne 0 ]; then
    error "Este script deve ser executado como root (com sudo) para configurar rede e servicos."
    echo "Execute: sudo bash $0"
    exit 1
fi

# Detectar diretorio de origem do projeto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/rpi_monitor.py" ]; then
    CURRENT_PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../rpi_monitor.py" ]; then
    CURRENT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    CURRENT_PROJECT_DIR="$SCRIPT_DIR"
fi

# Identificar o usuario real nao-root (quem executou o sudo)
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
    REAL_USER="$SUDO_USER"
else
    if id "pi" &>/dev/null; then
        REAL_USER="pi"
    else
        REAL_USER="$(stat -c '%U' "$CURRENT_PROJECT_DIR")"
    fi
fi
REAL_GROUP="$(id -gn "$REAL_USER")"

info "Usuario de execucao do servico: ${BOLD}$REAL_USER ($REAL_GROUP)${NC}"

# ------------------------------------------------------------------------------
# 2. Migracao / Movimentacao do Projeto para /etc/aids
# ------------------------------------------------------------------------------
TARGET_DIR="${TARGET_DIR:-/etc/aids}"

if [ "$CURRENT_PROJECT_DIR" != "$TARGET_DIR" ]; then
    info "Movendo o projeto de '$CURRENT_PROJECT_DIR' para '$TARGET_DIR'..."
    mkdir -p "$TARGET_DIR"

    # Copiar todos os arquivos preservando permissoes e metadados
    cp -a "$CURRENT_PROJECT_DIR"/. "$TARGET_DIR"/

    # Ajustar propriedade para o usuario real
    chown -R "$REAL_USER:$REAL_GROUP" "$TARGET_DIR"

    # Remover o diretorio original apos a copia para efetivar a movimentacao
    ORIGIN_DIR="$CURRENT_PROJECT_DIR"
    PROJECT_DIR="$TARGET_DIR"
    cd "$PROJECT_DIR"

    # Verificacao de seguranca antes de remover o diretorio de origem
    if [ "$ORIGIN_DIR" != "/" ] && [ "$ORIGIN_DIR" != "/etc" ] && [ "$ORIGIN_DIR" != "/home" ] && [ "$ORIGIN_DIR" != "/root" ]; then
        rm -rf "$ORIGIN_DIR"
    fi

    success "Projeto movido para $PROJECT_DIR com sucesso."
else
    PROJECT_DIR="$TARGET_DIR"
    cd "$PROJECT_DIR"
    info "Projeto ja se encontra no diretorio de destino: $PROJECT_DIR"
fi

# Garantir permissoes do projeto em /etc/aids
chown -R "$REAL_USER:$REAL_GROUP" "$PROJECT_DIR"

# Detectar ou atribuir interface de rede de captura e roteamento
if [ -n "${1:-}" ]; then
    NET_IFACE="$1"
else
    # Tentar detectar interface padrao conectada a internet
    DETECTED_IFACE="$(ip route show default 2>/dev/null | awk '{print $5}' | head -n 1)"
    NET_IFACE="${DETECTED_IFACE:-eth0}"
fi

info "Interface de rede selecionada: ${BOLD}$NET_IFACE${NC}"

# Verificar se a interface existe
if ! ip link show "$NET_IFACE" &>/dev/null; then
    warn "A interface '$NET_IFACE' nao foi encontrada localmente. Verifique as interfaces com 'ip link'."
fi

# ------------------------------------------------------------------------------
# 2. Instalacao de Dependencias de Sistema Linux (apt)
# ------------------------------------------------------------------------------
info "[1/7] Atualizando lista de pacotes e instalando dependencias do sistema..."
export DEBIAN_FRONTEND=noninteractive

# Pre-configurar debconf para instalacao silenciosa do iptables-persistent sem travar o script
echo "iptables-persistent iptables-persistent/autosave_v4 boolean true" | debconf-set-selections
echo "iptables-persistent iptables-persistent/autosave_v6 boolean true" | debconf-set-selections

apt-get update -y
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libpcap-dev \
    tcpdump \
    libcap2-bin \
    iptables \
    iptables-persistent \
    netfilter-persistent \
    build-essential \
    curl \
    git

success "Dependencias do sistema Linux instaladas com sucesso."

# ------------------------------------------------------------------------------
# 3. Configuracao do Ambiente Virtual Python (.venv) e Dependencias
# ------------------------------------------------------------------------------
info "[2/7] Configurando ambiente virtual Python (.venv)..."
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Criando ambiente virtual em $VENV_DIR..."
    sudo -u "$REAL_USER" python3 -m venv "$VENV_DIR"
fi

# Atualizar pip
sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install --upgrade pip

# Instalar requisitos do modulo edge
RPI_REQ="$PROJECT_DIR/raspberry_pi/requirements.txt"
ROOT_REQ="$PROJECT_DIR/requirements.txt"

if [ -f "$RPI_REQ" ]; then
    info "Instalando bibliotecas otimizadas para Raspberry Pi ($RPI_REQ)..."
    sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install -r "$RPI_REQ"
elif [ -f "$ROOT_REQ" ]; then
    info "Instalando dependencias de $ROOT_REQ..."
    sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install -r "$ROOT_REQ"
else
    warn "Arquivo de requirements nao encontrado. Instalando bibliotecas padrao..."
    sudo -u "$REAL_USER" "$VENV_DIR/bin/pip" install scapy psutil joblib scikit-learn pandas numpy python-dotenv
fi

success "Ambiente virtual e bibliotecas instalados."

# ------------------------------------------------------------------------------
# 4. Configuracao de Privilegios Linux Capabilities (Sem necessidade de root)
# ------------------------------------------------------------------------------
info "[3/7] Atribuindo capacidades Linux de rede ao binario Python..."
PYTHON_BIN="$VENV_DIR/bin/python3"
REAL_PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"

setcap cap_net_raw,cap_net_admin=eip "$REAL_PYTHON_BIN"
success "Capacidades atribuidas a $REAL_PYTHON_BIN:"
getcap "$REAL_PYTHON_BIN"

# ------------------------------------------------------------------------------
# 5. Configuracao de Rede: Raspberry Pi como Gateway Padrao (IP Forwarding & NAT)
# ------------------------------------------------------------------------------
info "[4/7] Configurando Raspberry Pi como Gateway (passando o trafego da rede)..."

# 5.1 Habilitar IP Forwarding imediatamente no Kernel
sysctl -w net.ipv4.ip_forward=1

# 5.2 Persistir IP Forwarding para inicializacoes futuras
SYSCTL_CONF="/etc/sysctl.d/99-aids-gateway.conf"
cat <<EOF > "$SYSCTL_CONF"
# Configuracao gerada pelo instalador AIDS-RPi
net.ipv4.ip_forward=1
EOF
sysctl -p "$SYSCTL_CONF" >/dev/null 2>&1 || sysctl --system >/dev/null 2>&1

# 5.3 Configurar Regras de NAT Masquerade e FORWARD via iptables
info "Configurando regras iptables para interface $NET_IFACE..."

# NAT Masquerade (permite que dispositivos da LAN naveguem para a Internet atraves do RPi)
if ! iptables -t nat -C POSTROUTING -o "$NET_IFACE" -j MASQUERADE 2>/dev/null; then
    iptables -t nat -A POSTROUTING -o "$NET_IFACE" -j MASQUERADE
fi

# Permitir encaminhamento de trafego existente/relacionado
if ! iptables -C FORWARD -i "$NET_IFACE" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null; then
    iptables -A FORWARD -i "$NET_IFACE" -m state --state RELATED,ESTABLISHED -j ACCEPT
fi

# Permitir encaminhamento de pacotes atraves do gateway
if ! iptables -C FORWARD -j ACCEPT 2>/dev/null; then
    iptables -A FORWARD -j ACCEPT
fi

# Salvar regras de iptables para restauracao automatica no boot
netfilter-persistent save
systemctl enable netfilter-persistent

success "Roteamento e NAT Masquerade ativados e persistidos no iptables."

# ------------------------------------------------------------------------------
# 6. Preparacao dos Arquivos de Configuracao (.env), Logs e Modelos
# ------------------------------------------------------------------------------
info "[5/7] Configurando diretorios de logs e variaveis de ambiente..."

# Criar pasta de logs com permissoes adequadas
mkdir -p "$PROJECT_DIR/logs"
chown -R "$REAL_USER:$REAL_GROUP" "$PROJECT_DIR/logs"
chmod 775 "$PROJECT_DIR/logs"

# Configurar raspberry_pi/.env
RPI_ENV="$PROJECT_DIR/raspberry_pi/.env"
RPI_ENV_EXAMPLE="$PROJECT_DIR/raspberry_pi/.env-example"

if [ ! -f "$RPI_ENV" ] && [ -f "$RPI_ENV_EXAMPLE" ]; then
    info "Copiando $RPI_ENV_EXAMPLE para $RPI_ENV..."
    cp "$RPI_ENV_EXAMPLE" "$RPI_ENV"
    chown "$REAL_USER:$REAL_GROUP" "$RPI_ENV"
fi

# Ajustar interface de rede no arquivo de configuracao se existir
if [ -f "$RPI_ENV" ]; then
    sed -i "s/^NETWORK_INTERFACE=.*/NETWORK_INTERFACE=$NET_IFACE/" "$RPI_ENV"
fi

# Garantir tambem que o .env raiz exista
ROOT_ENV="$PROJECT_DIR/.env"
ROOT_ENV_EXAMPLE="$PROJECT_DIR/.env-example"
if [ ! -f "$ROOT_ENV" ]; then
    if [ -f "$ROOT_ENV_EXAMPLE" ]; then
        cp "$ROOT_ENV_EXAMPLE" "$ROOT_ENV"
    elif [ -f "$RPI_ENV" ]; then
        cp "$RPI_ENV" "$ROOT_ENV"
    fi
    [ -f "$ROOT_ENV" ] && chown "$REAL_USER:$REAL_GROUP" "$ROOT_ENV"
fi

# Verificar existencia de modelos treinados
MODEL_FOUND=false
if ls "$PROJECT_DIR"/models/*stacking*.joblib 1>/dev/null 2>&1 || ls "$PROJECT_DIR"/models/*Stacking*.joblib 1>/dev/null 2>&1; then
    MODEL_FOUND=true
fi

if [ "$MODEL_FOUND" = true ]; then
    success "Modelo de deteccao encontrado em $PROJECT_DIR/models/."
else
    warn "Nenhum modelo treinado (*.joblib) foi encontrado na pasta '$PROJECT_DIR/models/'."
    warn "Lembre-se de transferir os modelos treinados (ex: stacking_pipeline_binary.joblib) para que o IDS funcione em tempo real!"
fi

# ------------------------------------------------------------------------------
# 7. Instalacao e Ativacao do Servico Systemd (systemctl)
# ------------------------------------------------------------------------------
info "[6/7] Instalando unidade de servico systemd (/etc/systemd/system/aids-rpi.service)..."
SERVICE_FILE="/etc/systemd/system/aids-rpi.service"

cat <<EOF > "$SERVICE_FILE"
[Unit]
Description=AIDS-RPi: Autonomous Intrusion Detection System for Raspberry Pi
Documentation=https://github.com/your-org/AIDS
After=network.target network-online.target netfilter-persistent.service
Wants=network-online.target

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_GROUP
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/raspberry_pi/.env
EnvironmentFile=-$PROJECT_DIR/.env

# Execucao com Python no ambiente virtual isolado
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/rpi_monitor.py --interface $NET_IFACE

# Permissoes de rede sem precisar rodar como root
AmbientCapabilities=CAP_NET_RAW CAP_NET_ADMIN
CapabilityBoundingSet=CAP_NET_RAW CAP_NET_ADMIN

# Politicas de reinicio automatico em caso de falha
Restart=on-failure
RestartSec=5s

# Limites de recursos para protecao de hardware do Raspberry Pi
CPUQuota=150%
MemoryMax=800M
MemoryHigh=650M

# Isolamento e Seguranca do Sistema de Arquivos
ProtectSystem=full
ProtectHome=read-only
ReadWritePaths=$PROJECT_DIR/logs
PrivateTmp=true

# Logs integrados ao systemd-journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aids-rpi

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SERVICE_FILE"

info "[7/7] Recarregando systemd e habilitando servico no boot..."
systemctl daemon-reload
systemctl enable aids-rpi.service

if [ "$MODEL_FOUND" = true ]; then
    info "Iniciando o servico aids-rpi..."
    systemctl restart aids-rpi.service
    sleep 2
    if systemctl is-active --quiet aids-rpi.service; then
        success "Servico aids-rpi.service iniciado e rodando com sucesso!"
    else
        warn "O servico foi criado e habilitado, mas pode ter encontrado um aviso na inicializacao."
        warn "Verifique os logs com: journalctl -u aids-rpi.service -n 30 --no-pager"
    fi
else
    warn "O servico aids-rpi.service foi habilitado no boot, mas NAO foi iniciado agora porque o modelo em 'models/' ainda nao foi colocado."
fi

# ------------------------------------------------------------------------------
# 8. Relatorio Final e Instrucoes de Configuracao do Roteador / DHCP
# ------------------------------------------------------------------------------
RPI_IP="$(ip -4 addr show "$NET_IFACE" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1 || echo "IP_DO_RASPBERRY_PI")"

echo ""
echo -e "${GREEN}${BOLD}==================================================================${NC}"
echo -e "${GREEN}${BOLD}    [OK] Configuracao do AIDS-RPi Concluida com Sucesso!         ${NC}"
echo -e "${GREEN}${BOLD}==================================================================${NC}"
echo ""
echo -e "[+] ${BOLD}Informacoes da Instalacao:${NC}"
echo -e "  - Local do Projeto:          ${CYAN}$PROJECT_DIR${NC}"
echo -e "  - Interface Sniffer/Gateway: ${CYAN}$NET_IFACE${NC}"
echo -e "  - IP do Raspberry Pi:        ${CYAN}$RPI_IP${NC}"
echo -e "  - Servico Systemd:           ${CYAN}aids-rpi.service${NC}"
echo -e "  - Arquivo de Log do IDS:     ${CYAN}$PROJECT_DIR/logs/detections.jsonl${NC}"
echo ""
echo -e "[!] ${BOLD}PASSO IMPORTANTE: Como passar o trafego da rede pelo Raspberry Pi:${NC}"
echo -e "  Para que os dispositivos da sua rede passem pelo Raspberry Pi:"
echo -e "  1. Acesse o painel de administracao do seu roteador Wi-Fi (ex: 192.168.1.1)."
echo -e "  2. Va nas configuracoes do servidor ${BOLD}DHCP${NC}."
echo -e "  3. Altere o campo ${BOLD}Default Gateway (Gateway Padrao)${NC} para o IP deste Raspberry Pi: ${CYAN}${BOLD}$RPI_IP${NC}."
echo -e "  4. (Opcional) Fixe um IP Estatico para este Raspberry Pi nas configuracoes do roteador."
echo -e "  5. Pronto! Todo o trafego de saida da rede passara pelo Raspberry Pi antes da internet."
echo ""
echo -e "[*] ${BOLD}Comandos Uteis de Operacao e Diagnostico:${NC}"
echo -e "  - Ver status do servico:     ${CYAN}sudo systemctl status aids-rpi.service${NC}"
echo -e "  - Acompanhar logs ao vivo:   ${CYAN}sudo journalctl -u aids-rpi.service -f${NC}"
echo -e "  - Ver deteccoes de ataque:   ${CYAN}tail -f $PROJECT_DIR/logs/detections.jsonl${NC}"
echo -e "  - Reiniciar o servico:       ${CYAN}sudo systemctl restart aids-rpi.service${NC}"
echo -e "  - Parar o servico:           ${CYAN}sudo systemctl stop aids-rpi.service${NC}"
echo ""
