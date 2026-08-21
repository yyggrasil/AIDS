# AIDS - Alternative Traffic Detection

Este projeto consiste em um Sistema de Detecção de Intrusão Baseado em Anomalias e Assinaturas (**AIDS - Anomaly-based Intrusion Detection System**). O objetivo principal é classificar pacotes e fluxos de tráfego de rede entre **Maligno** (ataques como DoS/DDoS, Port Scan, Botnet, Brute Force, etc.) e **Benigno** (tráfego legítimo de rede).

A solução emprega um algoritmo de **Stacking Ensemble (Aprendizado em Camadas)** otimizado com `passthrough=True` e validação cruzada estratificada, combinando as capacidades complementares do **LinearSVC** (alta velocidade linear), **Extra Trees (ET)**, **HistGradientBoosting (HGB)** e **Decision Tree (DT)**, utilizando uma **Regressão Logística com Validação Cruzada (LogisticRegressionCV)** como meta-classificador.

## Estrutura do Projeto

```text
AIDS_alternativo/
├── data/
│   └── CICFlowMeter_out.csv    # Dataset principal de tráfego (84 colunas)
├── models/                     # Diretório onde os modelos (.joblib) são salvos
├── results/                    # Gráficos de avaliação salvos após os testes
├── specs/                      # Especificações e arquitetura do projeto
├── src/                        # Código-fonte principal da pipeline
│   ├── data_loader.py          # Lógica de carregamento de dados e amostragem
│   ├── preprocessing.py        # Limpeza, Scalers (Standard/Robust) e seleção de atributos
│   ├── models.py               # Algoritmos Base e Stacking Ensemble
│   └── evaluation.py           # Cálculo de métricas e geração de gráficos (Matplotlib/Seaborn)
├── .env                        # Configurações dinâmicas para execução (toggles)
├── requirements.txt            # Dependências Python do projeto
└── main.py                     # Script orquestrador principal
```

## Pré-Requisitos e Instalação

É **obrigatório** o uso de um ambiente virtual isolado devido ao manuseio rigoroso das dependências de *Machine Learning*.

1. **Crie e ative o ambiente virtual:**
   *Windows (PowerShell):*
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   *Linux/macOS:*
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuração (.env)

O orquestrador `main.py` utiliza o arquivo `.env` localizado na raiz do projeto para controlar dinamicamente a execução.

Abra o arquivo `.env` para ligar ou desligar os algoritmos e modos de teste desejados:

```env
# Quais modelos base treinar?
TRAIN_LINEARSVC=True
TRAIN_DT=True
TRAIN_RF=True

# Treinar o Stacking Ensemble?
TRAIN_STACKING=True

# Modos de Treinamento a executar no main.py
RUN_BINARY=True
RUN_MULTICLASS=False

# Fração de amostragem do dataset (0.01 = 1%, 0.1 = 10%, 1.0 = 100%)
SAMPLE_FRAC=0.01
```

## Executando a Pipeline

Após configurar as opções no seu arquivo `.env`, execute o orquestrador:

```bash
python main.py
```

O script realizará automaticamente:
1. Leitura e descarte inteligente de dados super-dimensionados e de variância nula.
2. Construção e Treinamento do **Stacking Ensemble**.
3. Testes e persistência (pasta `models/`).
4. Geração e avaliação gráfica (pasta `results/`).

## Gráficos de Resultados

Para cada modo ativado (Binário ou Multiclasse), as seguintes imagens `.png` são automaticamente construídas com `matplotlib` e `seaborn` na pasta `./results/`:
- `metrics_comparison_*.png`: Gráfico de barras comparando a Eficiência (*Acurácia, F1-Score, Precisão, Recall*) dos algoritmos base contra o Stacking.
- `roc_curves_*.png`: Comparativo de Taxas Verdadeiros vs Falsos Positivos via *Curva ROC*.
- `confusion_matrices_*.png`: Matrizes de Confusão com rótulos explícitos das categorias (`Benigno` vs `Maligno`).
- `latency_vs_f1_*.png`: Relatório do Trade-off computacional, mostrando velocidade de predição *versus* Qualidade Analítica (*F1-Score*).
- `svm_feature_weights_*.png`: Gráfico de importância dos coeficientes (pesos) do modelo LinearSVC.

---

## Dicionário de Colunas e Atributos do Dataset (CICFlowMeter)

O dataset gerado pela ferramenta **CICFlowMeter** extrai 84 métricas estatísticas e temporais a partir de capturas de pacotes de rede (PCAP). A tabela abaixo detalha o significado e a função de cada coluna:

### 1. Identificadores e Metadados do Fluxo
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Flow ID` | Identificador único da conexão/fluxo no formato `IP_Origem-IP_Destino-Porta_Origem-Porta_Destino-Protocolo`. |
| `Src IP` | Endereço IP da máquina de origem (remetente). |
| `Src Port` | Porta lógica de origem (TCP/UDP). |
| `Dst IP` | Endereço IP da máquina de destino (destinatário). |
| `Dst Port` | Porta lógica de destino (ex: 80 para HTTP, 443 para HTTPS, 22 para SSH). |
| `Protocol` | Número do protocolo de transporte IP (ex: 6 para TCP, 17 para UDP). |
| `Timestamp` | Data e horário em que o fluxo foi registrado. |

---

### 2. Métricas Temporais e Duração do Fluxo
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Flow Duration` | Duração total do fluxo (em microssegundos, µs). |
| `Flow IAT Mean` | Média do tempo entre a chegada de dois pacotes consecutivos no fluxo (*Inter-Arrival Time*). |
| `Flow IAT Std` | Desvio padrão do tempo entre chegada de pacotes no fluxo. |
| `Flow IAT Max` | Maior intervalo de tempo observado entre dois pacotes consecutivos no fluxo. |
| `Flow IAT Min` | Menor intervalo de tempo observado entre dois pacotes consecutivos no fluxo. |
| `Fwd IAT Total` | Tempo total decorrido entre o primeiro e o último pacote enviado na direção direta (*Forward*). |
| `Fwd IAT Mean` | Média do tempo entre chegadas de pacotes na direção direta (*Forward*). |
| `Fwd IAT Std` | Desvio padrão do tempo entre chegadas de pacotes na direção direta. |
| `Fwd IAT Max` | Maior intervalo de tempo entre pacotes na direção direta. |
| `Fwd IAT Min` | Menor intervalo de tempo entre pacotes na direção direta. |
| `Bwd IAT Total` | Tempo total decorrido entre o primeiro e o último pacote recebido na direção reversa (*Backward*). |
| `Bwd IAT Mean` | Média do tempo entre chegadas de pacotes na direção reversa (*Backward*). |
| `Bwd IAT Std` | Desvio padrão do tempo entre chegadas de pacotes na direção reversa. |
| `Bwd IAT Max` | Maior intervalo de tempo entre pacotes na direção reversa. |
| `Bwd IAT Min` | Menor intervalo de tempo entre pacotes na direção reversa. |

---

### 3. Contagem e Tamanhos de Pacotes (Volumes e Estatísticas)
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Total Fwd Packet` | Número total de pacotes transmitidos na direção direta (*Forward*). |
| `Total Bwd packets` | Número total de pacotes recebidos na direção reversa (*Backward*). |
| `Total Length of Fwd Packet` | Volume total de dados (em bytes) transmitidos na direção direta. |
| `Total Length of Bwd Packet` | Volume total de dados (em bytes) recebidos na direção reversa. |
| `Fwd Packet Length Max` | Maior tamanho de pacote (em bytes) observado na direção direta. |
| `Fwd Packet Length Min` | Menor tamanho de pacote (em bytes) observado na direção direta. |
| `Fwd Packet Length Mean` | Tamanho médio dos pacotes na direção direta. |
| `Fwd Packet Length Std` | Desvio padrão dos tamanhos dos pacotes na direção direta. |
| `Bwd Packet Length Max` | Maior tamanho de pacote (em bytes) observado na direção reversa. |
| `Bwd Packet Length Min` | Menor tamanho de pacote (em bytes) observado na direção reversa. |
| `Bwd Packet Length Mean` | Tamanho médio dos pacotes na direção reversa. |
| `Bwd Packet Length Std` | Desvio padrão dos tamanhos dos pacotes na direção reversa. |
| `Packet Length Min` | Menor tamanho de pacote (em bytes) registrado em todo o fluxo. |
| `Packet Length Max` | Maior tamanho de pacote (em bytes) registrado em todo o fluxo. |
| `Packet Length Mean` | Tamanho médio geral dos pacotes no fluxo. |
| `Packet Length Std` | Desvio padrão do tamanho dos pacotes no fluxo. |
| `Packet Length Variance` | Variância do tamanho dos pacotes no fluxo. |
| `Average Packet Size` | Tamanho médio absoluto por pacote no fluxo. |
| `Fwd Segment Size Avg` | Tamanho médio de segmento de dados por pacote no sentido direto. |
| `Bwd Segment Size Avg` | Tamanho médio de segmento de dados por pacote no sentido reverso. |
| `Fwd Seg Size Min` | Tamanho mínimo do cabeçalho/segmento TCP na direção direta. |

---

### 4. Taxas e Vazão de Transferência
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Flow Bytes/s` | Vazão de dados do fluxo em bytes por segundo (Bytes/s). |
| `Flow Packets/s` | Taxa de transmissão de pacotes por segundo no fluxo (Pacotes/s). |
| `Fwd Packets/s` | Taxa de pacotes por segundo enviados na direção direta. |
| `Bwd Packets/s` | Taxa de pacotes por segundo recebidos na direção reversa. |
| `Down/Up Ratio` | Razão de proporção entre pacotes de download (*Backward*) e upload (*Forward*). |

---

### 5. Flags de Controle do Cabeçalho TCP
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Fwd PSH Flags` | Número de vezes que a flag PSH (Push) foi ativada na direção direta. |
| `Bwd PSH Flags` | Número de vezes que a flag PSH foi ativada na direção reversa. |
| `Fwd URG Flags` | Número de vezes que a flag URG (Urgent) foi ativada na direção direta. |
| `Bwd URG Flags` | Número de vezes que a flag URG foi ativada na direção reversa. |
| `FIN Flag Count` | Contagem total de pacotes com a flag FIN (Finalização de conexão). |
| `SYN Flag Count` | Contagem total de pacotes com a flag SYN (Solicitação de abertura de conexão). |
| `RST Flag Count` | Contagem total de pacotes com a flag RST (Redefinição/Resetação de conexão). |
| `PSH Flag Count` | Contagem total de pacotes com a flag PSH (Envio imediato sem buffer). |
| `ACK Flag Count` | Contagem total de pacotes com a flag ACK (Confirmação de recebimento). |
| `URG Flag Count` | Contagem total de pacotes com a flag URG (Dados urgentes). |
| `CWR Flag Count` | Contagem total de pacotes com a flag CWR (*Congestion Window Reduced*). |
| `ECE Flag Count` | Contagem total de pacotes com a flag ECE (*ECN-Echo* - Notificação de congestionamento). |

---

### 6. Tamanhos de Cabeçalho e Janela TCP
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Fwd Header Length` | Tamanho total acumulado (em bytes) dos cabeçalhos dos pacotes na direção direta. |
| `Bwd Header Length` | Tamanho total acumulado (em bytes) dos cabeçalhos dos pacotes na direção reversa. |
| `FWD Init Win Bytes` | Tamanho inicial da janela TCP oferecida no primeiro pacote na direção direta. |
| `Bwd Init Win Bytes` | Tamanho inicial da janela TCP oferecida no primeiro pacote na direção reversa. |
| `Fwd Act Data Pkts` | Quantidade de pacotes enviados na direção direta contendo pelo menos 1 byte de carga útil (*payload*). |

---

### 7. Métricas de Transferência em Lote (Bulk Metrics)
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Fwd Bytes/Bulk Avg` | Média de bytes por rajada de transferência em lote na direção direta. |
| `Fwd Packet/Bulk Avg` | Média de pacotes por rajada de transferência em lote na direção direta. |
| `Fwd Bulk Rate Avg` | Taxa média de transferência durante rajadas de lote na direção direta. |
| `Bwd Bytes/Bulk Avg` | Média de bytes por rajada de transferência em lote na direção reversa. |
| `Bwd Packet/Bulk Avg` | Média de pacotes por rajada de transferência em lote na direção reversa. |
| `Bwd Bulk Rate Avg` | Taxa média de transferência durante rajadas de lote na direção reversa. |

---

### 8. Métricas de Subfluxos (Subflows)
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Subflow Fwd Packets` | Número médio de pacotes por subfluxo na direção direta. |
| `Subflow Fwd Bytes` | Número médio de bytes por subfluxo na direção direta. |
| `Subflow Bwd Packets` | Número médio de pacotes por subfluxo na direção reversa. |
| `Subflow Bwd Bytes` | Número médio de bytes por subfluxo na direção reversa. |

---

### 9. Métricas de Atividade e Inatividade (Active / Idle Times)
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Active Mean` | Tempo médio em que o fluxo permaneceu ativo transmitindo pacotes antes de entrar em repouso. |
| `Active Std` | Desvio padrão do tempo de atividade do fluxo. |
| `Active Max` | Tempo máximo contínuo em que o fluxo permaneceu ativo. |
| `Active Min` | Tempo mínimo em que o fluxo permaneceu ativo. |
| `Idle Mean` | Tempo médio em que o fluxo permaneceu inativo (ocioso) sem enviar dados. |
| `Idle Std` | Desvio padrão do tempo de ociosidade do fluxo. |
| `Idle Max` | Tempo máximo em que o fluxo permaneceu ocioso. |
| `Idle Min` | Tempo mínimo em que o fluxo permaneceu ocioso. |

---

### 10. Classe Alvo (Target Label)
| Coluna | Descrição / Significado |
| :--- | :--- |
| `Label` | Categoria de tráfego. No modo **Binário**, valores diferentes de `BENIGN` são mapeados para `1` (**Maligno/Ataque**) e `BENIGN` para `0` (**Benigno**). No modo **Multiclasse**, preserva o nome do ataque original (ex: `DoS Hulk`, `PortScan`, `DDoS`, `Bot`, `FTP-Patator`, `SSH-Patator`, `DoSSlowloris`, etc.). |
