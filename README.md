# AIDS - Alternative Traffic Detection

Este projeto consiste em um Sistema de Detecção de Intrusão Baseado em Anomalias e Assinaturas (**AIDS - Anomaly-based Intrusion Detection System**). O objetivo principal é classificar pacotes e fluxos de tráfego de rede entre **Maligno** (ataques como DoS/DDoS, Port Scan, Botnet, Brute Force, etc.) e **Benigno** (tráfego legítimo de rede).

A solução emprega um algoritmo de **Stacking Ensemble (Aprendizado em Camadas)**, combinando as capacidades complementares do **LinearSVC** (alta velocidade linear) e do **Random Forest** (robustez a ruídos), utilizando uma **Regressão Logística** como meta-classificador.

## 🚀 Estrutura do Projeto

```text
AIDS_alternativo/
├── data/
│   └── CICFlowMeter_out.csv    # Dataset principal de tráfego
├── models/                     # Diretório onde os modelos (.joblib) serão salvos
├── results/                    # Gráficos de avaliação salvos após os testes
├── specs/                      # Especificações e arquitetura do projeto
├── src/                        # Código-fonte principal da pipeline
│   ├── data_loader.py          # Lógica de carregamento de dados e amostragem
│   ├── preprocessing.py        # Limpeza, Scalers (Standard/Robust) e SMOTE
│   ├── models.py               # Algoritmos Base e Stacking Ensemble
│   └── evaluation.py           # Cálculo de métricas e geração de gráficos (Matplotlib/Seaborn)
├── .env                        # Configurações dinâmicas para execução (toggles)
├── requirements.txt            # Dependências Python do projeto
└── main.py                     # Script orquestrador principal
```

## ⚙️ Pré-Requisitos e Instalação

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

## 🛠️ Configuração (.env)

O orquestrador `main.py` utiliza o arquivo `.env` localizado na raiz do projeto para controlar dinamicamente a execução.

Abra o arquivo `.env` para ligar ou desligar os algoritmos e modos de teste desejados:

```env
# Quais modelos base treinar?
TRAIN_LINEARSVC=True
TRAIN_RF=True

# Treinar o Stacking Ensemble?
TRAIN_STACKING=True

# Modos de Treinamento a executar no main.py
RUN_BINARY=True
RUN_MULTICLASS=True

# Fração de amostragem do dataset (0.1 = 10% - Útil para desenvolvimento rápido no dataset massivo)
SAMPLE_FRAC=0.1
```

## ▶️ Executando a Pipeline

Após configurar as opções no seu arquivo `.env`, execute o orquestrador:

```bash
python main.py
```

O script realizará automaticamente:
1. Leitura e descarte inteligente de dados super-dimensionados e de variância nula.
2. Treinamento de dados balanceados utilizando **SMOTE**.
3. Construção e Treinamento do **Stacking Ensemble**.
4. Testes e persistência (pasta `models/`).
5. Geração e avaliação gráfica (pasta `results/`).

## 📊 Gráficos de Resultados

Para cada modo ativado (Binário ou Multiclasse), as seguintes imagens `.png` são automaticamente construídas com `matplotlib` e `seaborn` na pasta `./results/`:
- `metrics_comparison_*.png`: Gráfico de barras comparando a Eficiência (*Acurácia, F1-Score, Precisão, Recall*) dos algoritmos base contra o Stacking.
- `roc_curves_*.png`: Comparativo de Taxas Verdadeiros vs Falsos Positivos via *Curva ROC*.
- `confusion_matrices_*.png`: Matrizes de Confusão em grade demonstrando os índices absolutos (normalizados de 0 a 1).
- `latency_vs_f1_*.png`: Relatório do Trade-off computacional, mostrando velocidade de predição *versus* Qualidade Analítica (*F1-Score*).
