import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import gc
import tempfile
import argparse
import joblib
import numpy as np
import pandas as pd

# Define diretório temporário para cache do Matplotlib para evitar avisos de permissão
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    validation_curve,
    learning_curve
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.tree import DecisionTreeClassifier

# Importações dos módulos do projeto
from src.data_loader import load_data, prepare_target
from src.preprocessing import drop_irrelevant_features, get_preprocessor, select_top_features
from src.models import get_base_learners


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnóstico e Testes de Overfitting para o Modelo Decision Tree (DT)"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='binary',
        choices=['binary', 'multiclass'],
        help="Modo de classificação: 'binary' ou 'multiclass' (padrão: binary)"
    )
    parser.add_argument(
        '--sample_frac',
        type=float,
        default=0.1,
        help="Fração do dataset a carregar para os testes (padrão: 0.1 = 10%%)"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/CICFlowMeter_out.csv',
        help="Caminho do CSV de dados (padrão: data/CICFlowMeter_out.csv)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help="Diretório de saída para salvar relatórios e gráficos (padrão: results)"
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help="Número de folds para Validação Cruzada (padrão: 5)"
    )
    parser.add_argument(
        '--select_k',
        type=int,
        default=60,
        help="Número de melhores features a selecionar via SelectKBest (0 = usar todas, padrão: 60)"
    )
    parser.add_argument(
        '--skip_plots',
        action='store_true',
        help="Se informado, pula a geração de gráficos para agilizar a execução em terminal"
    )
    return parser.parse_args()


def get_dt_instance(random_state=42):
    """
    Recupera a configuração padrão da Decision Tree utilizada no projeto.
    """
    estimators = dict(get_base_learners(random_state=random_state))
    if 'dt' in estimators:
        return estimators['dt']
    return DecisionTreeClassifier(
        criterion='entropy',
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=8,
        ccp_alpha=0.00015,
        class_weight='balanced',
        random_state=random_state
    )


def run_overfitting_analysis(args):
    print("=" * 70)
    print(f" DIAGNÓSTICO DE OVERFITTING - DECISION TREE ({args.mode.upper()}) ")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Carregamento dos dados
    if not os.path.exists(args.data_path):
        print(f"[-] Erro: Arquivo de dados '{args.data_path}' não foi encontrado.")
        sys.exit(1)

    print(f"\n[1/5] Carregando {args.sample_frac * 100:.1f}% dos dados de '{args.data_path}'...")
    df = load_data(args.data_path, sample_frac=args.sample_frac)
    df = drop_irrelevant_features(df)
    X, y = prepare_target(df, mode=args.mode)
    del df
    gc.collect()

    # Tipagem e preparação
    categorical_features = []
    if 'Protocol' in X.columns:
        categorical_features.append('Protocol')
        X['Protocol'] = X['Protocol'].astype(str)
    numeric_features = [c for c in X.columns if c not in categorical_features]
    X[numeric_features] = X[numeric_features].astype(np.float32)

    # Divisão Treino e Teste (80% / 20%)
    print("[2/5] Dividindo dados em Treino (80%) e Teste (20%) com estratificação...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    del X
    gc.collect()

    # Pré-processamento
    print("[3/5] Aplicando pré-processamento (Scaler e Encoders)...")
    preprocessor_path = os.path.join('models', f'scaler_{args.mode}.joblib')
    if os.path.exists(preprocessor_path):
        print(f"      Usando pré-processador existente: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        X_train_prep = preprocessor.transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
    else:
        print("      Ajustando novo pré-processador...")
        preprocessor = get_preprocessor(numeric_features, categorical_features, scale_type='log_standard')
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)

    del X_train, X_test
    gc.collect()

    # Seleção de Features (SelectKBest)
    if args.select_k and args.select_k > 0 and args.select_k < X_train_prep.shape[1]:
        print(f"      Selecionando as {args.select_k} features mais relevantes (de {X_train_prep.shape[1]}) via ANOVA F-value...")
        X_train_prep, X_test_prep, _ = select_top_features(X_train_prep, y_train, X_test_prep, k=args.select_k)

    # Obter o modelo Decision Tree
    model_path = os.path.join('models', f'DT_{args.mode}.joblib')
    dt_model = get_dt_instance(random_state=42)

    print("\n[4/5] Treinando modelo Decision Tree nos dados de treino...")
    dt_model.fit(X_train_prep, y_train)

    is_multiclass = (args.mode == 'multiclass')
    avg_method = 'macro' if is_multiclass else 'binary'

    # Previsões
    y_train_pred = dt_model.predict(X_train_prep)
    y_test_pred = dt_model.predict(X_test_prep)

    # Cálculo das métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_prec = precision_score(y_train, y_train_pred, average=avg_method, zero_division=0)
    test_prec = precision_score(y_test, y_test_pred, average=avg_method, zero_division=0)

    train_rec = recall_score(y_train, y_train_pred, average=avg_method, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average=avg_method, zero_division=0)

    train_f1 = f1_score(y_train, y_train_pred, average=avg_method, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average=avg_method, zero_division=0)

    f1_gap = train_f1 - test_f1
    acc_gap = train_acc - test_acc

    # =========================================================================
    # RELATÓRIO 1: GAP TREINO VS. TESTE
    # =========================================================================
    print("\n" + "=" * 70)
    print(" RELATÓRIO 1: COMPARAÇÃO TREINO VS. TESTE (GAP DE GENERALIZAÇÃO)")
    print("=" * 70)
    metrics_table = [
        ("Acurácia", f"{train_acc*100:6.2f}%", f"{test_acc*100:6.2f}%", f"{acc_gap*100:+6.2f}%"),
        ("Precisão", f"{train_prec*100:6.2f}%", f"{test_prec*100:6.2f}%", f"{(train_prec-test_prec)*100:+6.2f}%"),
        ("Recall",   f"{train_rec*100:6.2f}%",  f"{test_rec*100:6.2f}%",  f"{(train_rec-test_rec)*100:+6.2f}%"),
        ("F1-Score", f"{train_f1*100:6.2f}%",  f"{test_f1*100:6.2f}%",  f"{f1_gap*100:+6.2f}%")
    ]
    print(f"{'Métrica':<15} | {'Treino':<10} | {'Teste':<10} | {'Gap (Treino - Teste)':<20}")
    print("-" * 65)
    for name, tr, te, diff in metrics_table:
        print(f"{name:<15} | {tr:<10} | {te:<10} | {diff:<20}")
    print("-" * 65)

    # Diagnóstico automático do Gap
    if f1_gap > 0.10:
        verdict = "[ALERTA CRÍTICO] Sinais claros de OVERFITTING SEVERO (Gap F1 > 10%)."
        recommendation = "-> Reduza max_depth, aumente min_samples_leaf ou aumente ccp_alpha."
    elif f1_gap > 0.05:
        verdict = "[ATENÇÃO] Sinais de OVERFITTING MODERADO (Gap F1 entre 5% e 10%)."
        recommendation = "-> Ajustar regularização (min_samples_split, min_samples_leaf ou ccp_alpha)."
    else:
        verdict = "[EXCELENTE] O modelo apresenta BOA GENERALIZAÇÃO (Gap F1 <= 5%)."
        recommendation = "-> A árvore mantém performance consistente sem decorar os dados de treino."

    print(f"Diagnóstico: {verdict}")
    print(f"Recomendação: {recommendation}")

    # =========================================================================
    # RELATÓRIO 2: COMPLEXIDADE DA ÁRVORE
    # =========================================================================
    print("\n" + "=" * 70)
    print(" RELATÓRIO 2: COMPLEXIDADE ESTRUTURAL DA ÁRVORE")
    print("=" * 70)
    depth = dt_model.get_depth()
    leaves = dt_model.get_n_leaves()
    samples_per_leaf = len(y_train) / max(1, leaves)

    print(f"- Profundidade máxima configurada (max_depth): {dt_model.max_depth}")
    print(f"- Profundidade real atingida:                   {depth}")
    print(f"- Número total de folhas (n_leaves):            {leaves}")
    print(f"- Média de amostras de treino por folha:        {samples_per_leaf:.1f}")

    if samples_per_leaf < 5:
        print("[!] Aviso: Muitas folhas com pouquissimas amostras (< 5). "
              "Isso pode indicar particoes excessivamente especificas (memorizacao de ruido).")
    else:
        print("[OK] Numero de amostras por folha esta em patamar seguro.")

    # =========================================================================
    # RELATÓRIO 3: VALIDAÇÃO CRUZADA K-FOLD
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" RELATORIO 3: VALIDACAO CRUZADA ({args.cv_folds}-FOLD STRATIFIED)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    scoring = 'f1_macro' if is_multiclass else 'f1'
    cv_scores = cross_val_score(dt_model, X_train_prep, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    print(f"Pontuacoes dos Folds ({scoring}): {[round(s, 4) for s in cv_scores]}")
    print(f"Media CV: {cv_scores.mean()*100:.2f}% | Desvio Padrao: {cv_scores.std()*100:.2f}%")

    if cv_scores.std() > 0.05:
        print("[!] Alerta de alta variancia entre folds (std > 5%). "
              "O modelo pode ser sensivel a particoes especificas de dados.")
    else:
        print("[OK] Baixa variancia entre folds (estabilidade verificada).")

    # =========================================================================
    # RELATÓRIO 4: CLASSIFICATION REPORT NO CONJUNTO DE TESTE
    # =========================================================================
    print("\n" + "=" * 70)
    print(" RELATÓRIO 4: DETALHAMENTO DE PERFORMANCE NO CONJUNTO DE TESTE")
    print("=" * 70)
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # =========================================================================
    # GRÁFICOS: CURVA DE VALIDAÇÃO E CURVA DE APRENDIZADO
    # =========================================================================
    if not args.skip_plots:
        print("\n[5/5] Gerando painel gráfico de diagnóstico de overfitting...")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Comparação de Métricas (Treino vs Teste)
        metric_names = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        train_vals = [train_acc, train_prec, train_rec, train_f1]
        test_vals = [test_acc, test_prec, test_rec, test_f1]

        x = np.arange(len(metric_names))
        width = 0.35

        axes[0].bar(x - width/2, train_vals, width, label='Treino', color='#1f77b4', alpha=0.9)
        axes[0].bar(x + width/2, test_vals, width, label='Teste', color='#ff7f0e', alpha=0.9)
        axes[0].set_ylabel('Score (0 a 1.0)')
        axes[0].set_title(f'Treino vs. Teste ({args.mode.capitalize()})\nGap F1: {f1_gap*100:+.2f}%')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metric_names)
        axes[0].set_ylim(0, 1.05)
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # 2. Validation Curve (max_depth)
        print("      Calculando Curva de Validação (max_depth de 2 a 22)...")
        depth_range = np.array([2, 4, 6, 8, 10, 12, 15, 18, 22])
        train_scores, val_scores = validation_curve(
            estimator=get_dt_instance(random_state=42),
            X=X_train_prep,
            y=y_train,
            param_name="max_depth",
            param_range=depth_range,
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        axes[1].plot(depth_range, train_mean, 'o-', color='#1f77b4', label='Treino')
        axes[1].fill_between(depth_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1f77b4')
        axes[1].plot(depth_range, val_mean, 's-', color='#2ca02c', label='Validação (CV)')
        axes[1].fill_between(depth_range, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#2ca02c')
        axes[1].axvline(x=dt_model.max_depth or depth, color='red', linestyle='--', label=f'Profundidade Atual ({dt_model.max_depth})')

        axes[1].set_xlabel('Profundidade da Árvore (max_depth)')
        axes[1].set_ylabel(f'Score ({scoring})')
        axes[1].set_title('Curva de Validação (max_depth)\n(Divergência = Overfitting)')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # 3. Learning Curve
        print("      Calculando Curva de Aprendizado (Learning Curve)...")
        train_sizes, lc_train_scores, lc_val_scores = learning_curve(
            estimator=get_dt_instance(random_state=42),
            X=X_train_prep,
            y=y_train,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )

        lc_train_mean = np.mean(lc_train_scores, axis=1)
        lc_train_std = np.std(lc_train_scores, axis=1)
        lc_val_mean = np.mean(lc_val_scores, axis=1)
        lc_val_std = np.std(lc_val_scores, axis=1)

        axes[2].plot(train_sizes, lc_train_mean, 'o-', color='#1f77b4', label='Treino')
        axes[2].fill_between(train_sizes, lc_train_mean - lc_train_std, lc_train_mean + lc_train_std, alpha=0.15, color='#1f77b4')
        axes[2].plot(train_sizes, lc_val_mean, 's-', color='#ff7f0e', label='Validação (CV)')
        axes[2].fill_between(train_sizes, lc_val_mean - lc_val_std, lc_val_mean + lc_val_std, alpha=0.15, color='#ff7f0e')

        axes[2].set_xlabel('Amostras de Treino')
        axes[2].set_ylabel(f'Score ({scoring})')
        axes[2].set_title('Curva de Aprendizado (Learning Curve)\n(Convergência Treino vs. Validação)')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f'dt_overfitting_analysis_{args.mode}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[OK] Painel grafico salvo com sucesso em: {plot_path}")

    print("\n" + "=" * 70)
    print(" ANÁLISE CONCLUÍDA ")
    print("=" * 70)


if __name__ == '__main__':
    args = parse_args()
    run_overfitting_analysis(args)
