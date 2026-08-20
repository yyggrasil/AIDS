import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib
import pandas as pd
from sklearn.preprocessing import label_binarize

def evaluate_models(models_dict, X_test, y_test, results_dir='./results', suffix='binary'):
    """
    Avalia um dicionário de modelos (ex: {'LinearSVC': m1, 'RF': m2, 'Stacking': m3})
    e gera os gráficos comparativos requeridos na arquitetura.
    suffix: para diferenciar os gráficos binários dos multiclasse
    """
    os.makedirs(results_dir, exist_ok=True)
    metrics_data = {}
    latencies = {}
    predictions = {}
    probabilities = {} # Para ROC
    
    classes = np.unique(y_test)
    is_multiclass = len(classes) > 2

    # Inferencia e Coleta de Métricas
    for name, model in models_dict.items():
        print(f"Avaliando modelo: {name}...")
        
        # Medindo Latência
        start_time = time.time()
        preds = model.predict(X_test)
        end_time = time.time()
        
        # Latência em ms por 1000 amostras
        latency_ms_per_1k = ((end_time - start_time) * 1000) / (len(X_test) / 1000.0)
        latencies[name] = latency_ms_per_1k
        predictions[name] = preds
        
        # Probabilidades para ROC (LinearSVC usa decision_function)
        if hasattr(model, "predict_proba"):
            probabilities[name] = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            probabilities[name] = model.decision_function(X_test)
        else:
            probabilities[name] = None
        
        # Calculando métricas (macro para multiclass, binario default para binary)
        avg_method = 'macro' if is_multiclass else 'binary'
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average=avg_method, zero_division=0)
        rec = recall_score(y_test, preds, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, preds, average=avg_method, zero_division=0)
        
        metrics_data[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
        
        print(f"Classification Report para {name}:")
        print(classification_report(y_test, preds, zero_division=0))

    # 1. Gráfico de Barras Agrupadas
    plot_metrics_bar(metrics_data, os.path.join(results_dir, f'metrics_comparison_{suffix}.png'))
    
    # 2. Grid de Matrizes de Confusão
    plot_confusion_matrices(y_test, predictions, classes, os.path.join(results_dir, f'confusion_matrices_{suffix}.png'))
    
    # 3. Curva ROC Comparativa
    if not is_multiclass:
        plot_roc_curves(y_test, probabilities, os.path.join(results_dir, f'roc_curves_{suffix}.png'))
    else:
        # Para multiclass podemos plotar a ROC Macro-average
        plot_roc_curves_multiclass(y_test, probabilities, classes, os.path.join(results_dir, f'roc_curves_macro_{suffix}.png'))
        
    # 4. Latência vs F1-Score
    plot_latency_vs_f1(metrics_data, latencies, os.path.join(results_dir, f'latency_vs_f1_{suffix}.png'))
    
    # 5. Pesos do SVM (LinearSVC)
    if 'LinearSVC' in models_dict:
        scaler_path = os.path.join('models', f'scaler_{suffix}.joblib')
        preprocessor = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        plot_svm_weights(models_dict['LinearSVC'], preprocessor, os.path.join(results_dir, f'svm_feature_weights_{suffix}.png'), is_multiclass=is_multiclass)

    print(f"Todos os gráficos salvos em {results_dir}")


def plot_metrics_bar(metrics_data, save_path):
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = list(metrics_data.keys())
    n_models = len(models)
    
    x = np.arange(len(metrics_list))
    # Para garantir espaço entre os grupos (ex: Accuracy e Precision),
    # limitamos a largura total de todas as barras do grupo a 0.8, deixando 0.2 de margem.
    width = 0.8 / n_models if n_models > 0 else 0.2
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        measurement = [metrics_data[model][m] for m in metrics_list]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=model)
        # Rotacionar os números em 90 graus e reduzir um pouco a fonte para caber em barras finas
        ax.bar_label(rects, padding=3, fmt='%.3f', rotation=45, fontsize=9)
        multiplier += 1

    ax.set_ylabel('Scores')
    ax.set_title('Comparativo de Métricas entre Modelos')
    
    # Centralizar o label exatamente no meio do bloco de barras
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics_list)
    
    # Mover a legenda para fora para não sobrepor as barras muito altas
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Aumentar o limite superior para não cortar os rótulos rotacionados
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(y_true, predictions, classes, save_path):
    models = list(predictions.keys())
    n_models = len(models)
    
    # Se os rótulos forem 0 e 1 (classificação binária), substitui pelos nomes das categorias
    if len(classes) == 2 and set(classes) == {0, 1}:
        display_classes = ['Benigno', 'Maligno']
    else:
        display_classes = classes
    
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    # Se houver apenas 1 modelo, axes não é um array, então colocamos numa lista
    if n_models == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        cm = confusion_matrix(y_true, predictions[model], normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[i], 
                    xticklabels=display_classes, yticklabels=display_classes)
        axes[i].set_title(f'Matriz de Confusão: {model}')
        axes[i].set_xlabel('Predito')
        axes[i].set_ylabel('Verdadeiro')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, probabilities, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors
    
    for idx, (model, proba) in enumerate(probabilities.items()):
        if proba is None:
            continue
            
        # Decision function is 1D for binary
        if len(proba.shape) == 1:
            y_score = proba
        # Predict proba gives 2D for binary, we need the probability of class 1
        elif proba.shape[1] == 2:
            y_score = proba[:, 1]
        else:
            y_score = proba
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        style = linestyles[idx % len(linestyles)]
        color = colors[idx % len(colors)]
        
        # Plot principal
        ax.plot(fpr, tpr, color=color, linestyle=style, lw=2.5, alpha=0.9, 
                label=f'{model} (AUC = {roc_auc:.4f})')
        
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('Curva ROC Comparativa')
    ax.legend(loc="lower right")
    
    # Criar um "zoom" no canto superior esquerdo (onde os modelos geralmente se sobrepõem)
    axins = ax.inset_axes([0.3, 0.4, 0.4, 0.4]) # [x0, y0, width, height]
    for idx, (model, proba) in enumerate(probabilities.items()):
        if proba is None: continue
        if len(proba.shape) == 1: y_score = proba
        elif proba.shape[1] == 2: y_score = proba[:, 1]
        else: y_score = proba
        fpr, tpr, _ = roc_curve(y_true, y_score)
        style = linestyles[idx % len(linestyles)]
        color = colors[idx % len(colors)]
        axins.plot(fpr, tpr, color=color, linestyle=style, lw=2.5, alpha=0.9)
    
    # Limites do zoom
    x1, x2, y1, y2 = -0.01, 0.1, 0.9, 1.01
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves_multiclass(y_true, probabilities, classes, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors
    
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = y_bin.shape[1]
    
    # Pre-calcula os dados do plot
    macro_data = {}
    for idx, (model, proba) in enumerate(probabilities.items()):
        if proba is None or len(proba.shape) == 1:
            continue
            
        fpr = dict()
        tpr = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
            
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        
        roc_auc_macro = auc(all_fpr, mean_tpr)
        macro_data[model] = (all_fpr, mean_tpr, roc_auc_macro, linestyles[idx % len(linestyles)], colors[idx % len(colors)])
    
    for model, (fpr_macro, tpr_macro, roc_auc_macro, style, color) in macro_data.items():
        ax.plot(fpr_macro, tpr_macro, color=color, linestyle=style, lw=2.5, alpha=0.9, 
                label=f'{model} (Macro AUC = {roc_auc_macro:.4f})')

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)')
    ax.set_title('Curva ROC Comparativa (Macro-Average Multiclasse)')
    ax.legend(loc="lower right")
    
    # Criar um "zoom" no canto superior esquerdo
    axins = ax.inset_axes([0.3, 0.4, 0.4, 0.4]) 
    for model, (fpr_macro, tpr_macro, roc_auc_macro, style, color) in macro_data.items():
        axins.plot(fpr_macro, tpr_macro, color=color, linestyle=style, lw=2.5, alpha=0.9)
    
    x1, x2, y1, y2 = -0.01, 0.1, 0.9, 1.01
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_latency_vs_f1(metrics_data, latencies, save_path):
    plt.figure(figsize=(8, 6))
    
    models = list(metrics_data.keys())
    colors = ['blue', 'green', 'red']
    
    for i, model in enumerate(models):
        f1 = metrics_data[model]['F1-Score']
        latency = latencies[model]
        
        plt.scatter(latency, f1, color=colors[i%len(colors)], s=150, label=model)
        plt.annotate(model, (latency, f1), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.xlabel('Latência de Inferência (ms / 1000 amostras)')
    plt.ylabel('F1-Score')
    plt.title('Trade-off: Latência vs Eficácia (F1-Score)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_svm_weights(model, preprocessor, save_path, is_multiclass=False, top_n=25):
    """
    Plota a importância/pesos (coeficientes) do LinearSVC.
    """
    if preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]
    else:
        feature_names = [f"feature_{i}" for i in range(model.coef_.shape[1])]
        
    clean_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    
    if not is_multiclass:
        coefs = model.coef_[0]
        df = pd.DataFrame({'feature': clean_names, 'coefficient': coefs})
        df['abs_coef'] = df['coefficient'].abs()
        df_top = df.sort_values(by='abs_coef', ascending=False).head(top_n).sort_values(by='coefficient', ascending=True)
        
        colors = ['#d9534f' if val > 0 else '#428bca' for val in df_top['coefficient']]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(df_top['feature'], df_top['coefficient'], color=colors, edgecolor='black', linewidth=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.xlabel('Valor do Coeficiente (Peso no SVM)', fontsize=12, fontweight='bold')
        plt.ylabel('Atributo / Feature', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Atributos por Peso no LinearSVC (Binário)\n[ Azul = Tendência Benigna | Vermelho = Tendência Ataque ]', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.grid(True, linestyle=':', alpha=0.6, axis='x')
        
        max_val = max(abs(df_top['coefficient'].min()), abs(df_top['coefficient'].max()))
        offset = max_val * 0.02
        for bar in bars:
            width = bar.get_width()
            ha = 'left' if width >= 0 else 'right'
            x_pos = width + offset if width >= 0 else width - offset
            plt.text(x_pos, bar.get_y() + bar.get_height()/2, f'{width:.5f}', 
                     va='center', ha=ha, fontsize=9, fontweight='bold')
                     
        plt.xlim(df_top['coefficient'].min() - max_val * 0.25, df_top['coefficient'].max() + max_val * 0.25)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        if hasattr(model, 'classes_'):
            classes = model.classes_
        else:
            classes = [f"Classe {i}" for i in range(model.coef_.shape[0])]
            
        coef_df = pd.DataFrame(model.coef_, columns=clean_names, index=classes)
        top_features = coef_df.abs().max(axis=0).sort_values(ascending=False).head(top_n).index
        coef_df_top = coef_df[top_features].T
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(coef_df_top, cmap='vlag', center=0, annot=True, fmt='.4f', 
                    cbar_kws={'label': 'Peso / Coeficiente no SVM'},
                    linewidths=0.5, linecolor='gray')
                    
        plt.title(f'Heatmap dos Pesos do LinearSVC por Classe (Top {top_n} Atributos)', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Classe de Ataque / Benigno', fontsize=12, fontweight='bold')
        plt.ylabel('Atributo / Feature', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

