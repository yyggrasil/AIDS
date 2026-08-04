import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
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
    
    print(f"Todos os gráficos salvos em {results_dir}")


def plot_metrics_bar(metrics_data, save_path):
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = list(metrics_data.keys())
    
    x = np.arange(len(metrics_list))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        measurement = [metrics_data[model][m] for m in metrics_list]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=model)
        ax.bar_label(rects, padding=3, fmt='%.3f')
        multiplier += 1

    ax.set_ylabel('Scores')
    ax.set_title('Comparativo de Métricas entre Modelos')
    ax.set_xticks(x + width, metrics_list)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrices(y_true, predictions, classes, save_path):
    models = list(predictions.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if len(models) < 3:
        # Fallback if somehow not exactly 3 models
        fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
        if len(models) == 1:
            axes = [axes]
    
    for i, model in enumerate(models):
        cm = confusion_matrix(y_true, predictions[model])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], normalize='true', 
                    xticklabels=classes, yticklabels=classes)
        axes[i].set_title(f'Matriz de Confusão: {model}')
        axes[i].set_xlabel('Predito')
        axes[i].set_ylabel('Verdadeiro')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, probabilities, save_path):
    plt.figure(figsize=(8, 6))
    
    for model, proba in probabilities.items():
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
        
        plt.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC Comparativa')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves_multiclass(y_true, probabilities, classes, save_path):
    plt.figure(figsize=(8, 6))
    
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = y_bin.shape[1]
    
    for model, proba in probabilities.items():
        if proba is None:
            continue
        
        # Macro ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # If it's a decision function of shape (n_samples, n_classes)
        if len(proba.shape) == 1:
             continue # Shouldn't happen for multiclass
             
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
        
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        
        plt.plot(fpr_macro, tpr_macro, lw=2, label=f'{model} (Macro AUC = {roc_auc_macro:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC Comparativa (Macro-Average Multiclasse)')
    plt.legend(loc="lower right")
    
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
