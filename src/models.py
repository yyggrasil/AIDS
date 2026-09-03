import os
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def get_mini_neural_network(random_state=42):
    """
    Retorna uma Mini Rede Neural (Multi-Layer Perceptron - MLP) otimizada para
    classificação rápida e eficiente de tráfego de rede tabular:
    - Arquitetura: 2 camadas ocultas compactas (64, 32 neurônios).
    - Ativação: ReLU com otimizador Adam e regularização L2 (alpha=0.0001).
    - Early Stopping: Habilitado para evitar overfitting e acelerar convergência.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=random_state
    )

def get_base_learners(random_state=42):
    """
    Retorna os modelos de base (Nível 0) configurados com hiperparâmetros otimizados para alta velocidade,
    diversidade e máxima capacidade de generalização.
    """
    estimators = [
        ('linearsvc', LinearSVC(
            C=1.0,
            loss="squared_hinge",
            dual="auto",
            tol=1e-3,
            max_iter=5000,
            class_weight='balanced',
            random_state=random_state
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=1,  # n_jobs=1 para evitar sobrecarga de threads no Stacking paralelo
            class_weight='balanced',
            random_state=random_state
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=1,  # n_jobs=1 para evitar concorrência com StackingClassifier n_jobs=-1
            class_weight='balanced',
            random_state=random_state
        )),
        ('dt', DecisionTreeClassifier(
            criterion='entropy',
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            ccp_alpha=0.0001,
            class_weight='balanced',
            random_state=random_state
        )),
        ('mlp', get_mini_neural_network(random_state=random_state)),
        ('hgb', HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=12,
            min_samples_leaf=20,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=random_state
        )),
        ('linearsvcCalibrated', CalibratedClassifierCV(LinearSVC(
            C=1.0,
            loss="squared_hinge",
            dual="auto",
            tol=1e-3,
            max_iter=5000,
            class_weight='balanced',
            random_state=random_state
            ), cv=3, n_jobs=1
        ))
    ]
    return estimators

def get_meta_learner(random_state=42, class_weight='balanced', C=1.0):
    """
    Retorna o meta-classificador (Nível 1).
    Utiliza Regressão Logística com pesos balanceados para combinar de forma
    otimizada e regularizada as probabilidades calibradas geradas pelos estimadores de Nível 0.
    """
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state
    )

def get_stacking_classifier(
    random_state=42,
    passthrough=False,
    cv_splits=5,
    profile='edge',
    estimator_names=None,
    n_jobs=None
):
    """
    Constrói o Stacking Ensemble otimizado com modelos de alta diversidade e calibração:
    - Perfis suportados:
      * 'edge' (Padrão otimizado para Raspberry Pi): linearsvcCalibrated, hgb, rf
      * 'balanced': linearsvcCalibrated, rf, et, hgb
      * 'performance': linearsvcCalibrated, rf, et, hgb, mlp
      * 'legacy': linearsvcCalibrated, rf, et, dt
    - Estimadores personalizados podem ser passados via estimator_names (lista de strings).
    - Meta-classificador (Nível 1): Regressão Logística balanceada operando sobre probabilidades [0, 1].
    - StratifiedKFold com cv_splits para geração de meta-features out-of-fold sem data leakage.
    - passthrough: Se True, concatena os atributos de entrada com as meta-features.
    - n_jobs: Paralelismo na validação cruzada do Stacking. Padrão controlado via STACKING_N_JOBS
      ou min(2, cpu_count) para evitar esgotamento de memória (OOM) no Linux.
    """
    base_estimators = get_base_learners(random_state=random_state)
    estimators_dict = dict(base_estimators)

    if estimator_names is not None:
        stacking_estimators = [(name, estimators_dict[name]) for name in estimator_names if name in estimators_dict]
    elif profile == 'edge':
        stacking_estimators = [
            ('linearsvcCalibrated', estimators_dict['linearsvcCalibrated']),
            ('hgb', estimators_dict['hgb']),
            ('rf', estimators_dict['rf'])
        ]
    elif profile == 'balanced':
        stacking_estimators = [
            ('linearsvcCalibrated', estimators_dict['linearsvcCalibrated']),
            ('rf', estimators_dict['rf']),
            ('et', estimators_dict['et']),
            ('hgb', estimators_dict['hgb'])
        ]
    elif profile == 'performance':
        stacking_estimators = [
            ('linearsvcCalibrated', estimators_dict['linearsvcCalibrated']),
            ('rf', estimators_dict['rf']),
            ('et', estimators_dict['et']),
            ('hgb', estimators_dict['hgb']),
            ('mlp', estimators_dict['mlp'])
        ]
    elif profile == 'legacy':
        stacking_estimators = [
            ('linearsvcCalibrated', estimators_dict['linearsvcCalibrated']),
            ('rf', estimators_dict['rf']),
            ('et', estimators_dict['et']),
            ('dt', estimators_dict['dt'])
        ]
    else:
        raise ValueError(f"Perfil de Stacking desconhecido: '{profile}'. Use 'edge', 'balanced', 'performance' ou 'legacy'.")

    meta_learner = get_meta_learner(random_state=random_state)
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if n_jobs is None:
        env_jobs = os.getenv('STACKING_N_JOBS', '').strip()
        if env_jobs:
            n_jobs = int(env_jobs)
        else:
            # Padrão seguro para evitar estouro de memória e OOM Killer em desktops/servidores Linux:
            # 2 jobs em paralelo equilibra velocidade e uso moderado de RAM (evitando que 8 workers saturem a máquina).
            n_jobs = min(2, os.cpu_count() or 1)

    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=meta_learner,
        cv=cv_strategy,
        stack_method='auto',
        passthrough=passthrough,
        n_jobs=n_jobs
    )

    return stacking_clf

def get_stacking_weights_summary(stacking_clf):
    """
    Retorna um dicionário e representação textual dos coeficientes aprendidos
    pelo meta-classificador para cada estimador base, facilitando interpretabilidade.
    """
    if not hasattr(stacking_clf, 'final_estimator_') or not hasattr(stacking_clf.final_estimator_, 'coef_'):
        return {}
    
    coefs = stacking_clf.final_estimator_.coef_
    estimator_names = [name for name, _ in stacking_clf.estimators]
    return {
        'estimators': estimator_names,
        'coef_shape': coefs.shape,
        'coefficients': coefs.tolist(),
        'intercept': stacking_clf.final_estimator_.intercept_.tolist()
    }







