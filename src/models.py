from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def get_base_learners(random_state=42):
    """
    Retorna os modelos de base (Nível 0) configurados com hiperparâmetros otimizados para alta velocidade,
    diversidade e máxima capacidade de generalização.
    """
    estimators = [
        ('linearsvc', LinearSVC(
            C=0.5,
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
        ('linearsvcCalibrated', CalibratedClassifierCV(LinearSVC(
            C=0.5,
            loss="squared_hinge",
            dual="auto",
            tol=1e-3,
            max_iter=5000,
            class_weight='balanced',
            random_state=random_state
            ), cv=3
        ))
    ]
    return estimators

def get_meta_learner(random_state=42):
    """
    Retorna o meta-classificador (Nível 1).
    Utiliza Regressão Logística padrão para combinar de forma balanceada e ótima
    as probabilidades calibradas geradas pelos estimadores de Nível 0.
    """
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state
    )

def get_stacking_classifier(random_state=42, passthrough=False):
    """
    Constrói o Stacking Ensemble otimizado (sem estimador de gradiente):
    - Estimadores Base (Nível 0): LinearSVC Calibrado, RandomForest, ExtraTrees e DecisionTree.
    - Meta-classificador (Nível 1): Regressão Logística operando sobre probabilidades [0, 1].
    - StratifiedKFold de 3 splits para geração de meta-features out-of-fold sem data leakage.
    - passthrough=False: Foca o meta-aprendizado na combinação ótima das predições especializadas.
    """
    base_estimators = get_base_learners(random_state=random_state)
    estimators_dict = dict(base_estimators)
    
    linearsvc_key = 'linearsvcCalibrated' if 'linearsvcCalibrated' in estimators_dict else 'linearsvc'
    
    stacking_estimators = [
        (linearsvc_key, estimators_dict[linearsvc_key]),
        ('rf', estimators_dict['rf']),
        ('et', estimators_dict['et']),
        ('dt', estimators_dict['dt'])
    ]
    
    meta_learner = get_meta_learner(random_state=random_state)
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=meta_learner,
        cv=cv_strategy,
        stack_method='auto',
        passthrough=passthrough,
        n_jobs=-1
    )
    
    return stacking_clf






