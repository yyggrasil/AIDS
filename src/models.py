from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
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
    diversidade e acurácia.
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
            n_estimators=70,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=1,  # n_jobs=1 para evitar sobrecarga de threads quando executado no Stacking paralelo
            class_weight='balanced',
            random_state=random_state
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=70,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=1,  # n_jobs=1 para evitar concorrência com StackingClassifier n_jobs=-1
            class_weight='balanced',
            random_state=random_state
        )),
        ('hgb', HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=15,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=random_state
        )),
        ('dt', DecisionTreeClassifier(
            criterion='entropy',
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            ccp_alpha=0.0005,
            class_weight='balanced',
            random_state=random_state
        )),
        ('linearsvcCalibrated', CalibratedClassifierCV(LinearSVC(
            C=0.5,
            loss="squared_hinge",
            dual=False,
            tol=1e-4,
            max_iter=10000,
            class_weight='balanced',
            random_state=random_state
            ), cv=3
        ))
    ]
    return estimators

def get_meta_learner(random_state=42):
    """
    Retorna o meta-classificador (Nível 1).
    Utiliza LogisticRegressionCV com validação cruzada para ajuste dinâmico da regularização (C)
    e ponderação equilibrada de classes.
    """
    return LogisticRegressionCV(
        Cs=np.logspace(-2, 2, 5),
        cv=3,
        scoring='accuracy',
        solver="lbfgs",
        max_iter=2000,
        l1_ratios=(0.0,),
        class_weight='balanced',
        random_state=random_state
    )

def get_stacking_classifier(random_state=42, passthrough=True):
    """
    Constrói o Stacking Ensemble otimizado com base na documentação do scikit-learn:
    - Diversidade de estimadores de base (LinearSVC Calibrado, ExtraTrees, HistGradientBoosting e DecisionTree).
    - passthrough=True: Concatena os atributos originais X às probabilidades dos modelos base para que o
      meta-learner avalie o contexto dos fluxos.
    - Meta-classificador LogisticRegressionCV para regularização automática out-of-fold.
    - StratifiedKFold de 3 splits para manter proporção de classes sem data leakage.
    """
    base_estimators = get_base_learners(random_state=random_state)
    estimators_dict = dict(base_estimators)
    
    stacking_estimators = [
        ('linearsvc', estimators_dict['linearsvc']),
        ('et', estimators_dict['et']),
        ('hgb', estimators_dict['hgb']),
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





