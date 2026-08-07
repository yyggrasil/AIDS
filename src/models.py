from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

def get_base_learners(random_state=42):
    """
    Retorna os modelos de base (Nível 0) configurados conforme as especificações.
    """
    estimators = [
        ('linearsvc', LinearSVC(
            C=1.0,
            loss="squared_hinge",
            dual=False,
            max_iter=2000,
            random_state=random_state
        )),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight='balanced',
            random_state=random_state
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight='balanced',
            random_state=random_state
        )),
        ('dt', DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        )),
        ('linearsvcCalibrated', CalibratedClassifierCV(LinearSVC(
            C=1.0,
            loss="squared_hinge",
            dual=False,
            max_iter=2000,
            random_state=random_state
            ), cv=3
        ))
    ]
    return estimators

def get_meta_learner(random_state=42):
    """
    Retorna o meta-classificador (Nível 1).
    Utiliza Regressão Logística ponderada para combinar adequadamente as probabilidades do Nível 0.
    """
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight='balanced',
        random_state=random_state
    )

def get_stacking_classifier(random_state=42):
    """
    Constrói o Stacking Ensemble otimizado com modelos de base diversificados (LinearSVC Calibrado, RF e Extra Trees).
    Mantém passthrough=False para focar exclusivamente nas decisões de Nível 0.
    """
    base_estimators = get_base_learners(random_state=random_state)
    estimators_dict = dict(base_estimators)
    
    linearsvc_key = 'linearsvcCalibrated' if 'linearsvcCalibrated' in estimators_dict else 'linearsvc'
    stacking_estimators = [
        (linearsvc_key, estimators_dict[linearsvc_key]),
        ('rf', estimators_dict['rf'])
    ]
    if 'et' in estimators_dict:
        stacking_estimators.append(('et', estimators_dict['et']))
    
    meta_learner = get_meta_learner(random_state=random_state)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=meta_learner,
        cv=cv_strategy,
        passthrough=False, # Foca nas meta-features (probabilidades/decisões dos modelos base)
        n_jobs=-1 # Paraleliza o cross-validation
    )
    
    return stacking_clf



