from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

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
        ('dt', DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state
        ))
    ]
    return estimators

def get_meta_learner(random_state=42):
    """
    Retorna o meta-classificador (Nível 1) configurado conforme especificações.
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
    Constrói o Stacking Ensemble com os modelos de base e meta-learner.
    """
    estimators = get_base_learners(random_state=random_state)
    meta_learner = get_meta_learner(random_state=random_state)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    stacking_clf = StackingClassifier(
        estimators=estimators[:2],
        final_estimator=meta_learner,
        cv=cv_strategy,
        passthrough=False,
        n_jobs=-1 # Paraleliza o cross-validation
    )
    
    return stacking_clf
