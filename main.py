import os
import joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, prepare_target
from src.preprocessing import drop_irrelevant_features, get_preprocessor, apply_smote
from src.models import get_base_learners, get_stacking_classifier
from src.evaluation import evaluate_models
from sklearn.pipeline import Pipeline
import numpy as np

def run_pipeline(target_mode='binary', sample_frac=0.1):
    print(f"\n{'='*50}")
    print(f"INICIANDO PIPELINE - MODO: {target_mode.upper()} (Amostra: {sample_frac*100}%)")
    print(f"{'='*50}\n")
    
    # 1. Carregar e Preparar Dados
    df = load_data('data/CICFlowMeter_out.csv', sample_frac=sample_frac)
    df = drop_irrelevant_features(df)
    X, y = prepare_target(df, mode=target_mode)
    
    # Separando numéricos e categóricos baseados nos dtypes
    # Geralmente, Protocol é numérico mas representa categoria. 
    # Aqui vamos tentar inferir ou usar tipos
    categorical_features = []
    if 'Protocol' in X.columns:
        categorical_features.append('Protocol')
        X['Protocol'] = X['Protocol'].astype(str)
    numeric_features = [c for c in X.columns if c not in categorical_features]
    
    # 2. Split Treino/Teste
    print("\nDividindo conjunto de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Pré-processamento e SMOTE
    print("\nAplicando Transformações (Scaler, OneHot)...")
    preprocessor = get_preprocessor(numeric_features, categorical_features, scale_type='standard')
    
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    print("\nBalanceando os dados de treino com SMOTE...")
    X_train_res, y_train_res = apply_smote(X_train_prep, y_train, random_state=42)
    
    # 4. Treinamento dos Modelos Isolados para Comparação
    print("\nTreinando Modelos Isolados...")
    estimators = dict(get_base_learners(random_state=42))
    models_dict = {}
    
    for name, model in estimators.items():
        print(f"Treinando {name}...")
        model.fit(X_train_res, y_train_res)
        models_dict[name] = model
        
    # 5. Treinamento do Stacking Ensemble
    print("\nTreinando Stacking Ensemble (isso pode demorar devido ao CV)...")
    stacking_clf = get_stacking_classifier(random_state=42)
    stacking_clf.fit(X_train_res, y_train_res)
    models_dict['Stacking'] = stacking_clf
    
    # 6. Salvar Modelos (Persistência)
    print("\nSalvando artefatos...")
    os.makedirs('models', exist_ok=True)
    
    # O pipeline completo com Stacking
    stacking_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stacking_clf)
    ])
    joblib.dump(stacking_pipeline, f'models/stacking_pipeline_{target_mode}.joblib')
    joblib.dump(preprocessor, f'models/scaler_{target_mode}.joblib')
    joblib.dump(stacking_clf.final_estimator_, f'models/meta_learner_{target_mode}.joblib')
    print("Modelos salvos em ./models/")
    
    # 7. Avaliação e Gráficos
    print("\nAvaliando modelos no conjunto de Teste e gerando gráficos...")
    evaluate_models(models_dict, X_test_prep, y_test, results_dir='./results', suffix=target_mode)
    
    print("\nPipeline Finalizado com Sucesso!")

if __name__ == "__main__":
    # 10% sample for initial verification as agreed
    # Treina o pipeline binário
    run_pipeline(target_mode='binary', sample_frac=0.1)
    
    # Treina o pipeline multiclasse
    # Algumas classes raras podem quebrar o StratifiedKFold e SMOTE,
    # então o pipeline possui fallbacks (try/except) ou lida com isso adaptando.
    #run_pipeline(target_mode='multiclass', sample_frac=0.1)
