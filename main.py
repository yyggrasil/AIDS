import os
import gc
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, prepare_target
from src.preprocessing import drop_irrelevant_features, get_preprocessor
from src.models import (
    get_base_learners,
    get_stacking_classifier,
    get_stacking_weights_summary
)
from src.evaluation import evaluate_models
from sklearn.pipeline import Pipeline
import numpy as np

# Carrega variáveis do .env
load_dotenv()

def run_pipeline(target_mode='binary', sample_frac=0.1):
    print(f"\n{'='*50}")
    print(f"INICIANDO PIPELINE - MODO: {target_mode.upper()} (Amostra: {sample_frac*100}%)")
    print(f"{'='*50}\n")
    
    # Flags do .env
    train_linearsvc = os.getenv('TRAIN_LINEARSVC', 'True').strip().lower() == 'true'
    train_dt = os.getenv('TRAIN_DT', 'True').strip().lower() == 'true'
    train_rf = os.getenv('TRAIN_RF', 'True').strip().lower() == 'true'
    train_et = os.getenv('TRAIN_ET', 'False').strip().lower() == 'true'
    train_hgb = os.getenv('TRAIN_HGB', 'False').strip().lower() == 'true'
    train_mlp = os.getenv('TRAIN_MLP', 'True').strip().lower() == 'true'
    train_stacking = os.getenv('TRAIN_STACKING', 'True').strip().lower() == 'true'
    stacking_profile = os.getenv('STACKING_PROFILE', 'edge').strip().lower()
    stacking_cv_splits = int(os.getenv('STACKING_CV_SPLITS', '5'))
    stacking_passthrough = os.getenv('STACKING_PASSTHROUGH', 'False').strip().lower() == 'true'
    stacking_n_jobs_str = os.getenv('STACKING_N_JOBS', '').strip()
    stacking_n_jobs = int(stacking_n_jobs_str) if stacking_n_jobs_str else min(2, os.cpu_count() or 1)
        
    # 1. Carregar e Preparar Dados
    df = load_data('data/CICFlowMeter_out.csv', sample_frac=sample_frac)
    df = drop_irrelevant_features(df)
    X, y = prepare_target(df, mode=target_mode)
    del df
    gc.collect()
    
    categorical_features = []
    if 'Protocol' in X.columns:
        categorical_features.append('Protocol')
        X['Protocol'] = X['Protocol'].astype(str)
    numeric_features = [c for c in X.columns if c not in categorical_features]
    X[numeric_features] = X[numeric_features].astype(np.float32)
    
    # 2. Split Treino/Teste
    print("\nDividindo conjunto de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    del X
    gc.collect()
    
    # Determina se há algum treinamento a ser feito
    any_training = train_linearsvc or train_dt or train_rf or train_et or train_hgb or train_mlp or train_stacking
    
    # 3. Pré-processamento
    if any_training:
        print("\nAplicando Transformações (Scaler, OneHot, VarianceThreshold)...")
        preprocessor = get_preprocessor(numeric_features, categorical_features, scale_type='log_standard')
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
        
        # Salva o scaler recém-treinado
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, f'models/scaler_{target_mode}.joblib')
        print(f"Pré-processador salvo em models/scaler_{target_mode}.joblib")
        
        del X_train, X_test
        gc.collect()
    else:
        print("\nCarregando Pré-processador existente (nenhum treinamento habilitado)...")
        preprocessor_path = f'models/scaler_{target_mode}.joblib'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            X_test_prep = preprocessor.transform(X_test)
            del X_train, X_test
            gc.collect()
        else:
            print(f"Erro: {preprocessor_path} não encontrado. Treine pelo menos um modelo primeiro.")
            return

    # 4. Treinamento ou Carregamento dos Modelos Isolados
    print(f"\nObtendo Modelos Isolados (utilizando class_weight='balanced')...")
    estimators = dict(get_base_learners(random_state=42))
    models_dict = {}
    os.makedirs('models', exist_ok=True)
    
    if train_linearsvc and 'linearsvc' in estimators:
        print("Treinando LinearSVC...")
        m = estimators['linearsvc']
        m.fit(X_train_prep, y_train)
        models_dict['LinearSVC'] = m
        path = f'models/LinearSVC_{target_mode}.joblib'
        joblib.dump(m, path)
        print(f"LinearSVC treinado e salvo com sucesso em {path}")
    else:
        path = f'models/LinearSVC_{target_mode}.joblib'
        if os.path.exists(path):
            print("Carregando LinearSVC existente...")
            models_dict['LinearSVC'] = joblib.load(path)
            
    if train_dt and 'dt' in estimators:
        print("Treinando Decision Tree...")
        m = estimators['dt']
        m.fit(X_train_prep, y_train)
        models_dict['DT'] = m
        path = f'models/DT_{target_mode}.joblib'
        joblib.dump(m, path)
        print(f"Decision Tree treinada e salva com sucesso em {path}")
    else:
        path = f'models/DT_{target_mode}.joblib'
        if os.path.exists(path):
            print("Carregando Decision Tree existente...")
            models_dict['DT'] = joblib.load(path)
            
    if train_rf and 'rf' in estimators:
        print("Treinando RandomForest...")
        m = estimators['rf']
        m.fit(X_train_prep, y_train)
        models_dict['RF'] = m
        path = f'models/RF_{target_mode}.joblib'
        joblib.dump(m, path)
        print(f"RandomForest treinado e salvo com sucesso em {path}")
    else:
        path = f'models/RF_{target_mode}.joblib'
        if os.path.exists(path):
            print("Carregando RandomForest existente...")
            models_dict['RF'] = joblib.load(path)

    if train_et and 'et' in estimators:
        print("Treinando ExtraTrees...")
        m = estimators['et']
        m.fit(X_train_prep, y_train)
        models_dict['ET'] = m
        path = f'models/ET_{target_mode}.joblib'
        joblib.dump(m, path)
        print(f"ExtraTrees treinado e salvo com sucesso em {path}")
    else:
        path = f'models/ET_{target_mode}.joblib'
        if os.path.exists(path):
            print("Carregando ExtraTrees existente...")
            models_dict['ET'] = joblib.load(path)

    if train_hgb and 'hgb' in estimators:
        print("Treinando HistGradientBoosting...")
        m = estimators['hgb']
        m.fit(X_train_prep, y_train)
        models_dict['HGB'] = m
        path = f'models/HGB_{target_mode}.joblib'
        joblib.dump(m, path)
        print(f"HistGradientBoosting treinado e salvo com sucesso em {path}")
    else:
        path = f'models/HGB_{target_mode}.joblib'
        if os.path.exists(path):
            print("Carregando HistGradientBoosting existente...")
            models_dict['HGB'] = joblib.load(path)

    if train_mlp and 'mlp' in estimators:
        print("Treinando Mini Rede Neural (MLP)...")
        m = estimators['mlp']
        m.fit(X_train_prep, y_train)
        models_dict['MLP'] = m
        path = f'models/MLP_{target_mode}.joblib'
        joblib.dump(m, path)
        print(f"Mini Rede Neural (MLP) treinada e salva com sucesso em {path}")
    else:
        path = f'models/MLP_{target_mode}.joblib'
        if os.path.exists(path):
            print("Carregando Mini Rede Neural (MLP) existente...")
            models_dict['MLP'] = joblib.load(path)
            
    # 5. Treinamento ou Carregamento do Stacking Ensemble
    if train_stacking:
        print(f"\nTreinando Stacking Ensemble (Perfil: {stacking_profile.upper()}, CV Splits: {stacking_cv_splits}, Passthrough: {stacking_passthrough}, N_Jobs: {stacking_n_jobs})...")
        stacking_clf = get_stacking_classifier(
            random_state=42,
            profile=stacking_profile,
            cv_splits=stacking_cv_splits,
            passthrough=stacking_passthrough,
            n_jobs=stacking_n_jobs
        )
        stacking_clf.fit(X_train_prep, y_train)
        models_dict['Stacking'] = stacking_clf
        
        path = f'models/Stacking_{target_mode}.joblib'
        joblib.dump(stacking_clf, path)
        
        stacking_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', stacking_clf)
        ])
        joblib.dump(stacking_pipeline, f'models/stacking_pipeline_{target_mode}.joblib')
        joblib.dump(stacking_clf.final_estimator_, f'models/meta_learner_{target_mode}.joblib')
        print(f"Stacking Ensemble e artefatos treinados e salvos com sucesso em ./models/")
        
        # Resumo explicativo dos pesos do Meta-Learner
        weights_summary = get_stacking_weights_summary(stacking_clf)
        if weights_summary and 'estimators' in weights_summary:
            print("\n" + "-"*40)
            print("PESOS APRENDIDOS PELO META-LEARNER (Logistic Regression):")
            print(f"Estimadores Base: {weights_summary['estimators']}")
            print(f"Coeficientes: {weights_summary['coefficients']}")
            print(f"Intercepto: {weights_summary['intercept']}")
            print("-" * 40 + "\n")
    else:
        path = f'models/Stacking_{target_mode}.joblib'
        if os.path.exists(path):
            print("\nCarregando Stacking Ensemble existente...")
            models_dict['Stacking'] = joblib.load(path)
    
    # 6. Avaliação e Gráficos
    if models_dict:
        print("\nAvaliando modelos no conjunto de Teste e gerando gráficos...")
        evaluate_models(models_dict, X_test_prep, y_test, results_dir='./results', suffix=target_mode)
    
    print("\nPipeline Finalizado com Sucesso!")

if __name__ == "__main__":
    sample_frac = float(os.getenv('SAMPLE_FRAC', '0.1'))
    run_binary = os.getenv('RUN_BINARY', 'True').strip().lower() == 'true'
    run_multiclass = os.getenv('RUN_MULTICLASS', 'True').strip().lower() == 'true'
    
    if run_binary:
        run_pipeline(target_mode='binary', sample_frac=sample_frac)
        
    if run_multiclass:
        run_pipeline(target_mode='multiclass', sample_frac=sample_frac)
