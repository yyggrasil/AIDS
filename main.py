import os
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from src.data_loader import load_data, prepare_target
from src.preprocessing import drop_irrelevant_features, get_preprocessor
from src.models import get_base_learners, get_stacking_classifier
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
    train_stacking = os.getenv('TRAIN_STACKING', 'True').strip().lower() == 'true'
        
    # 1. Carregar e Preparar Dados
    df = load_data('data/CICFlowMeter_out.csv', sample_frac=sample_frac)
    df = drop_irrelevant_features(df)
    X, y = prepare_target(df, mode=target_mode)
    
    categorical_features = []
    if 'Protocol' in X.columns:
        categorical_features.append('Protocol')
        X['Protocol'] = X['Protocol'].astype(str)
    numeric_features = [c for c in X.columns if c not in categorical_features]
    X[numeric_features] = X[numeric_features].astype(np.float32)
    
    # 2. Split Treino/Teste
    print("\nDividindo conjunto de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Determina se há algum treinamento a ser feito
    any_training = train_linearsvc or train_dt or train_rf or train_stacking
    
    # 3. Pré-processamento
    if any_training:
        print("\nAplicando Transformações (Scaler, OneHot, VarianceThreshold)...")
        preprocessor = get_preprocessor(numeric_features, categorical_features, scale_type='robust')
        X_train_prep = preprocessor.fit_transform(X_train)
        X_test_prep = preprocessor.transform(X_test)
        
        # Salva o scaler recém-treinado
        os.makedirs('models', exist_ok=True)
        joblib.dump(preprocessor, f'models/scaler_{target_mode}.joblib')
        print(f"Pré-processador salvo em models/scaler_{target_mode}.joblib")
    else:
        print("\nCarregando Pré-processador existente (nenhum treinamento habilitado)...")
        preprocessor_path = f'models/scaler_{target_mode}.joblib'
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            X_test_prep = preprocessor.transform(X_test)
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
            
    # 5. Treinamento ou Carregamento do Stacking Ensemble
    if train_stacking:
        print("\nTreinando Stacking Ensemble (isso pode demorar devido ao CV)...")
        stacking_clf = get_stacking_classifier(random_state=42)
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
