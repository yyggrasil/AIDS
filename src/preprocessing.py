import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

def get_preprocessor(numeric_features, categorical_features, scale_type='standard'):
    """
    Constrói o pré-processador do scikit-learn otimizado para float32.
    scale_type: 'standard' ou 'robust'
    """
    
    if scale_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('variance', VarianceThreshold(threshold=0.0)),
        ('scaler', scaler)
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', dtype=np.float32))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def apply_smote(X_train, y_train, random_state=42):
    """
    Aplica o SMOTE aos dados de treino para balanceamento.
    """
    print(f"Aplicando SMOTE... (Shape antes: {X_train.shape})")
    
    min_samples = pd.Series(y_train).value_counts().min()
    k_neighbors = 5
    if min_samples <= 5:
        k_neighbors = max(1, min_samples - 1)
        
    if k_neighbors < 1:
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE Concluído. (Shape depois: {X_res.shape})")
    return X_res, y_res

def drop_irrelevant_features(df):
    """
    Remove colunas identificadoras e colunas com variância zero detectadas.
    """
    # Identificadores
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    
    # Constantes conhecidas (zero variância)
    constant_columns = [
        'Bwd Packet Length Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 
        'Fwd URG Flags', 'Bwd URG Flags', 'URG Flag Count', 
        'CWR Flag Count', 'ECE Flag Count', 'Fwd Bytes/Bulk Avg', 
        'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg'
    ]
    
    columns_to_drop.extend(constant_columns)
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_cols_to_drop:
        print(f"Removendo {len(existing_cols_to_drop)} colunas irrelevantes/constantes.")
        df = df.drop(columns=existing_cols_to_drop)
        
    return df
