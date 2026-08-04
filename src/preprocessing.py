import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
# Note: we use imblearn.pipeline.Pipeline in the model script to include SMOTE, 
# or we apply SMOTE manually in the training loop. We'll provide a function to build the preprocessor.

def get_preprocessor(numeric_features, categorical_features, scale_type='standard'):
    """
    Constrói o pré-processador do scikit-learn.
    scale_type: 'standard' ou 'robust'
    """
    
    # Pipeline para atributos numéricos
    if scale_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Precaution for any remaining NaNs
        ('scaler', scaler)
    ])

    # Pipeline para atributos categóricos
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combina tudo no ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def apply_smote(X_train, y_train, random_state=42):
    """
    Aplica o SMOTE aos dados de treino para balanceamento.
    Atenção: Deve ser aplicado APÓS o split de treino/teste e APENAS no conjunto de treino.
    """
    print(f"Aplicando SMOTE... (Shape antes: {X_train.shape}, Distribuição: {pd.Series(y_train).value_counts().to_dict()})")
    
    # SMOTE k_neighbors padrão é 5. Se alguma classe tiver menos que 6 amostras, 
    # o SMOTE falha. Ajustaremos k_neighbors dinamicamente se necessário,
    # ou usaremos um try/except para contornar classes raras no multiclasse.
    min_samples = pd.Series(y_train).value_counts().min()
    
    k_neighbors = 5
    if min_samples <= 5:
        k_neighbors = max(1, min_samples - 1)
        print(f"Aviso: Classes com poucos exemplos detectadas. Ajustando k_neighbors do SMOTE para {k_neighbors}")
        
    if k_neighbors < 1:
        print("Aviso: Há classes com apenas 1 amostra. O SMOTE não pode ser aplicado para essas classes. Retornando os dados originais.")
        return X_train, y_train

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print(f"SMOTE Concluído. (Shape depois: {X_res.shape}, Distribuição: {pd.Series(y_res).value_counts().to_dict()})")
    return X_res, y_res

def drop_irrelevant_features(df):
    """
    Remove colunas identificadoras que podem causar data leakage / overfitting.
    Ex: IPs, portas de origem (geralmente aleatórias), Timestamp, Flow ID.
    """
    columns_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    
    # Remove only the columns that actually exist in the dataframe
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_cols_to_drop:
        print(f"Removendo colunas irrelevantes/identificadoras: {existing_cols_to_drop}")
        df = df.drop(columns=existing_cols_to_drop)
        
    return df
