import pandas as pd
import numpy as np

def load_data(file_path: str, sample_frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Carrega os dados de tráfego de rede e aplica amostragem estratificada, se solicitado.
    """
    print(f"Carregando dados de {file_path}...")
    # Lendo o CSV
    df = pd.read_csv(file_path)
    
    # Removendo espaços em branco dos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Removendo valores infinitos e NaN gerados por divisões por zero em métricas de fluxo
    print("Limpando valores nulos e infinitos...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Removendo linhas duplicadas
    df.drop_duplicates(inplace=True)
    
    if sample_frac < 1.0:
        print(f"Amostrando {sample_frac*100}% do dataset de forma estratificada pela coluna 'Label'...")
        # Usa groupby para amostragem estratificada
        # Se alguma classe tiver menos amostras do que o necessário, pode dar erro no frac,
        # mas para o tamanho do dataset e 10%, deve ser tranquilo, a menos que existam classes raríssimas.
        # Para evitar erros com classes raras (< 10 exemplos), filtramos as que tem poucos exemplos
        # ou usamos um fallback de sample não-estratificado.
        try:
            df = df.groupby('Label').sample(frac=sample_frac, random_state=random_state)
        except Exception:
            print("Aviso: Falha na amostragem estratificada. Realizando amostragem aleatória simples...")
            df = df.sample(frac=sample_frac, random_state=random_state)
            
        # Filtrar classes com menos de 5 amostras para viabilizar StratifiedKFold
        if 'Label' in df.columns:
            class_counts = df['Label'].value_counts()
            rare_classes = class_counts[class_counts < 5].index
            if len(rare_classes) > 0:
                print(f"Filtrando {len(rare_classes)} classe(s) com menos de 5 amostras para garantir estabilidade da validação cruzada...")
                df = df[~df['Label'].isin(rare_classes)]
            
    print(f"Shape final dos dados carregados: {df.shape}")
    return df

def prepare_target(df: pd.DataFrame, target_col: str = 'Label', mode: str = 'binary'):
    """
    Prepara a coluna alvo para classificação binária ou multiclasse.
    mode: 'binary' ou 'multiclass'
    """
    y = df[target_col].copy()
    
    if mode == 'binary':
        print("Preparando alvo para classificação Binária (Maligno vs Benigno)...")
        # Benigno -> 0, Maligno (qualquer outro) -> 1
        y = y.apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    elif mode == 'multiclass':
        print("Preparando alvo para classificação Multiclasse...")
        # Pode manter como string para o LabelEncoder ou retornar direto, 
        # mas aqui só deixaremos como está e os algoritmos lidam com os labels ou usamos LabelEncoder dps.
        # Stripping just in case
        y = y.apply(lambda x: str(x).strip())
    else:
        raise ValueError("Mode deve ser 'binary' ou 'multiclass'")
        
    X = df.drop(columns=[target_col])
    return X, y
