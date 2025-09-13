import pandas as pd
import joblib
from sentimentizer.preprocess import limpar_texto  # pré-processamento

# Caminhos dos artefatos
MODEL_PATH = "models/modelo_hibrido.pkl"
INPUT_PATH = "data/raw/incrivel.csv"
OUTPUT_PATH = "data/processed/comentarios_classificados.csv"

# Carrega o modelo treinado
pipeline_hibrido = joblib.load(MODEL_PATH)

# Carrega o CSV com os comentários reais
df = pd.read_csv(INPUT_PATH, usecols=["Comment"])
df["Comment"] = df["Comment"].fillna("")

# Aplica pré-processamento
df["Comment_limpo"] = df["Comment"].apply(limpar_texto)

# Predição
df["Predicao"] = pipeline_hibrido.predict(df["Comment_limpo"])

def mapear_classe(pred: int) -> str:
    if pred == 1:
        return "Positivo"
    if pred == -1:
        return "Negativo"
    if pred == 0:
        return "Neutro"
    return str(pred)

df["Classe"] = df["Predicao"].apply(mapear_classe)

# Mostra preview
print(df[["Comment", "Comment_limpo", "Classe"]].head())

# Salva os resultados
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Comentários classificados salvos em '{OUTPUT_PATH}'.")
