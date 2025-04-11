import pandas as pd
import joblib
from preprocess import limpar_texto  # Importa a função de pré-processamento definida em preprocess.py

# Carrega o modelo treinado (certifique-se de que modelo_hibrido.pkl está no mesmo diretório)
pipeline_hibrido = joblib.load("modelo_hibrido.pkl")

# Carrega o CSV com os comentários reais
# O CSV deve possuir uma coluna chamada "Comment"
df = pd.read_csv("incrivel.csv", usecols=["Comment"])

# Preenche valores NaN na coluna "Comment" para evitar erros
df["Comment"] = df["Comment"].fillna("")

# Aplica o pré-processamento para gerar a coluna "Comment_limpo"
df["Comment_limpo"] = df["Comment"].apply(limpar_texto)

# Faz a predição usando o modelo treinado
df["Predicao"] = pipeline_hibrido.predict(df["Comment_limpo"])

def mapear_classe(pred):
    if pred == 1:
        return "Positivo"
    elif pred == -1:
        return "Negativo"
    elif pred == 0:
        return "Neutro"
    return str(pred)

df["Classe"] = df["Predicao"].apply(mapear_classe)

# Exibe as primeiras linhas com os comentários, texto limpo e a classificação
print(df[["Comment", "Comment_limpo", "Classe"]].head())

# Salva os resultados em um novo CSV
df.to_csv("coments_classificados.csv", index=False)
print("Os comentários foram classificados e salvos em 'coments_classificados.csv'.")
