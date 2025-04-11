import os
import pandas as pd
import random
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import limpar_texto, LexiconScoreTransformer, ajustar_contexto

# ------------------------------
# Pré-processamento e Configurações
# ------------------------------
nltk.download('stopwords')
portuguese_stopwords = stopwords.words('portuguese')

tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    stop_words=portuguese_stopwords
)

# Defina o lexicon com pesos ajustados
lexicon_palavras = {
    # Positivos
    "show": 1,
    "sensacional": 1.5,
    "incrível": 1.5,
    "excelente": 2,
    "excelentes": 2,
    "bom": 1,
    "boa": 1,
    "ótimo": 1.5,
    "ótima": 1.5,
    "fantástico": 2,
    "fantástica": 2,
    "adoro": 1,
    "amei": 1,
    "maravilhoso": 2,
    "maravilhosa": 2,
    "recomendo": 1,
    "engajado": 1,
    "engajada": 1,
    "influente": 1.5,
    "carismático": 1,
    "carismática": 1,
    "musa": 1,
    "muso": 1,
    # Derivados de emojis positivos:
    "amor": 1.5,
    "brilho": 1,
    "encantado": 1.5,
    "quente": 1,
    "aplauso": 1,
    "divertido": 1,
    "feliz": 1,
    "perfeito": 1.5,
    "deslumbrante": 1.5,
    "ok": 0.5,
    # Negativos
    "ruim": -1,
    "péssimo": -1.5,
    "horrível": -2,
    "decepcionante": -1,
    "chato": -0.5,
    "chata": -0.5,
    "terrível": -2,
    "desastroso": -2,
    "defeituoso": -1.5,
    "defeituosa": -1.5,
    # Derivados de emojis negativos:
    "irritado": -1.5,
    "triste": -1,
    "negativo": -1,
    "nojento": -1.5,
    "assustado": -1,
    # Neutros
    "mediano": 0,
    "mediana": 0,
    "normal": 0,
    "regular": 0,
    "comum": 0,
    "básico": 0,
    "padrão": 0,
    "convencional": 0,
    "simples": 0,
    "neutro": 0,
    "moderado": 0,
    "moderada": 0,
    # Derivados de emojis neutros:
    "explosivo": 0
}

# ------------------------------
# Carregamento do Dataset Atualizado
# ------------------------------
dataset_nome = "dataset_atualizado.csv"
if os.path.exists(dataset_nome):
    df_total = pd.read_csv(dataset_nome, usecols=["Comment", "Sentimento"])
    print(f"Dataset atualizado carregado de '{dataset_nome}'.")
else:
    print(f"O arquivo '{dataset_nome}' não existe. Verifique se o dataset foi criado.")
    exit()

df_total["Comment"] = df_total["Comment"].fillna("")
df_total["texto_limpo"] = df_total["Comment"].apply(limpar_texto)
df_total["y"] = df_total["Sentimento"].map({"Positivo": 1, "Negativo": -1, "Neutro": 0})

print("Dataset Total (atualizado):")
print(df_total.head())

# ------------------------------
# Divisão dos Dados e Treinamento do Modelo
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_total["texto_limpo"], df_total["y"], test_size=0.2, random_state=42, stratify=df_total["y"]
)

pipeline_hibrido = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", tfidf),
        ("lexicon", LexiconScoreTransformer(lexicon=lexicon_palavras, usar_contexto=True, escala=2.0))
    ])),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline_hibrido.fit(X_train, y_train)
preds_hibrido = pipeline_hibrido.predict(X_test)
print("\nRelatório de Classificação - Pipeline Híbrido (TF-IDF + Lexicon):")
from sklearn.metrics import classification_report
print(classification_report(y_test, preds_hibrido, zero_division=0))

pipeline_tfidf = Pipeline([
    ("features", tfidf),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline_tfidf.fit(X_train, y_train)
preds_tfidf = pipeline_tfidf.predict(X_test)
print("\nRelatório de Classificação - Pipeline Apenas TF-IDF:")
print(classification_report(y_test, preds_tfidf, zero_division=0))

# ------------------------------
# Salva o Modelo Treinado
# ------------------------------
joblib.dump(pipeline_hibrido, "modelo_hibrido.pkl")
print("Modelo treinado salvo como 'modelo_hibrido.pkl'")
