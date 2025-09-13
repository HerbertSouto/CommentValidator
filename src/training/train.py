import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from collections import Counter

# Oversampling
from imblearn.over_sampling import RandomOverSampler

DATASET_PATH = "data/processed/dataset_atualizado.csv"
MODEL_PATH = "models/modelo_hibrido.pkl"
METRICS_PATH = "models/metrics.json"

# Configuração
USE_OVERSAMPLING = True   # 👈 mantém ligado

# 1. Carregar dataset
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset não encontrado em {DATASET_PATH}")

df_total = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8-sig")

# Garante colunas obrigatórias
required_cols = {"texto_limpo", "y"}
if not required_cols.issubset(df_total.columns):
    raise ValueError(f"❌ Dataset deve conter as colunas: {required_cols}")

# Remove linhas inválidas
df_total = df_total.dropna(subset=["texto_limpo", "y"])
df_total["y"] = df_total["y"].astype(int)

print(f"✅ Dataset carregado: {len(df_total)} registros válidos após limpeza")

X = df_total["texto_limpo"]
y = df_total["y"]

# 2. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📊 Distribuição original (treino):", Counter(y_train))
print("📊 Distribuição original (teste):", Counter(y_test))

# 3. Oversampling simples (RandomOverSampler)
if USE_OVERSAMPLING:
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train.to_frame(), y_train)
    X_train = X_train["texto_limpo"]  # volta para Series
    print("📊 Distribuição após oversampling (treino):", Counter(y_train))
    print(f"🔄 Oversampling aplicado. Novo tamanho de treino: {len(X_train)} registros")

# 4. Pipeline (TF-IDF + Logistic Regression atualizado)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs"  # sem multi_class para evitar warning
    ))
])

# 5. Treinar modelo
pipeline.fit(X_train, y_train)

# 6. Avaliação
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print("\n📊 Relatório de Classificação (Logistic Regression)")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Salvar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"\n💾 Modelo salvo em '{MODEL_PATH}'")

# 8. Salvar métricas em JSON
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)
print(f"📊 Métricas salvas em '{METRICS_PATH}'")
