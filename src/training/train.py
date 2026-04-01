import os
import sys
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from collections import Counter

# Garante que src/ está no path para imports relativos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentimentizer.pipeline import pipeline_hibrido

DATASET_PATH = "data/processed/dataset_atualizado.csv"
MODEL_PATH = "models/modelo_hibrido.pkl"
METRICS_PATH = "models/metrics.json"

# 1. Carregar dataset
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset não encontrado em {DATASET_PATH}")

df_total = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8-sig")

required_cols = {"texto_limpo", "y"}
if not required_cols.issubset(df_total.columns):
    raise ValueError(f"❌ Dataset deve conter as colunas: {required_cols}")

df_total = df_total.dropna(subset=["texto_limpo", "y"])
df_total = df_total[df_total["texto_limpo"].str.strip() != ""]
df_total["y"] = df_total["y"].astype(int)

print(f"[OK] Dataset carregado: {len(df_total)} registros validos")
print(f"[INFO] Distribuicao das classes: {Counter(df_total['y'])}")

X = df_total["texto_limpo"]
y = df_total["y"]

# 2. Split treino/teste estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[INFO] Distribuicao treino: {Counter(y_train)}")
print(f"[INFO] Distribuicao teste:  {Counter(y_test)}")

# 3. Treinar pipeline híbrido (TF-IDF + Lexicon + Logistic Regression)
# Nota: class_weight='balanced' já está no pipeline; não usar oversampling junto.
print("\n[INFO] Treinando pipeline hibrido (TF-IDF + Lexicon)...")
pipeline_hibrido.fit(X_train, y_train)

# 4. Validação cruzada (5-fold) para estimativa mais confiável
cv_scores = cross_val_score(pipeline_hibrido, X_train, y_train, cv=5, scoring="f1_macro")
print(f"[INFO] Cross-validation F1-macro (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# 5. Avaliação no conjunto de teste
y_pred = pipeline_hibrido.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print("\n[INFO] Relatorio de Classificacao (Teste)")
print(classification_report(y_test, y_pred, zero_division=0,
                             target_names=["Negativo (-1)", "Neutro (0)", "Positivo (1)"]))

# 6. Salvar modelo e métricas
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline_hibrido, MODEL_PATH)
print(f"[OK] Modelo salvo em '{MODEL_PATH}'")

report["cv_f1_macro_mean"] = float(cv_scores.mean())
report["cv_f1_macro_std"] = float(cv_scores.std())
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)
print(f"[OK] Metricas salvas em '{METRICS_PATH}'")
