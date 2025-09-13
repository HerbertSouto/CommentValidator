import pandas as pd
from sentimentizer.preprocess import limpar_texto

NEG_FILE = "data/raw/comentarios_negativos 2.txt"
DATASET_PATH = "data/processed/dataset_atualizado.csv"

# Carrega os novos comentários negativos
with open(NEG_FILE, "r", encoding="utf-8") as f:
    comentarios = [linha.strip() for linha in f if linha.strip()]

df_neg = pd.DataFrame({
    "Comment": comentarios,
    "Classe": "Negativo",
})
df_neg["texto_limpo"] = df_neg["Comment"].apply(limpar_texto)
df_neg["y"] = -1

print(f"✅ {len(df_neg)} comentários negativos carregados.")

# Carrega dataset existente
try:
    df_total = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8-sig")
except FileNotFoundError:
    df_total = pd.DataFrame(columns=["Comment", "Classe", "texto_limpo", "y"])

# Concatena e remove duplicados
df_total = pd.concat([df_total, df_neg], ignore_index=True)
df_total = df_total.drop_duplicates(subset=["Comment"], keep="last")

# Salva atualizado
df_total.to_csv(DATASET_PATH, sep=";", index=False, encoding="utf-8")
print(f"💾 Dataset atualizado salvo em {DATASET_PATH} (tamanho: {len(df_total)} registros).")
