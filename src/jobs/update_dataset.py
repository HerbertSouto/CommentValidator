import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentimentizer.preprocess import limpar_texto

# Caminhos
DATASET_PATH = "data/processed/dataset_atualizado.csv"
NEW_DATA_PATH = "data/raw/pendentes_validacao.csv"

# Se não houver pendentes, para
if not os.path.exists(NEW_DATA_PATH):
    print("⚠️ Nenhum arquivo de pendentes encontrado.")
    exit()

# Força separador vírgula e remove BOM se existir
df_novos = pd.read_csv(NEW_DATA_PATH, sep=";", encoding="utf-8-sig")

# Garante colunas obrigatórias
required_cols = {"Comment", "Classe"}
if not required_cols.issubset(df_novos.columns):
    raise ValueError("Arquivo de pendentes deve ter colunas 'Comment' e 'Classe'.")

# Se não houver coluna Validado, assume "Não"
if "Validado" not in df_novos.columns:
    df_novos["Validado"] = "Não"

# Filtra apenas registros validados
df_novos = df_novos[df_novos["Validado"].astype(str).str.lower() == "sim"]

if df_novos.empty:
    print("⚠️ Nenhum comentário validado encontrado. Nada a atualizar.")
    exit()

# Normaliza classes para evitar NaN
classe_map = {"positivo": 1, "negativo": -1, "neutro": 0}
df_novos["Classe"] = df_novos["Classe"].astype(str).str.strip().str.lower().str.capitalize()
df_novos["texto_limpo"] = df_novos["Comment"].fillna("").apply(limpar_texto)
df_novos["y"] = (
    df_novos["Classe"].str.strip().str.lower().map(classe_map)
)

# Remove linhas inválidas
df_novos = df_novos.dropna(subset=["y"])
df_novos["y"] = df_novos["y"].astype(int)

# Carrega dataset atual
if os.path.exists(DATASET_PATH):
    df_atual = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8-sig")
else:
    df_atual = pd.DataFrame(columns=["Comment", "Classe", "texto_limpo", "y"])

# Concatena + remove duplicados
df_total = pd.concat([df_atual, df_novos], ignore_index=True)
df_total = df_total.drop_duplicates(subset=["Comment"], keep="last")

# Salva atualizado
df_total.to_csv(DATASET_PATH,  sep=";", index=False, encoding="utf-8")
print(f"✅ Dataset atualizado salvo em '{DATASET_PATH}' (tamanho: {len(df_total)} registros).")
