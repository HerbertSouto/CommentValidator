import os
import pandas as pd
from preprocess import limpar_texto  # Certifique-se de que essa função esteja disponível

# Nome do arquivo do dataset atualizado
nome_dataset = "dataset_atualizado.csv"

# Verifica se o arquivo existe; se não existir, cria um DataFrame vazio com as colunas necessárias
if os.path.exists(nome_dataset):
    df_atual = pd.read_csv(nome_dataset)
else:
    df_atual = pd.DataFrame(columns=["Comment", "Sentimento", "texto_limpo", "y"])

# Agora, carrega o CSV com os novos comentários (novos_comentarios.csv)
# Esse arquivo deve ter pelo menos as colunas "Comment" e "Sentimento"
df_novos = pd.read_csv("nestle.csv")

# Aplica o pré-processamento aos novos comentários
df_novos["texto_limpo"] = df_novos["Comment"].apply(limpar_texto)
df_novos["y"] = df_novos["Sentimento"].map({"Positivo": 1, "Negativo": -1, "Neutro": 0})

# Concatena o dataset atual com os novos comentários
df_total = pd.concat([df_atual, df_novos], ignore_index=True)

# Salva o dataset atualizado para uso futuro
df_total.to_csv(nome_dataset, index=False)
print("Dataset atualizado salvo em '{}'.".format(nome_dataset))
