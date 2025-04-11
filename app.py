import streamlit as st
import pandas as pd
import joblib
from preprocess import limpar_texto  # Função definida no seu módulo preprocess.py

def mapear_classe(pred):
    if pred == 1:
        return "Positivo"
    elif pred == -1:
        return "Negativo"
    elif pred == 0:
        return "Neutro"
    return str(pred)

st.title("Classificador de Comentários")
st.write("Envie um arquivo CSV com a coluna 'Comment' para classificar os comentários.")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
if uploaded_file is not None:
    try:
        # Lê somente a coluna "Comment"
        df = pd.read_csv(uploaded_file, usecols=["Comment"])
    except Exception as e:
        st.error("Erro ao ler o CSV. Certifique-se de que ele contém a coluna 'Comment'.")
    else:
        # Preenche valores nulos e cria a coluna limpa para fins de classificação
        df["Comment"] = df["Comment"].fillna("")
        df["Comment_limpo"] = df["Comment"].apply(limpar_texto)

        # Carrega o modelo treinado
        model = joblib.load("modelo_hibrido.pkl")

        # Faz a predição
        df["Predicao"] = model.predict(df["Comment_limpo"])
        df["Classe"] = df["Predicao"].apply(mapear_classe)

        st.subheader("Resultados Classificados")
        # Exibe somente as colunas "Comment" e "Classe"
        st.dataframe(df[["Comment", "Classe"]])

        # Opção de download: CSV com somente "Comment" e "Classe"
        csv_data = df[["Comment", "Classe"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV Classificado",
            data=csv_data,
            file_name="coments_classificados.csv",
            mime="text/csv"
        )
