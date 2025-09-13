import streamlit as st
import pandas as pd
import joblib
import os
import sys
import subprocess
from sentimentizer.preprocess import limpar_texto  # Função de pré-processamento

MODEL_PATH = "models/modelo_hibrido.pkl"
PENDENTES_PATH = "data/raw/pendentes_validacao.csv"

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

# Upload e classificação
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, usecols=["Comment"])
    except Exception:
        st.error("Erro ao ler o CSV. Certifique-se de que ele contém a coluna 'Comment'.")
    else:
        # Preenche valores nulos e cria a coluna limpa
        df["Comment"] = df["Comment"].fillna("")
        df["Comment_limpo"] = df["Comment"].apply(limpar_texto)

        # Verificação extra: não rodar modelo se não houver comentários válidos
        if df["Comment_limpo"].str.strip().eq("").all():
            st.warning("⚠️ Nenhum comentário válido encontrado após pré-processamento.")
        else:
            # Carrega o modelo treinado
            model = joblib.load(MODEL_PATH)

            # Faz a predição
            df["Predicao"] = model.predict(df["Comment_limpo"])
            df["Classe"] = df["Predicao"].apply(mapear_classe)

            st.subheader("Comentários Classificados")
            st.dataframe(df[["Comment", "Classe"]])

            # Download CSV
            csv_data = df[["Comment", "Classe"]].to_csv(index=False, sep=";").encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV Classificado",
                data=csv_data,
                file_name="comentarios_classificados.csv",
                mime="text/csv"
            )

            # 🔹 Validar/Editar comentários
            if st.button("✏️ Validar/Editar Comentários"):
                df_validacao = df[["Comment", "Classe"]].copy()
                df_validacao["Validado"] = "Não"  # default
                df_validacao.to_csv(PENDENTES_PATH, sep=";", index=False, encoding="utf-8")
                st.success("Arquivo salvo em data/raw/pendentes_validacao.csv. Corrija abaixo ou edite manualmente antes do re-treino.")

st.markdown("---")

# Mostrar status e editor de validação
if os.path.exists(PENDENTES_PATH):
    df_pend = pd.read_csv(PENDENTES_PATH, sep=";", encoding="utf-8-sig")

    if "Validado" in df_pend.columns:
        total = len(df_pend)
        validados = (df_pend["Validado"].astype(str).str.lower() == "sim").sum()
        st.info(f"📊 Arquivo de validação encontrado: {validados}/{total} comentários validados.")

        # 🔹 Editor de validação dentro do app
        st.subheader("Validação de Comentários")
        edited_df = st.data_editor(
            df_pend,
            width="stretch",   # substitui use_container_width
            num_rows="dynamic"
        )

        if st.button("💾 Salvar validações"):
            edited_df.to_csv(PENDENTES_PATH, sep=";", index=False, encoding="utf-8")
            st.success("Validações salvas com sucesso!")

    else:
        st.warning("📂 Arquivo de validação encontrado, mas sem coluna 'Validado'.")
else:
    st.info("Nenhum arquivo pendente de validação encontrado.")

st.markdown("---")

# 🔄 Re-treino (só funciona se houver registros validados)
if st.button("🔄 Re-treinar Modelo com Novos Dados"):
    if os.path.exists(PENDENTES_PATH):
        df_check = pd.read_csv(PENDENTES_PATH, sep=";", encoding="utf-8-sig")
        if "Validado" in df_check.columns and (df_check["Validado"].astype(str).str.lower() == "sim").any():
            with st.spinner("Atualizando dataset e re-treinando modelo..."):
                try:
                    subprocess.run([sys.executable, "src/jobs/update_dataset.py"], check=True)
                    subprocess.run([sys.executable, "src/training/train.py"], check=True)
                    st.success("✅ Modelo atualizado com sucesso! Recarregue a página para usar o novo modelo.")
                except Exception as e:
                    st.error(f"Erro no re-treino: {e}")
        else:
            st.warning("⚠️ Nenhum comentário validado encontrado. Valide ao menos um comentário antes de re-treinar.")
    else:
        st.warning("⚠️ Nenhum arquivo de pendentes encontrado.")
