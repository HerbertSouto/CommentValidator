import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentimentizer.analyzer import get_analyzer, classificar

PENDENTES_PATH = "data/raw/pendentes_validacao.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# Deve ser a primeira chamada Streamlit — define aba do browser e layout.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classificador de Comentários",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL
# Importa DM Sans (corpo) e DM Mono (dados/números) do Google Fonts.
# Customiza componentes Streamlit que o config.toml não cobre.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fontes ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Números e dados tabulares usam fonte mono — leitura mais precisa */
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"],
.dataframe td {
    font-family: 'DM Mono', monospace !important;
}

/* ── Cabeçalho ──────────────────────────────────────────────────────────── */
h1 {
    font-weight: 600;
    font-size: 1.75rem !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem !important;
}

/* ── Cards de métrica ───────────────────────────────────────────────────── */
/* Usa var() do Streamlit — funciona em tema claro e escuro sem hardcode */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #6366f1;
}

[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.55;
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 500;
}

[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
}

/* ── Botão primário ─────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    border: none !important;
    color: #fff !important;         /* texto sempre branco — contraste garantido */
    font-weight: 500;
    letter-spacing: 0.01em;
    border-radius: 8px;
    transition: opacity 0.15s, transform 0.1s;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Botão secundário e download ────────────────────────────────────────── */
/* Borda com a cor de texto atual — legível em qualquer tema */
.stButton > button[kind="secondary"],
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    color: var(--text-color) !important;
    border-radius: 8px;
    transition: border-color 0.15s;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
    border-color: #6366f1 !important;
}

/* ── Upload area ────────────────────────────────────────────────────────── */
/* Remove background hardcoded — herda do tema; mantém só a borda e raio */
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #6366f1 !important;
}

/* Botão "Browse files" dentro do uploader — sempre legível */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border: 1px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 6px !important;
}

/* ── Abas ───────────────────────────────────────────────────────────────── */
[data-baseweb="tab"] {
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #6366f1 !important;
}

/* ── Progress bar com gradiente indigo ──────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a5b4fc) !important;
    border-radius: 99px;
}

/* ── Divisória ──────────────────────────────────────────────────────────── */
hr {
    margin: 1.25rem 0 !important;
    opacity: 0.15;
}

/* ── Alertas ────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.85rem;
}

/* ── Cabeçalho da tabela em uppercase discreto ──────────────────────────── */
[data-testid="stDataFrame"] th {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    opacity: 0.55;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO
# @st.cache_resource garante que o modelo é carregado uma única vez por sessão
# do servidor — reutilizado por todos os usuários sem recarregar (~500MB).
# O spinner aparece antes de qualquer renderização para evitar layout shift.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    return get_analyzer()


with st.spinner("Iniciando classificador..."):
    modo = carregar_modelo()


# ─────────────────────────────────────────────────────────────────────────────
# CABEÇALHO
# Título + caption numa linha; badge de status no canto direito.
# ─────────────────────────────────────────────────────────────────────────────
col_titulo, col_status = st.columns([5, 1])

with col_titulo:
    st.title("💬 Classificador de Comentários")
    st.caption("Analise o sentimento de comentários do Instagram em segundos.")

with col_status:
    # Badge de modo — só mostra aviso se estiver no fallback TF-IDF
    if modo in ("local", "api"):
        st.success("Online", icon="✅")
    else:
        st.warning("Modo básico", icon="⚠️")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ABAS PRINCIPAIS
# Separação clara entre fluxo de classificação e fluxo de validação.
# Tabs evitam scroll infinito e deixam cada função no seu contexto.
# ─────────────────────────────────────────────────────────────────────────────
aba_classificar, aba_validar = st.tabs(["📂  Classificar", "✏️  Validação Manual"])


# ═════════════════════════════════════════════════════════════════════════════
# ABA 1 — CLASSIFICAR
# ═════════════════════════════════════════════════════════════════════════════
with aba_classificar:

    # ── Upload ───────────────────────────────────────────────────────────────
    # file_uploader aceita apenas CSV; mensagem de ajuda inline.
    uploaded_file = st.file_uploader(
        "Selecione o arquivo CSV de comentários",
        type=["csv"],
        help="O arquivo deve conter uma coluna chamada **Comment**.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        # Estado vazio — instrução central, não deixar a tela em branco
        st.markdown(
            "<p style='color:#475569; font-size:0.9rem; margin-top:0.5rem;'>"
            "Arraste um arquivo CSV aqui ou clique para selecionar.</p>",
            unsafe_allow_html=True,
        )

    else:
        # ── Leitura do CSV ────────────────────────────────────────────────────
        try:
            df = pd.read_csv(uploaded_file, usecols=["Comment"])
        except Exception:
            st.error(
                "Não foi possível ler o arquivo. "
                "Verifique se ele contém a coluna **Comment**.",
                icon="🚨",
            )
            st.stop()

        df["Comment"] = df["Comment"].fillna("").astype(str)
        validos = df["Comment"].str.strip().ne("")
        total = int(validos.sum())

        if not validos.any():
            st.warning("Nenhum comentário válido encontrado no arquivo.", icon="⚠️")
            st.stop()

        # ── Classificação com progress bar real ───────────────────────────────
        # Atualiza a barra a cada comentário — dá feedback concreto ao usuário
        # em vez de um spinner que não informa quanto falta.
        barra = st.progress(0, text=f"Analisando comentários...  0 / {total}")
        resultados = []

        for i, texto in enumerate(df.loc[validos, "Comment"], start=1):
            resultados.append(classificar(texto))
            pct = i / total
            barra.progress(pct, text=f"Analisando comentários...  {i} / {total}")

        barra.empty()  # Remove a barra após concluir — não deixa "100%" estático

        # ── Monta colunas de resultado ────────────────────────────────────────
        df.loc[validos, "Classe"]      = [r["label"]              for r in resultados]
        # Multiplica por 100 para exibir como percentual inteiro (ex: 99, não 0.99)
        df.loc[validos, "Confiança"]   = [r["confianca"] * 100    for r in resultados]
        df.loc[validos, "_classe_num"] = [r["classe"]              for r in resultados]

        df["Classe"]    = df["Classe"].fillna("Neutro")
        df["Confiança"] = df["Confiança"].fillna(0.0)

        # ── Métricas de resumo ────────────────────────────────────────────────
        # Três cards com contagem + percentual de cada sentimento.
        # delta_color="inverse" nos negativos: seta vermelha = algo a melhorar.
        st.subheader("Resumo", divider=False)
        contagem = df["Classe"].value_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Positivos ✅",
            contagem.get("Positivo", 0),
            f"{contagem.get('Positivo', 0) / total:.0%} do total",
        )
        c2.metric(
            "Negativos ❌",
            contagem.get("Negativo", 0),
            f"{contagem.get('Negativo', 0) / total:.0%} do total",
            delta_color="inverse",
        )
        c3.metric(
            "Neutros ➖",
            contagem.get("Neutro", 0),
            f"{contagem.get('Neutro', 0) / total:.0%} do total",
            delta_color="off",
        )

        st.divider()

        # ── Tabela de resultados ──────────────────────────────────────────────
        # column_config define label, largura e tipo de cada coluna.
        # ProgressColumn para confiança dá leitura visual imediata.
        st.subheader("Comentários Classificados", divider=False)

        st.dataframe(
            df[["Comment", "Classe", "Confiança"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Comment": st.column_config.TextColumn(
                    "Comentário",
                    width="large",
                ),
                "Classe": st.column_config.TextColumn(
                    "Sentimento",
                    width="small",
                ),
                "Confiança": st.column_config.ProgressColumn(
                    "Confiança",
                    min_value=0,
                    max_value=100,
                    format="%d%%",  # valor já está em 0-100, exibe como "99%"
                    width="small",
                ),
            },
        )

        # ── Ações pós-classificação ───────────────────────────────────────────
        # Dois botões lado a lado; download à esquerda (ação principal),
        # validação à direita (fluxo secundário opcional).
        st.divider()
        col_dl, col_val, _ = st.columns([1.2, 1.2, 3])

        csv_bytes = (
            df[["Comment", "Classe"]]
            .assign(Confianca=df["Confiança"].apply(lambda v: f"{v:.0%}"))
            .to_csv(index=False, sep=";", encoding="utf-8")
            .encode("utf-8")
        )
        col_dl.download_button(
            label="⬇️  Baixar CSV",
            data=csv_bytes,
            file_name="comentarios_classificados.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if col_val.button("✏️  Enviar para Validação", use_container_width=True):
            df_val = df[["Comment", "Classe"]].copy()
            df_val["Validado"] = "Não"
            os.makedirs(os.path.dirname(PENDENTES_PATH), exist_ok=True)
            df_val.to_csv(PENDENTES_PATH, sep=";", index=False, encoding="utf-8")
            st.success("Enviado! Abra a aba **Validação Manual** para revisar.", icon="✅")


# ═════════════════════════════════════════════════════════════════════════════
# ABA 2 — VALIDAÇÃO MANUAL
# Permite revisar e corrigir predições antes de usar os dados para re-treino.
# ═════════════════════════════════════════════════════════════════════════════
with aba_validar:

    if not os.path.exists(PENDENTES_PATH):
        # Estado vazio — orienta o usuário sem deixar a tela "quebrada"
        st.info(
            "Nenhuma fila de validação encontrada.  \n"
            "Classifique um CSV na aba **Classificar** e clique em "
            "**Enviar para Validação**.",
            icon="ℹ️",
        )
    else:
        df_pend = pd.read_csv(PENDENTES_PATH, sep=";", encoding="utf-8-sig")

        if "Validado" not in df_pend.columns:
            st.warning("Arquivo de validação sem coluna 'Validado'.", icon="⚠️")
        else:
            total_p = len(df_pend)

            # Converte "Sim"/"Não" → bool para usar CheckboxColumn
            # Checkbox é muito mais rápido de marcar do que um selectbox
            df_pend["Validado"] = df_pend["Validado"].astype(str).str.lower() == "sim"

            # ── Editor de validação ───────────────────────────────────────────
            edited_df = st.data_editor(
                df_pend,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",   # evita adição acidental de linhas
                column_config={
                    "Comment": st.column_config.TextColumn(
                        "Comentário",
                        width="large",
                        disabled=True,          # texto original não é editável
                    ),
                    "Classe": st.column_config.SelectboxColumn(
                        "Sentimento",
                        options=["Positivo", "Negativo", "Neutro"],
                        width="small",
                    ),
                    "Validado": st.column_config.CheckboxColumn(
                        "Validado?",
                        width="small",
                        default=False,
                    ),
                },
            )

            # ── Progresso em tempo real (lê do edited_df) ─────────────────────
            # Com checkbox o estado é imediato — o contador acompanha cada clique
            validados = int(edited_df["Validado"].sum())
            pct = validados / total_p if total_p else 0

            st.progress(pct, text=f"**{validados} de {total_p}** comentários validados")

            st.divider()

            # ── Download ──────────────────────────────────────────────────────
            # Exporta todas as linhas com os sentimentos corrigidos.
            # Nenhum arquivo é salvo no servidor — tudo em memória.
            csv_val = (
                edited_df[["Comment", "Classe"]]
                .to_csv(index=False, sep=";", encoding="utf-8")
                .encode("utf-8")
            )
            st.download_button(
                label="⬇️  Baixar CSV Validado",
                data=csv_val,
                file_name="comentarios_validados.csv",
                mime="text/csv",
            )
