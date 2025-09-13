# Classificador de Comentários

Este projeto é uma solução para classificar comentários em português em
três categorias: **Positivo**, **Negativo** e **Neutro**.

Utiliza um pipeline de **Machine Learning** baseado em **TF-IDF +
Regressão Logística** (com balanceamento de classes via oversampling),
além de um módulo de pré-processamento customizado que inclui conversão
de emojis em palavras-chave.

A aplicação web é construída com **Streamlit**, permitindo que usuários
façam upload de um arquivo CSV com comentários para classificação,
validem resultados e baixem os arquivos processados.

foi utilizado o site https://pt.exportcomments.com/ para exportação dos comentários.

------------------------------------------------------------------------

## 🚀 Funcionalidades

-   **Pré-processamento de Texto**
    -   Limpeza de menções, hashtags e URLs.\
    -   Substituição de emojis por palavras (ex.: ❤️ → "amor").\
    -   Normalização de repetições de caracteres.
-   **Classificação Automática**
    -   Combinação de **TF-IDF** com **Logistic Regression**.\
    -   Classes: **Positivo (1)**, **Neutro (0)**, **Negativo (-1)**.
-   **Treinamento Contínuo**
    -   Script para atualizar o dataset com novos comentários validados
        (`update_dataset.py`).\
    -   Re-treino automático do modelo com balanceamento de classes.
-   **Aplicação Web (Streamlit)**
    -   Upload de CSVs com comentários.\
    -   Exibição dos resultados classificados.\
    -   Validação/edição dos comentários dentro da própria interface.\
    -   Download de arquivos classificados ou validados.

------------------------------------------------------------------------

## 📊 Desempenho do Modelo

Último treino realizado (com **2988 comentários válidos** e oversampling
balanceado):

  Classe          Precision   Recall   F1-score   Suporte
  --------------- ----------- -------- ---------- ---------
  Negativo (-1)   0.99        0.99     0.99       413
  Neutro (0)      0.74        0.83     0.78       70
  Positivo (1)    0.93        0.86     0.89       115
  **Acurácia**                         **0.94**   598

📌 **Resumo:** O modelo apresenta excelente desempenho para comentários
**Negativos e Positivos**, e melhora significativa na classe **Neutro**,
que era a mais frágil.

------------------------------------------------------------------------

## 🛠️ Como rodar localmente

1.  Clone o repositório:

    ``` bash
    git clone <url-do-repo>
    cd CommentValidator
    ```

2.  Instale dependências (usando [uv](https://github.com/astral-sh/uv)):

    ``` bash
    uv sync
    ```

3.  Rode a aplicação Streamlit:

    ``` bash
    uv run streamlit run src/app/main.py
    ```

4.  (Opcional) Atualize dataset e re-treine o modelo:

    ``` bash
    uv run python src/jobs/update_dataset.py
    uv run python src/training/train.py
    ```
