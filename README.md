# Classificador de Comentários

Este projeto é uma solução para classificar comentários em português em três categorias: **Positivo**, **Negativo** e **Neutro**. Utilizamos um pipeline híbrido que combina **TF-IDF** com um transformador de léxico customizado (LexiconScoreTransformer), permitindo uma análise mais refinada das entradas – inclusive a conversão de emojis em palavras-chaves.

A aplicação web é construída com **Streamlit**, permitindo que usuários façam upload de um arquivo CSV com comentários para classificação e baixem os resultados.

## Funcionalidades

- **Pré-processamento de Texto:**  
  - Limpeza dos comentários, substituição de emojis por palavras (por exemplo, "❤️" por "amor").
  - Remoção de menções, hashtags e URLs.
  - Preservação de acentos e redução de repetições (mantendo repetições duplas).

- **Análise Híbrida:**  
  - Combinação de **TF-IDF** com um léxico customizado para pontuar palavras e emojis.
  - Classificação dos comentários em **Positivo**, **Negativo** ou **Neutro**.

- **Treinamento e Atualização do Modelo:**  
  - Treinamento com um dataset atualizado (possibilidade de ser enriquecido com novos comentários reais).
  - Atualização do dataset com novos comentários através de um script específico.

- **Aplicação Web com Streamlit:**  
  - Upload de CSV com os comentários originais.
  - Exibição dos resultados, com o comentário original e a classificação.
  - Download do CSV com as predições.



