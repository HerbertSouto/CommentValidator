# Classificador de Comentários

Classifica o sentimento de comentários do Instagram em **Positivo**, **Negativo** e **Neutro** usando o modelo **pysentimiento** (BERTimbau), treinado especificamente em português de redes sociais.

## Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) — Windows, necessário para o PyTorch

## Instalação

```bash
git clone https://github.com/HerbertSouto/CommentValidator.git
cd CommentValidator
uv sync
```

Na primeira execução o modelo (~500MB) é baixado automaticamente do HuggingFace.

## Rodando

```bash
uv run streamlit run src/app/main.py
```

Acesse `http://localhost:8501`.

## Como usar

**Aba Classificar**
1. Upload de um CSV com a coluna `Comment`
2. Aguarda a classificação com barra de progresso
3. Vê resumo por sentimento e tabela com nível de confiança
4. Baixa o CSV classificado ou envia para validação manual

**Aba Validação Manual**
1. Corrige os sentimentos errados via dropdown
2. Marca o checkbox nos comentários revisados
3. Baixa o CSV validado

## Formato do CSV de entrada

O arquivo deve conter ao menos a coluna `Comment`:

```
Comment
Que conteúdo incrível!
Não gostei nada...
```

Arquivos exportados do [exportcomments.com](https://pt.exportcomments.com) já vêm no formato correto.

## Modos de funcionamento

O classificador detecta automaticamente o melhor modo disponível:

| Modo | Quando ativa | Qualidade |
|------|-------------|-----------|
| **Local** (pysentimiento) | PyTorch disponível | Melhor |
| **API** (HuggingFace) | `HF_TOKEN` definido | Boa |
| **TF-IDF** (fallback) | Nenhum dos anteriores | Básica |

Para usar o modo API, crie um token gratuito em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) e defina no ambiente:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

## Re-treinar o modelo TF-IDF (fallback)

```bash
uv run python src/training/train.py
```
