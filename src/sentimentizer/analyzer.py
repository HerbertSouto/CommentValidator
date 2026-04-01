"""
Classificador de sentimento para português (redes sociais).

Prioridade de execução:
  1. pysentimiento local (melhor qualidade — requer torch funcionando)
  2. HuggingFace Inference API (requer HF_TOKEN no ambiente)
  3. Modelo TF-IDF treinado localmente (fallback sempre disponível)

Variáveis de ambiente:
  HF_TOKEN  — token gratuito: https://huggingface.co/settings/tokens
"""

import os
import logging

logger = logging.getLogger(__name__)

_HF_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{_HF_MODEL}"
_MODEL_PATH = "models/modelo_hibrido.pkl"

_LABEL_MAP = {
    # pysentimiento
    "POS": (1, "Positivo"),
    "NEG": (-1, "Negativo"),
    "NEU": (0, "Neutro"),
    # HF API (cardiffnlp)
    "positive": (1, "Positivo"),
    "negative": (-1, "Negativo"),
    "neutral": (0, "Neutro"),
}

# Estado do modo ativo
_mode = None          # "local" | "api" | "tfidf"
_local_analyzer = None
_tfidf_model = None


def _try_local() -> bool:
    global _local_analyzer
    try:
        from pysentimiento import create_analyzer
        _local_analyzer = create_analyzer(task="sentiment", lang="pt")
        return True
    except Exception as e:
        logger.warning(f"pysentimiento indisponivel: {e}")
        return False


def _try_api() -> bool:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return False
    # Verifica conectividade com uma chamada rápida
    try:
        import requests
        r = requests.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": "teste"},
            timeout=10,
        )
        return r.status_code != 401
    except Exception:
        return False


def _load_tfidf():
    global _tfidf_model
    import joblib
    if os.path.exists(_MODEL_PATH):
        _tfidf_model = joblib.load(_MODEL_PATH)
        return True
    return False


def get_analyzer() -> str:
    """Inicializa e retorna o modo ativo: 'local', 'api' ou 'tfidf'."""
    global _mode
    if _mode is not None:
        return _mode

    if _try_local():
        _mode = "local"
    elif _try_api():
        _mode = "api"
    else:
        _load_tfidf()
        _mode = "tfidf"

    logger.info(f"Modo de classificacao ativo: {_mode}")
    return _mode


def _classificar_local(texto: str) -> dict:
    resultado = _local_analyzer.predict(texto)
    output = resultado.output
    classe, label = _LABEL_MAP.get(output, (0, "Neutro"))
    return {"classe": classe, "label": label, "confianca": resultado.probas.get(output, 0.0)}


def _classificar_api(texto: str) -> dict:
    import requests
    token = os.getenv("HF_TOKEN", "")
    try:
        resp = requests.post(
            _HF_API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": texto},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        inner = data[0] if isinstance(data[0], list) else data
        melhor = max(inner, key=lambda x: x.get("score", 0))
        output = melhor.get("label", "neutral")
        classe, label = _LABEL_MAP.get(output, (0, "Neutro"))
        return {"classe": classe, "label": label, "confianca": melhor.get("score", 0.0)}
    except Exception as e:
        logger.error(f"Erro HF API: {e}")
        return {"classe": 0, "label": "Neutro", "confianca": 0.0}


def _classificar_tfidf(texto: str) -> dict:
    from sentimentizer.preprocess import limpar_texto
    texto_limpo = limpar_texto(texto)
    classe = int(_tfidf_model.predict([texto_limpo])[0])
    proba = _tfidf_model.predict_proba([texto_limpo])[0]
    confianca = float(max(proba))
    mapa_label = {1: "Positivo", -1: "Negativo", 0: "Neutro"}
    return {"classe": classe, "label": mapa_label.get(classe, "Neutro"), "confianca": confianca}


def classificar(texto: str) -> dict:
    """
    Classifica o sentimento de um texto em português.
    Retorna dict: {classe: int, label: str, confianca: float}
    """
    texto = str(texto).strip()
    if not texto:
        return {"classe": 0, "label": "Neutro", "confianca": 0.0}

    modo = get_analyzer()
    if modo == "local":
        return _classificar_local(texto)
    if modo == "api":
        return _classificar_api(texto)
    return _classificar_tfidf(texto)
