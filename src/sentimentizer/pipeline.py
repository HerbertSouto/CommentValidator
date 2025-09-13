import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Baixar stopwords se necessário
nltk.download('stopwords')
portuguese_stopwords = stopwords.words('portuguese')

# Configuração do TfidfVectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    stop_words=portuguese_stopwords
)

# Lexicon customizado expandido (incluindo termos derivados dos emojis)
lexicon_palavras = {
    # Termos positivos
    "show": 1,
    "sensacional": 1.5,
    "incrivel": 1.5,
    "excelente": 2,
    "excelentes": 2,
    "bom": 1,
    "boa": 1,
    "otimo": 1.5,
    "otima": 1.5,
    "fantastico": 2,
    "fantastica": 2,
    "adoro": 1,
    "amei": 1,
    "maravilhoso": 2,
    "maravilhosa": 2,
    "recomendo": 1,
    "engajado": 1,
    "engajada": 1,
    "influente": 1.5,
    "carismatico": 1,
    "carismatica": 1,
    # Derivados de emojis positivos:
    "amor": 1.5,
    "brilho": 1,
    "encantado": 1.5,
    "quente": 1,
    "aplauso": 1,
    "divertido": 1,
    "feliz": 1,
    "perfeito": 1.5,
    "deslumbrante": 1.5,
    "ok": 0.5,
    
    # Termos negativos
    "ruim": -1,
    "pesimo": -1.5,
    "pessimo": -1.5,
    "horrivel": -2,
    "horrível": -2,
    "decepcionante": -1,
    "chato": -0.5,
    "chata": -0.5,
    "terrivel": -2,
    "desastroso": -2,
    "defeituoso": -1.5,
    "defeituosa": -1.5,
    # Derivados de emojis negativos:
    "irritado": -1.5,
    "triste": -0.5,
    "negativo": -1,
    "nojento": -1.5,
    "assustado": -1,
    
    # Termos neutros
    "mediano": 0,
    "mediana": 0,
    "normal": 0,
    "regular": 0,
    "comum": 0,
    "basico": 0,
    "padrao": 0,
    "convencional": 0,
    "simples": 0,
    "neutro": 0,
    "moderado": 0,
    "moderada": 0,
    # Derivados de emojis neutros (exemplo)
    "explosivo": 0
}

# Importa funções de pré-processamento do módulo preprocess.py
from preprocess import ajustar_contexto

# Define o transformador do lexicon
class LexiconScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon, usar_contexto=False, escala=1.0):
        self.lexicon = lexicon
        self.usar_contexto = usar_contexto
        self.escala = escala

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scores = []
        for texto in X:
            if self.usar_contexto:
                score = ajustar_contexto(texto, self.lexicon)
            else:
                palavras = texto.split()
                score = sum([self.lexicon.get(p, 0) for p in palavras])
            scores.append(score * self.escala)
        return pd.DataFrame(scores, columns=["lexicon_score"])

# Definição do pipeline híbrido
pipeline_hibrido = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", tfidf),
        ("lexicon", LexiconScoreTransformer(lexicon=lexicon_palavras, usar_contexto=True, escala=1.0))
    ])),
    ("clf", LogisticRegression(max_iter=1000))
])

# Definição do pipeline apenas TF-IDF
pipeline_tfidf = Pipeline([
    ("features", tfidf),
    ("clf", LogisticRegression(max_iter=1000))
])
