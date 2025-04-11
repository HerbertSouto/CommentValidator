import re
import unicodedata
import pandas as pd

# Dicionário de emojis e suas traduções
emoji_dict = {
    "❤️": " amor ",
    "♥️": " amor ",
    "💚": " amor ",
    "🩷": " amor ",
    "💖": " amor ",
    "🫶": " amor ",        
    "🥰": " amor ",          
    "🔥": " quente ",
    "✨": " brilho ",
    "😍": " encantado ",
    "👏": " aplauso ",
    "😂": " risos ",
    "😊": " feliz ",
    "😭": " triste ",
    "😡": " irritado ",
    "🙄": " desdenhoso ",    
    "👍": "positivo ",
    "👎": " negativo ",
    "💥": " explosivo ",
    "💫": " deslumbrante ",
    "💯": " perfeito ",
    "👌": " ok ",
    "😢": " triste ",
    "😎": " descolado ",
    "😉": " sutil ",
    "😘": " beijo ",
    "💋": " beijo ",
    "👄": " beijo ",        
    "😁": " feliz ",
    "😆": " engraçado ",
    "🤩": " estonteante ",
    "🙌": " gratidão ",
    "🤔": " reflexão ",
    "💔": " desoludido ",
    "😒": " desapontado ",
    "🤮": " nojento ",
    "😱": " assustado ",
    "😜": " divertido ",
    "👑": " empoderado ",
    "🔝": " positivo ",    
    "🚀": " subindo ",         
}

def substituir_emojis(texto):
    texto = str(texto)
    for emoji, traducao in emoji_dict.items():
        texto = texto.replace(emoji, traducao)
    return texto

def limpar_texto(texto):
    texto = str(texto)
    texto = substituir_emojis(texto)
    texto = re.sub(r'@[A-Za-z0-9]+', '', texto)
    texto = re.sub(r'#', '', texto)
    texto = re.sub(r'https?:\/\/\S+', '', texto)
    texto = re.sub(r'RT ', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto, flags=re.UNICODE)
    texto = re.sub(r'(.)\1{2,}', r'\1\1', texto)
    return texto.strip().lower()

def ajustar_contexto(texto, lexicon):
    texto = str(texto)
    palavras = texto.split()
    score_total = sum([lexicon.get(p, 0) for p in palavras])
    positivo = any(lexicon.get(p, 0) > 0 for p in palavras)
    negativo = any(lexicon.get(p, 0) < 0 for p in palavras)
    if positivo and negativo:
        return -abs(score_total)
    return score_total

from sklearn.base import BaseEstimator, TransformerMixin

class LexiconScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon, usar_contexto=False, escala=2.0):
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