import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def multipreprocessing_text(text):
    filtered_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
    word_tokens = word_tokenize(filtered_text.lower(), language='russian')
    filtered_text = [word for word in word_tokens if word not in stop_words]
    lem = [morph.normal_forms(word)[0] for word in filtered_text]
    return lem
