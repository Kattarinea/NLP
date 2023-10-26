import pymorphy2
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

def multipreprocessing_text(text):
    filtered_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)
    word_tokens = word_tokenize(filtered_text.lower(), language='russian')
    filtered_text = [word for word in word_tokens if word not in stop_words]
    lem = [morph.normal_forms(word)[0] for word in filtered_text]
    return lem

def multipreprocessing_text_spacy(data):
    result = [token.lemma_.lower() for token in nlp(data) if
            not token.is_stop
            and not token.is_space
            and not token.like_num
            and not token.is_punct
            and not token.is_digit]
    return result
