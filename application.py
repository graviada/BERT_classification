import re
import numpy as np
from functools import lru_cache

import nltk
import torch
import pymorphy2
import streamlit as st
from nltk.corpus import stopwords
from transformers import BertTokenizerFast, BertForSequenceClassification


ORIGINAL_MODEL_NAME = 'ai-forever/ruBert-base'
MY_MODEL_NAME = 'graviada/ruBERT-medicine-classification'

nltk.download('punkt')
nltk.download('stopwords')
RUSSIAN_STOPWORDS = set(stopwords.words("russian"))

morph = pymorphy2.MorphAnalyzer()

# Чистка текста от лишних символов
def clean_text(text: str) -> str:
    # Удаление тегов HTML
    text = re.sub(r'<[^>]*>', '', text)
    # Удаление тегов в начале абзаца
    text = re.sub(r'[\n\t\br\b]', '', text)
    # Добавление пробелов после символов
    text = re.sub(r'([,.:?])([^\s])', r'\1 \2', text)
    # Разделение предложения, если в слове содержится одна заглавная буква
    text = re.sub(r'(?<=[a-zа-яё])\s*(?=[A-ZА-ЯЁ])', '. ', text)
    text = re.sub(r'(?<=\S)-(?=\s|$)', ' -', text)
    # Добавление пространства между числом и значением (15мм в 15 мм)
    text = re.sub(r'(?<=\d)(?=\D)|(?<=\D)(?=\d)', ' ', text)
    # Удаление двух пробелов
    text = re.sub(r'\s+', ' ', text)
    # Удаление пробела в конце
    text = text.rstrip()
    return text


# Приведение слов к нормальной форме
@lru_cache(maxsize=None) 
def norm_form_cached(word: str):
    return morph.parse(word)[0].normal_form


# Очищает текст от стоп-слов в русском языке
def stop_words_detection(text: str) -> str:
    russian_stopwords = RUSSIAN_STOPWORDS
    text_tokens = nltk.word_tokenize(text)
    cleaned_text = ' '.join(token.strip() for token in text_tokens
                            if token not in russian_stopwords)
    return cleaned_text


# Обработка текстов обращений
def preprocess_case(text: str) -> str:
    text = clean_text(text)
    text = stop_words_detection(text)
    text = ' '.join([norm_form_cached(word) for word in text.split()])
    return text

@st.cache_resource()
def get_model():
    tokenizer = BertTokenizerFast.from_pretrained(ORIGINAL_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MY_MODEL_NAME)
    return tokenizer, model


tokenizer, model = get_model()

user_input = st.text_area('Опишите свою жалобу ниже')
button = st.button('Отправить')

values = {
    0: 'Акушерство - Беременность-Роды',
    1: 'Аллергология - Иммунология',
    2: 'Гастроэнтерология',
    3: 'Гинекология',
    4: 'Дерматология и венерология',
    5: 'Инфекционные болезни',
    6: 'Кардиология',
    7: 'Неврология',
    8: 'Онкология и маммология',
    9: 'Остальные разделы медицины',
    10: 'Оториноларингология',
    11: 'Офтальмология',
    12: 'Педиатрия',
    13: 'Психология - психотерапия',
    14: 'Терапия',
    15: 'Травматология',
    16: 'Урология',
    17: 'Хирургия'
}

doctors = {
    'Остальные разделы медицины': 'Общий врач',
    'Урология': 'Уролог',
    'Оториноларингология': 'ЛОР (оториноларинголог)',
    'Травматология': 'Травматолог',
    'Акушерство - Беременность-Роды': 'Акушер-гинеколог',
    'Терапия': 'Терапевт',
    'Гинекология': 'Акушер-гинеколог',
    'Хирургия': 'Хирург',
    'Кардиология': 'Кардиолог',
    'Инфекционные болезни': 'Инфекционист',
    'Неврология': 'Невролог',
    'Гастроэнтерология': 'Гастроэнтеролог',
    'Дерматология и венерология': 'Дерматолог-венеролог',
    'Аллергология - Иммунология': 'Аллерголог-иммунолог',
    'Психология - психотерапия': 'Психолог-психотерапевт',
    'Онкология и маммология': 'Онколог',
    'Офтальмология': 'Офтальмолог',
    'Педиатрия': 'Педиатр'
}

if user_input and button:
    user_input = preprocess_case(user_input)
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=250, return_tensors='pt')
    output = model(**test_sample)
    # st.write("Logits: ", output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    st.write('Предполагаемый раздел медицины: ', values[y_pred[0]])
    st.write('Вы можете обратиться к профильному специалисту. Специалист: ', doctors[values[y_pred[0]]])