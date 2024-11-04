import re
import math
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch

# Загружаем и очищаем текст
with open('cleaned_text.csv', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# Предобработка текста: приведение к нижнему регистру, удаление знаков препинания и разбиение на токены
def preprocess_text(text):
    # Убираем знаки препинания, приводим к нижнему регистру, разбиваем текст на слова
    text = re.sub(r'[^\w\s]', '', text.lower())  # Преобразует текст к нижнему регистру, убирает знаки
    return text.split()  # Разделяет текст по пробелам и создает список слов (токенов)

tokens = preprocess_text(raw_text)
vocab = sorted(set(tokens))  # Создаем словарь уникальных слов для модели

# 1. Bag-of-Words (BoW)
# Краткая версия
vectorizer = CountVectorizer()
bow_vectors = vectorizer.fit_transform([raw_text])
bow_df = pd.DataFrame(bow_vectors.toarray(), columns=vectorizer.get_feature_names_out())
bow_df.to_csv('bow_vectors.csv', index=False)

# Полная версия
def bag_of_words(tokens, vocab):
    word_count = defaultdict(int)  # Создаем словарь для хранения частоты слов
    for word in tokens:  # Перебираем все слова в тексте
        word_count[word] += 1  # Увеличиваем счетчик для каждого слова
    bow_vector = [word_count[word] for word in vocab]  # Создаем вектор с частотами для каждого слова в словаре
    return bow_vector

bow_vector = bag_of_words(tokens, vocab)  # Применяем функцию для создания BoW

# Сохраняем BoW вектор
with open('bow_vector.csv', 'w', encoding='utf-8') as file:
    file.write(','.join(vocab) + '\n')  # Записываем слова как заголовки столбцов
    file.write(','.join(map(str, bow_vector)) + '\n')  # Записываем частоты слов

# 2. TF-IDF
# Краткая версия
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform([raw_text])
tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.to_csv('tfidf_vectors.csv', index=False)

# Полная версия
def compute_tf(tokens, vocab):
    tf = defaultdict(float)
    total_terms = len(tokens)
    for word in tokens:
        tf[word] += 1
    tf = {word: (count / total_terms) for word, count in tf.items()}
    return [tf.get(word, 0) for word in vocab]

def compute_idf(vocab, tokens):
    idf = defaultdict(float)
    num_docs_with_word = defaultdict(int)
    total_docs = 1  # Один документ (текущий текст)
    for word in vocab:
        if word in tokens:
            num_docs_with_word[word] += 1
    idf = {word: math.log(total_docs / (1 + num_docs_with_word[word])) for word in vocab}
    return [idf[word] for word in vocab]

tf_vector = compute_tf(tokens, vocab)
idf_vector = compute_idf(vocab, tokens)
tf_idf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vector)]

# Сохраняем TF-IDF вектор
with open('tfidf_vector.csv', 'w', encoding='utf-8') as file:
    file.write(','.join(vocab) + '\n')
    file.write(','.join(map(str, tf_idf_vector)) + '\n')

# 3. Word2Vec
# Краткая версия
tokens_list = [tokens]
w2v_model = Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
word2vec_vectors = [w2v_model.wv[word] for word in w2v_model.wv.index_to_key]
word2vec_df = pd.DataFrame(word2vec_vectors, index=w2v_model.wv.index_to_key)
word2vec_df.to_csv('word2vec_vectors.csv', index=True)

# Полная версия
def generate_random_word_vectors(vocab, vector_dim=100):
    random_vectors = {}
    for word in vocab:
        random_vectors[word] = [(hash(word) % 100) / 100 for _ in range(vector_dim)]
    return random_vectors

word_vectors = generate_random_word_vectors(vocab)
text_vector = [0] * 100

for word in tokens:
    if word in word_vectors:
        text_vector = [sum(x) for x in zip(text_vector, word_vectors[word])]
text_vector = [x / len(tokens) for x in text_vector]

with open('word2vec_average.csv', 'w', encoding='utf-8') as file:
    file.write(','.join(map(str, text_vector)) + '\n')

# 4. FastText (без изменений)
# Краткая версия
fasttext_model = FastText(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
fasttext_vectors = [fasttext_model.wv[word] for word in fasttext_model.wv.index_to_key]
fasttext_df = pd.DataFrame(fasttext_vectors, index=fasttext_model.wv.index_to_key)
fasttext_df.to_csv('fasttext_vectors.csv', index=True)

# Полная версия не нужна, так как FastText работает аналогично Word2Vec, но дополнительно учитывает подслова.

# 5. BERT
# Краткая версия
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(raw_text, return_tensors='pt', max_length=512, truncation=True)
outputs = model(**inputs)
bert_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
bert_df = pd.DataFrame(bert_embedding)
bert_df.to_csv('bert_embedding.csv', index=False)

# Полная версия
# BERT токенизирует текст и создает эмбеддинги, учитывая контекст слов.
# 1. Создаем `BertTokenizer` и `BertModel` для токенизации текста и генерации эмбеддингов.
# 2. `tokenizer(...)` разделяет текст на токены, создает входные данные для BERT.
# 3. `model(**inputs)` пропускает текст через модель и возвращает эмбеддинги для каждого токена.
# 4. `outputs.last_hidden_state.mean(dim=1)` усредняет эмбеддинги, чтобы получить общее представление текста.
# 5. Сохраняем усредненный BERT-эмбеддинг в `bert_embedding.csv`.

print("Все файлы успешно сохранены: bow_vector.csv, tfidf_vector.csv, word2vec_average.csv, word2vec_vectors.csv, fasttext_vectors.csv, bert_embedding.csv.")
