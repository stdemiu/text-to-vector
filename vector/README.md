## Импортируем необходимые библиотеки
```python
import re
import math
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
```

- **re** — библиотека для работы с регулярными выражениями, используется для очистки текста.
- **math** — используется для вычисления логарифмов в IDF.
- **defaultdict из collections** — создаёт словарь со значением по умолчанию, используется в подсчёте частот слов.
- **nltk** — библиотека для обработки естественного языка.
- **CountVectorizer, TfidfVectorizer** — классы из sklearn для создания BoW и TF-IDF векторов.
- **Word2Vec, FastText** — модели из библиотеки gensim для создания векторных представлений слов.
- **BertTokenizer, BertModel из transformers** — для токенизации и создания эмбеддингов с помощью модели BERT.
- **pandas** — библиотека для работы с таблицами и сохранения данных в формате CSV.
- **torch** — используется для работы с тензорами, необходимыми для BERT.

## Метод 1: Bag-of-Words (BoW)
### Краткая версия
```python
vectorizer = CountVectorizer()
bow_vectors = vectorizer.fit_transform([raw_text])
bow_df = pd.DataFrame(bow_vectors.toarray(), columns=vectorizer.get_feature_names_out())
bow_df.to_csv('bow_vectors.csv', index=False)
```

**Что делает код:**
- **CountVectorizer()** создает объект, который преобразует текст в Bag-of-Words.
- **fit_transform([raw_text])** обучает объект vectorizer на тексте и создает BoW-вектор, где каждый элемент — это частота слова.
- **DataFrame** создает таблицу, где строки — это частоты слов, а столбцы — слова.
- **to_csv('bow_vectors.csv', index=False)** сохраняет таблицу в файл.

### Полная версия
```python
def bag_of_words(tokens, vocab):
    word_count = defaultdict(int)
    for word in tokens:
        word_count[word] += 1
    bow_vector = [word_count[word] for word in vocab]
    return bow_vector

bow_vector = bag_of_words(tokens, vocab)

with open('bow_vector.csv', 'w', encoding='utf-8') as file:
    file.write(','.join(vocab) + '\n')
    file.write(','.join(map(str, bow_vector)) + '\n')
```

- **defaultdict(int)** создаёт словарь word_count, который будет хранить частоты каждого слова.
- **for word in tokens** перебирает все слова и увеличивает их счетчик в word_count.
- **bow_vector = [word_count[word] for word in vocab]** создаёт список частот для слов в vocab.

Сохраняем BoW-вектор в bow_vector.csv.

## Метод 2: TF-IDF
### Краткая версия
```python
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform([raw_text])
tfidf_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df.to_csv('tfidf_vectors.csv', index=False)
```

- **TfidfVectorizer()** — создаёт объект, чтобы вычислить TF-IDF для текста.
- **fit_transform([raw_text])** обучает векторизатор на тексте и создаёт TF-IDF вектор, где учитывается важность каждого слова.
DataFrame и to_csv сохраняют результат в CSV.

### Полная версия
```python
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
    total_docs = 1
    for word in vocab:
        if word in tokens:
            num_docs_with_word[word] += 1
    idf = {word: math.log(total_docs / (1 + num_docs_with_word[word])) for word in vocab}
    return [idf[word] for word in vocab]

tf_vector = compute_tf(tokens, vocab)
idf_vector = compute_idf(vocab, tokens)
tf_idf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vector)]


with open('tfidf_vector.csv', 'w', encoding='utf-8') as file:
    file.write(','.join(vocab) + '\n')
    file.write(','.join(map(str, tf_idf_vector)) + '\n')
```

- **compute_tf** вычисляет частоты слов (TF) и возвращает список с относительными частотами для каждого слова из vocab.
- **compute_idf** подсчитывает, в скольких документах встречается каждое слово, и вычисляет IDF.
- **tf_idf_vector** умножает TF и IDF, чтобы получить вес каждого слова.
Сохраняем в CSV-файл.


## Метод 3: Word2Vec
### Краткая версия
```python
tokens_list = [tokens]
w2v_model = Word2Vec(tokens_list, vector_size=100, window=5, min_count=1, workers=4)
word2vec_vectors = [w2v_model.wv[word] for word in w2v_model.wv.index_to_key]
word2vec_df = pd.DataFrame(word2vec_vectors, index=w2v_model.wv.index_to_key)
word2vec_df.to_csv('word2vec_vectors.csv', index=True)
```

- **Word2Vec** обучает модель, используя слова в tokens_list, и создаёт векторы для каждого слова.
DataFrame и to_csv сохраняют векторы в CSV.

### Полная версия
```python

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
```

- **generate_random_word_vectors** создаёт случайные векторы для каждого слова.
- **text_vector** усредняет векторы всех слов в тексте.
Сохраняем усреднённый вектор в CSV.

## Метод 4: BERT
### Краткая версия
```python

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer(raw_text, return_tensors='pt', max_length=512, truncation=True)
outputs = model(**inputs)
bert_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
bert_df = pd.DataFrame(bert_embedding)
bert_df.to_csv('bert_embedding.csv', index=False)
```

Что делает код:
- Создаёт BERT-токенизатор и модель.
- **outputs.last_hidden_state.mean(dim=1)** усредняет эмбеддинги по всему предложению.
