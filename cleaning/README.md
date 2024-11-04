Этот код выполняет очистку текста, извлекая из него только полезные слова и избавляясь от ненужных символов, стоп-слов и знаков препинания.

## Импортируем библиотеки
```python
import re
import nltk
from nltk.corpus import stopwords
```
- **re** — библиотека для работы с регулярными выражениями, которая позволяет находить и заменять шаблоны в тексте, например, удалять ненужные символы.
- **nltk (Natural Language Toolkit)** — библиотека для обработки естественного языка. Мы используем её для работы со стоп-словами.
- **stopwords из nltk.corpus** — содержит часто встречающиеся, но малоинформативные слова (стоп-слова), такие как "and", "the", "is" и т.д., которые не добавляют смысла в анализируемый текст.

## Загружаем стоп-слова
```python
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
```

- **nltk.download('stopwords')** — загружает список стоп-слов из NLTK. Это нужно выполнить один раз.
- **stopwords.words('english')** — возвращает список стоп-слов для английского языка.
- **set(stopwords.words('english'))** — сохраняем стоп-слова в виде множества для более быстрого поиска при обработке текста.

## Чтение текста из CSV-файла
```python
with open('sherlock_holmes_text.csv', 'r', encoding='utf-8') as file:
    raw_text = file.read()
```

- Открываем файл sherlock_holmes_text.csv, чтобы прочитать текст.
- file.read() считывает весь текст из файла и сохраняет его в переменной raw_text.

## Функция для очистки текста
```python
def preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')  
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)
```

- **text.encode('ascii', 'ignore').decode('ascii')** — кодирует текст в ASCII, удаляя все символы, которые не соответствуют ASCII-кодировке. Например, если в тексте есть специальные символы или знаки, они будут убраны.
- **re.sub(r'[^\w\s]', '', text.lower())** — удаляет все знаки препинания, оставляя только буквы, цифры и пробелы:
  - **text.lower()** — переводит текст в нижний регистр.
  - **re.sub(r'[^\w\s]', '', ...)** — регулярное выражение, которое убирает все символы, не являющиеся буквами, цифрами или пробелами.
- **words = [word for word in text.split() if word not in stop_words]** — разделяет текст на слова (токены) и удаляет стоп-слова:
  - **text.split()** — разбивает текст по пробелам на отдельные слова.
  - **[word for word in text.split() if word not in stop_words]** — сохраняет только те слова, которые не являются стоп-словами.
- **return ' '.join(words)** — объединяет оставшиеся слова обратно в строку с пробелами.

## Очищаем текст
```python
cleaned_text = preprocess_text(raw_text)
```
- Вызываем функцию preprocess_text для переменной raw_text, очищая текст от ненужных символов и слов. Результат сохраняется в переменной cleaned_text.

## Сохраняем очищенный текст в новый файл
```python
with open('cleaned_text.csv', 'w', encoding='utf-8') as file:
    file.write("Cleaned_Text\n")
    file.write(cleaned_text)
```

- **with open('cleaned_text.csv', 'w', encoding='utf-8') as file:** — открывает файл cleaned_text.csv для записи.
  - **mode='w'** — открытие файла в режиме записи, что создаст новый файл, если его не существует, или перезапишет его, если он уже есть.
- **file.write("Cleaned_Text\n")** — добавляет заголовок Cleaned_Text в первой строке.
- **file.write(cleaned_text)** — записывает очищенный текст в файл.

Итак, этот код считывает текст из CSV-файла, очищает его от ненужных символов и слов, и сохраняет его в новый CSV-файл с названием cleaned_text.csv.
