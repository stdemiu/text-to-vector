import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('sherlock_holmes_text.csv', 'r', encoding='utf-8') as file:
    raw_text = file.read()

def preprocess_text(text):
    text = text.encode('ascii', 'ignore').decode('ascii')  
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

cleaned_text = preprocess_text(raw_text)

with open('cleaned_text.csv', 'w', encoding='utf-8') as file:
    file.write("Cleaned_Text\n")
    file.write(cleaned_text)

print("Файл cleaned_text.csv успешно сохранен с очищенным текстом.")

