import requests
from bs4 import BeautifulSoup
import csv


url = "https://sherlock-holm.es/stories/html/cano.html"


response = requests.get(url)


if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    book_text = ""
    for element in soup.find_all(['p', 'div'], class_=['letter', 'letter-block', 'letter-signature']):
        book_text += element.text.strip() + "\n"

    with open('sherlock_holmes_text.csv', mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Text"])
        writer.writerow([book_text])

    print("Текст успешно сохранен в sherlock_holmes_text.csv")
else:
    print("Не удалось загрузить страницу")
