В этом коде мы загружаем веб-страницу с рассказами Шерлока Холмса, используя библиотеку requests, анализируем HTML-контент с помощью BeautifulSoup, а затем извлекаем текст и сохраняем его в CSV-файл. Вот как это происходит:
## Импорт библиотек

```python
import requests
from bs4 import BeautifulSoup
import csv
```
