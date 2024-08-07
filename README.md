# BERT
## Задача
Решаем задачу Sentiment Analysis, мультиклассовая классификация текста по его тональности, используя модель BERT.

## Стек

Python, pandas, numpy, sklearn, SQL, BERT

## Решение

Первым шагом сделаем выгрузку датасета с отзывами из базы данных (файл query_BERT.sql).
Вторым шагом напишем утилиту, которая загружает i-ый батч из скачанного датасета и токенизирует его. В labels возвращаются -1, 0 или +1, в зависимости от сентимента отзыва.
Для того, чтобы унифицировать длину текстов воспольемся таким приемом как паддинг. Существуют два вида паддинга:
- Обычный паддинг (фиксированная длина, равная заданному лимиту).
- Динамический паддинг (все элементы дополняются по длине до самого длинного).

Реализуем оба вида паддинга, добавив их в класс DataLoader. Для этого напишем вспомогательную функцию attention_mask, зануляющую веса пустых токенов. Тогда модель хоть и будет обрабатывать пустые токены вместе с остальными в общем ключе, но их эффект и влияние на результат вычислений будут нивелированы.
Итерируемся по батчам нашего датасета, забираем токены (уже с паддингом) и метки классов.
Используем модель DistilBERT из библиотекой transformers от HuggingFace. Задаём attention mask и применяем DistilBERT к нашему батчу.
Наконец, берём последний скрытый слой трансформера – контексто-обусловленные эмбеддинги.
Первый эмбеддинг относится к [CLS] и является sentence embedding для всей строки. 
Так мы получаем эмбеддинги отзывов.
Для нашей задачи (мультиклассификация на 3 класса) мы обучаем логистическую регрессию поверх выходного эмбеддинга [CLS] токена, который является хорошей репрезентацией всего текста.
Возьмём кросс-валидацию на 5 фолдов (без перемешивания).

## Итоги:
- Выгрузили датасет с отзывами FlyingFood с помощью SQL-запроса
- Провели токенизацию текста в режиме обработки по батчам и добавили паддинг
- Токенизированный текст прогнали через BERT и получили sentence embeddings
- На полученных эмбеддингах обучили логистическую регрессию
- Наконец, оценили её качество с помощью Cross-Entropy Loss
