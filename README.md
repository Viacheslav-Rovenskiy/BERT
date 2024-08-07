# BERT

Данная задача называется Sentiment Analysis, классификация текста по его тональности:

Мы будем решать задачу, используя BERT. 

Еще раз подчеркнем, что предобучение модели под обе задачи (MLM и NSP) происходит одновременно.

За счёт этого на выходе модели мы получаем:

Контексто-обусловленные репрезентации отдельных слов
Хорошую репрезентацию всего текста в токене [CLS]

Предобучив BERT (pre-training stage), мы можем подогнать его уже под нашу узкую задачу (fine-tuning stage), чтобы использовать выученные общие паттерны языка, например, для упомянутого Sentiment Analysis. 

Для задач классификации обучают логистическую регрессию поверх выходного эмбеддинга [CLS] токена, который является хорошей репрезентацией всего текста.

Сделаем выгрузку датасета с отзывами из базы данных.

Он лежит в ClickHouse, в таблице: simulator.flyingfood_reviews

Мы воспользуемся библиотекой transformers от HuggingFace

После установки библиотеки импортируем модель и токенизатор. Мы будем использовать DistilBERT, маленькую, легковесную и быструю версию BERT-архитектуры

На выходе мы получаем индексы слов, пригодные для модели.
