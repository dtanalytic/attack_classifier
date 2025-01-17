# Настройка
1) Клонировать проект с переходом в нужную ветку
   - git clone https://github.com/dtanalytic/attack_classifier.git attack_classifier --branch aug
2) создать окружение и установить зависимости (+ pytorch)
   - cd attack_classifier
   - pip install -r requirements.txt
   - установить [pytorch и cuda в зависимости от конфигурации](https://pytorch.org/get-started/locally/)
3) Создать каталоги:
   - mkdir data data/interim data/out data/artifacts data/interim/ttp data/out/bert data/out/bert_ttp data/out/ttp
4) Перейти в каталог data и распаковать external, где хранятся скачанные с hugging face предварительные модели (предварительно архивировал - "tar -cvzf external.tar.gz external"):
   - cd data
   - tar -xvf external.tar.gz
     
Опционально
5) Если потребуется вносить изменения, для первичной настройки git-а задать имя пользователя и почту:
   - git config --global user.name "as"
   - git config --global user.email "as@mail.ru"
6) Для воспроизведения пайплайна валидации dvc, в ячейке любого ноутбука скачать составляющие nltk:
   
```
    import nltk
    nltk.download('punkt'),  nltk.download('punkt_tab')
```
# Тренировка
Используется функция train из модуля src.train:

```
from src.train import train

train()
```
# Предсказание
Используется функция predict из модуля src.predict:

```
from src.predict import predict

pred_df = predict(['APT32 compromised McAfee ePO to move laterally by distributing malware as a software deployment task.', 'Monitor executed commands and arguments that may indicate common cryptomining or proxyware functionality.', 'TeamTNT has created system services to execute cryptocurrency mining software',
        'An adversary may abuse configurations where an application has the setuid or setgid bits set in order to get code running in a different (and possibly more privileged) user’s context'])

```
Примеры тренировки и обучения есть в ноутбуке notebooks/use_cases.ipynb
# Воспроизведение валидационного пайплайна dvc
- dvc repro
