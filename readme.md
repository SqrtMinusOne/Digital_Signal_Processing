# Цифровая обработка сигналов на Python

## Установка
1. Установить Python >= 3.6
    * Можно [скачать с официального сайта](https://www.python.org/downloads/)  
    Для Windows стоит добавить в PATH, чтобы было проще вызывать
    * Или установить в [Anaconda](https://www.anaconda.com/distribution/)
    * Или `sudo apt install python3`
2. Скачать репозиторий  
    `git clone https://github.com/SqrtMinusOne/Digital_Signal_Processing.git`  
    `cd Digital_Signal_Processing`
3. Рекомендуется создать Virtual Environment
    * Если используется Anaconda, то  
    `conda env create -f environment. yml`  
    `conda activate dsp`
    * Иначе:  
    `pip install virtualenv`  
    `virtualenv venv`  
    `source venv/bin/activate` (Linux) или `env\Scripts\activate` (Windows)  
    `pip install -r requirements.txt`
4. Запуск скриптов: `python [имя-скрипта]`
