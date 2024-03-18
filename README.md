# Кто я
Кармадонов Лев 972203

# Как использовать и как работает
При обучении модели она сохраняется в `./data/model/trained_model.pkl`, затем эта же модель используется для предсказаний

Чтобы установить пакет нужно, находясь в директории package, установить пакет при помощи:
```
pip install "./dist/spaceshiptitanichits-0.3.0-py3-none-any.whl"
```
Затем чтобы использовать этот пакет можно вызвать его при помощи:
```
python -m spaceshiptitanic.module [train/predict] --dataset "/path/to/data.csv"
```

Так же можно просто запустить `model.py` с такими же аргументами

Хочется заметить что обучение каждый раз сопровождается поиском оптимальных параметров при помощи optuna - возможно это не самое логичное решение, однако как я думаю, это позволит гибко подбирать параметры в случае внесения изменений в обучающую выборку. Но если же это не так, то надеюсь на обратную связь по этому поводу:)


# Источники
* https://python-poetry.org/docs/ - установка poetry
* https://pythonru.com/uroki/funkcija-main-v-python-dlja-nachinajushhih - __main__
* https://habr.com/ru/articles/704432/ - небольшая статья по optuna
* https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html - большая документация по optuna, частности о том как работает study
* https://docs-python.ru/packages/modul-pandas-analiz-dannykh-python/dataframe-drop/ - статья по pandas .drop 
* https://python-poetry.org/docs/cli/ - документация по использованию poetry в консоли
* https://www.adventuresinmachinelearning.com/python-poetry-an-overview-of-the-ultimate-dependency-management-tool/ - тоже использование poetry в консоли
* https://translated.turbopages.org/proxy_u/en-ru.ru.ea667a8c-65f59b36-4872f990-74722d776562/https/stackoverflow.com/questions/76969532/windows-poetry-isnt-recognized-and-commands-arent-working-although-it-has-alr - помощь в осознании того, что у меня проблема с переменными окружения
* https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax - документация по красивому README
* https://github.com/optuna/optuna-examples - optuna examples (Помогло меньше, чем прочие статьи и документация, но пусть будет)
* https://habr.com/ru/articles/594077/ - статья по catboost, с которой я ознакомился уже после того как переписал функцию для подготовки данных для модели
* Ну и решение,которое сохранилось с предыдущего модуля, поэтому где-то половина от функции с подготовкой данных взята оттуда, ну и работа с моделями происходит примерно по тому же принципу. К сожалению, после переустановки винды, источники которые использовались тогда достать уже не смогу
