При обучении модели она сохраняется в /data/model/trained_model.pkl, затем эта же модель используется для предсказаний

Чтобы создать пакет - нужно из папки package вызвать команду - poetry build, после чего установить пакет при помощи pip install absolute/path/to/project/package/dict/spaceshiptitanichits-0.2.0-py3-none-any.whl
Затем чтобы использовать этот пакет можно вызвать его при помощи - python -m spaceshiptitanic.module [train/predict] --dataset "/path/to/data.csv"

Так же можно просто запустить model.py с такими же аргументами