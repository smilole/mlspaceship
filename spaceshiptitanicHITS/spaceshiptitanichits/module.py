# Подключаем необходимые библиотеки
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib
import argparse

# Функция для подготовки данных
def data_preparation(data_frame):

    # Создаем новые столбцы
    data_frame['deck'] = data_frame['Cabin'].str.slice(0, 1)  # deck
    data_frame['num'] = data_frame['Cabin'].str.extract(r'(\d+)')  # num
    data_frame['side'] = data_frame['Cabin'].str.slice(4, 5)  # side

    # Приводим номер каюты к числу
    data_frame['num'] = pd.to_numeric(data_frame['num'], errors='coerce')

    # Удаляем изначальный стобец кают - он больше нам не нужен
    data_frame.drop(columns='Cabin', inplace=True)

    # Заполняем пропуски
    mode_features = ['HomePlanet', 'Destination', 'side', 'num', 'deck'] # так как эти значение ограничены несколькими вариантами - заполняем их по моде

    for f in mode_features:
        # .iloc[0]- получение первой моды из результата вычисления моды
        data_frame[f] = data_frame[f].fillna(data_frame[f].mode().iloc[0])

    # Бинарные признаки
    bin_feats = ['CryoSleep', 'VIP','side']

    # Категориальные признаки
    cat_feats = ['HomePlanet', 'Destination','deck']

    # Преобразование бинарных признаков в числовой формат
    for f in bin_feats:
        map_dict = {value: i for i, value in enumerate(set(data_frame[f]))}
        data_frame[f] = data_frame[f].map(map_dict)

    # Преобразование категориальных признаков при помощи "one-hot encoding"
    # Да, существует библиотека которая это делает автоматически, но частично код остался с прошлого модуля
    for f in cat_feats:
        values = set(data_frame[f])
        for v in values:
            data_frame[f + '_' + str(v)] = data_frame[f] == v
        data_frame = data_frame.drop(columns=f)

    # Заполняем пропуски в числовых столбцах средними значениями и округляем
    numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    for col in numeric_columns:
        mean_value = data_frame[col].mean()
        data_frame[col] = data_frame[col].fillna(mean_value).round()


    #айдишники уникальны, однако из них можно получить полезную информацию, связанную с именами и каютами, но нужно думать как
    data_frame = data_frame.drop(columns='PassengerId')
    data_frame = data_frame.drop(columns='Name')

    # все колонки кроме transported приводим к типу int64 чтобы модель могла без проблем работать с ними
    for col in data_frame.columns:
        if col != 'Transported':
            data_frame[col] = data_frame[col].astype("int64")

    # Небольшой костыль: Если это обучающий датасет - он вернет X с данными пассажиров и y - с данными о transported
    # Ну а если же это уже датасет для предсказания, то в нем нет колонки transported поэтому мы вернем только X - всю таблицу
    if "Transported" not in data_frame.columns:
        X = data_frame.values

        return X

    y = data_frame["Transported"].values
    X = data_frame.drop("Transported", axis=1).values

    return X,y

# Функция для обучения модели и её сохранения
def train(path_to_train_csv):

    # Открываем файл по полученному пути
    data_frame = pd.read_csv(path_to_train_csv)

    # вызываем функцию для подготовки данных для обучения модели
    X, y = data_preparation(data_frame)

    # Разделяем данные на обучающий и валидационный наборы
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        # задаём значения для перебора параметров
        iterations = trial.suggest_int('iterations', 100, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0)
        depth = trial.suggest_int('depth', 1, 10)

        # создаем модель с параметрами
        model = CatBoostClassifier(iterations = iterations, learning_rate = learning_rate, depth = depth)

        # Обучаем модель
        model.fit(X_train, y_train, verbose=False)

        # Делаем предсказания
        pred = model.predict(X_val)

        # вычисляем точность предсказания
        accuracy = accuracy_score(y_val, pred)

        return accuracy

    # Создаем объект для оптимизации
    study = optuna.create_study(direction='maximize')  # Максимизируем метрику производительности

    # оптимизируем запуская objective n_trials раз
    study.optimize(objective, n_trials=100,show_progress_bar=True)

    # получаем лучшие параметры при оптимизации полученные из одного из trials
    best_params = study.best_trial.params

    # Создаем модель с лучшими параметрами для обучения
    model = CatBoostClassifier(
        iterations=best_params['iterations'],
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth']
    )

    # Обучаем модель
    model.fit(X_train, y_train, verbose=False)

    # Сохраняем модель
    joblib.dump(model, "trained_model.pkl")

    print("Model has trained")

    return

# Функция для предсказания при помощи сохраненной модели
def predict(path_to_predict_csv):

    # Открываем файл по полученному пути
    data_frame = pd.read_csv(path_to_predict_csv)

    # Подготавливаем данные для предсказания
    X = data_preparation(data_frame)

    # Достаем ранее сохраненную модель
    model = joblib.load("trained_model.pkl")

    # Делаем предсказание и сохраняем его
    prediction = model.predict(X)

    # Создаем новый датафрейм с айдишника пассажиров и предсказанием
    new_data_frame = pd.DataFrame({'PassengerId': data_frame['PassengerId'],'Transported': prediction})

    # Сохраняем предсказание в csv
    new_data_frame.to_csv("submission.csv", index=False)

    print("prediction saved")

    return


def main():
    # указываем параметры необходимые к передаче
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--dataset", required=True)

    # Считываем параметры
    args = parser.parse_args()

    # Если train - обучаем модель, если predict - делаем предсказание
    if args.mode == "train":
        train(args.dataset)
    elif args.mode == "predict":
        predict(args.dataset)

if __name__ == "__main__":
	main()