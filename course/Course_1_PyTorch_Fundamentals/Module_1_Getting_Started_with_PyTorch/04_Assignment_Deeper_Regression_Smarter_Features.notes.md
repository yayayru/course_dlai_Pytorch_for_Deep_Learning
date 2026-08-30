# Assignment: Deeper Regression, Smarter Features

## Источник

`04_Assignment_Deeper_Regression_Smarter_Features/` (notebook `C1M1_Assignment.ipynb`,
данные `data/data_with_features.csv`, вспомогательные модули `helper_utils.py`,
`unittests.py`, `unittests_utils.py`).

## Конспект по коду

### Назначение

Итоговое (graded) задание модуля: применить всё изученное в модуле к более реалистичной
задаче регрессии — предсказанию времени доставки по **нескольким** признакам, загруженным
из `.csv`-файла (100 исторических записей), а не по одному вручную заданному признаку, как
в лабораторных работах. Задание также впервые вводит **feature engineering** — написание
функции, которая создаёт новый признак из уже существующих данных. Задание разбито на 4
оцениваемых (graded) упражнения с автоматической проверкой через `unittests.py`.

### Ключевые импорты и настройки

- `pandas as pd`, `torch`, `torch.nn as nn`, `torch.optim as optim`, `matplotlib.pyplot as
  plt` — базовые библиотеки.
- `helper_utils` — локальный модуль этого задания (визуализация и форматированный вывод
  предсказания).
- `unittests` — локальный модуль автоматической проверки решений (`exercise_1`…
  `exercise_4`), который, в свою очередь, использует `unittests_utils`
  (вспомогательные функции для тестов: `remove_comments`, `load_rows`).

### Данные

CSV-файл `data/data_with_features.csv` содержит 100 записей о доставках с четырьмя
столбцами:

- `distance_miles` — расстояние доставки в милях (float);
- `time_of_day_hours` — время отправки заказа в часах, 24-часовой формат (float);
- `is_weekend` — бинарный признак дня недели (`1` — выходной, `0` — будний день);
- `delivery_time_minutes` — целевая переменная (target), время доставки в минутах.

Бизнес-правила датасета: доставки происходят только между 8:00 (8.0) и 20:00 (20.0), и
компания не доставляет дальше 20 миль.

### Основные шаги в порядке выполнения (по структуре notebook)

#### Раздел 1 — Multi-Feature Data

**1.1 Загрузка и разведка данных.** `data_df = pd.read_csv('./data/data_with_features.csv')`
— форма `(100, 4)`. Просмотр первых строк через `data_df.head(rows_to_display)`.

Визуализация `helper_utils.plot_delivery_data(data_df)` — scatter-график: X — расстояние,
Y — время доставки, цвет точки — время суток, стиль (закрашенный/полый маркер) — будний
день/выходной. Визуализация показывает: при одинаковом расстоянии некоторые доставки
занимают дольше — предположительно из-за трафика в час пик.

**1.2 Feature Engineering: добавление признака «час пик»** (**Exercise 1 —
`rush_hour_feature`**, GRADED FUNCTION). Новый признак должен быть равен `1`, если
доставка отправлена в утренний час пик (8:00–10:00) или вечерний час пик
(16:00–19:00) **в будний день**, и `0` в остальных случаях (по будням час пик — по
выходным трафик-паттерн не воспроизводится). Решение:

```python
def rush_hour_feature(hours_tensor, weekends_tensor):
    is_morning_rush = (hours_tensor >= 8.0) & (hours_tensor < 10.0)
    is_evening_rush = (hours_tensor >= 16.0) & (hours_tensor < 19.0)
    is_weekday = (weekends_tensor == 0)
    is_rush_hour_mask = is_weekday & (is_morning_rush | is_evening_rush)
    return is_rush_hour_mask.float()
```

Проверено на семпле из 5 строк: `Is Rush Hour?: [1. 0. 0. 1. 0.]` — совпадает с Expected
Output; `unittests.exercise_1(rush_hour_feature)` — все тесты пройдены.

**1.3 Пайплайн подготовки данных** (**Exercise 2 — `prepare_data`**, GRADED FUNCTION).
Функция принимает сырой DataFrame и возвращает подготовленные `features` и `targets`.
Заполненная часть (`### START CODE HERE ###` … `### END CODE HERE ###`):

```python
full_tensor = torch.tensor(all_values, dtype=torch.float32)

raw_distances = full_tensor[:, 0]
raw_hours = full_tensor[:, 1]
raw_weekends = full_tensor[:, 2]
raw_targets = full_tensor[:, 3]

is_rush_hour_feature = rush_hour_feature(raw_hours, raw_weekends)

distances_col = raw_distances.unsqueeze(1)
hours_col = raw_hours.unsqueeze(1)
weekends_col = raw_weekends.unsqueeze(1)
rush_hour_col = is_rush_hour_feature.unsqueeze(1)
```

Далее (код уже дан в заготовке, не часть упражнения): нормализация (standardization)
`distances_col`/`hours_col` через собственные `mean`/`std`; объединение всех четырёх 2D
признаков через `torch.cat([...], dim=1)` в единый тензор `prepared_features`
(`[N, 4]`: нормализованное расстояние, нормализованный час, `is_weekend` (без
нормализации), `is_rush_hour`); `prepared_targets = raw_targets.unsqueeze(1)`. Функция
также возвращает `results_dict` — словарь промежуточных тензоров для нужд тестов.

Проверено на семпле из 5 строк — вывод (`Shape`, `Values`) совпадает с Expected Output.
`unittests.exercise_2(prepare_data)` — все тесты пройдены. Затем пайплайн запускается на
всём датасете: `features, targets, _ = prepare_data(data_df)`.

**1.4 Визуализация подготовленных данных.**
`helper_utils.plot_rush_hour(data_df, features)` — scatter-график с выделением доставок в
час пик (цвет) против остальных.
`helper_utils.plot_final_data(features, targets)` — scatter-график итоговых данных с
четырьмя категориями (будни/выходные × час пик/не час пик); категория «Weekend (Rush
Hour)» пуста по построению, так как признак `is_rush_hour` определён только для будних
дней.

#### Раздел 2 — Building the Neural Network (**Exercise 3 — `init_model`**, GRADED FUNCTION)

Архитектура — `nn.Sequential` из трёх линейных слоёв с `ReLU` между первыми двумя (кроме
последнего):

```python
torch.manual_seed(41)  # не менять, задано в заготовке

model = nn.Sequential(
    nn.Linear(in_features=4, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=1)
)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()
```

То есть: входной слой 4→64, скрытый слой 64→32, выходной слой 32→1 (один прогноз);
оптимизатор — SGD с `lr=0.01`; функция потерь — MSE. Вывод архитектуры/оптимизатора/
функции потерь совпадает с Expected Output. `unittests.exercise_3(init_model)` — все
тесты пройдены.

#### Раздел 3 — Training the Model (**Exercise 4 — `train_model`**, GRADED FUNCTION)

```python
def train_model(features, targets, epochs, verbose=True):
    losses = []
    model, optimizer, loss_function = init_model()
    for epoch in range(epochs):
        outputs = model(features)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5000 == 0:
            losses.append(loss.item())
            if verbose:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model, losses
```

Порядок шагов внутри эпохи в этой реализации: forward pass → расчёт loss →
`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` (обнуление градиентов
здесь стоит после расчёта loss, но до `backward()`/`step()` — по факту эквивалентно
порядку из лабораторных, так как важно лишь то, что `zero_grad()` предшествует
`backward()`). Каждые 5000 эпох loss сохраняется в список `losses` и опционально
печатается.

Тестовый запуск на 10 000 эпох: `Epoch [5000/10000], Loss: 3.0901`,
`Epoch [10000/10000], Loss: 1.6064` — совпадает с Expected Output (приблизительно).
`unittests.exercise_4(train_model, features, targets)` — все тесты пройдены.

Полное обучение — 30 000 эпох на всём датасете:
`model, loss = train_model(features, targets, 30000)`.

#### Раздел 4 — Evaluating Model Performance

Предсказания для всего датасета получаются под `torch.no_grad()`, затем
`helper_utils.plot_model_predictions(predicted_outputs, targets)` строит scatter-график
«Фактическое значение (X) vs. Предсказанное (Y)» с линией «идеального предсказания» —
чем ближе точки к диагонали, тем точнее модель.

#### Раздел 5 — Making a New Prediction

Задаётся новый сценарий доставки (`distance_miles`, `time_of_day_hours`, `is_weekend`),
оборачивается в тензор `raw_input_tensor`, и вызывается
`helper_utils.prediction(model, data_df, raw_input_tensor, rush_hour_feature)` — эта
функция сама валидирует вход, нормализует его статистиками из `data_df`, инженерит
признак часа пик через переданную `rush_hour_feature`, делает предсказание и печатает
результат в виде форматированной таблицы.

### Важные функции, классы, переменные и структуры данных

**В notebook:**

- `rush_hour_feature(hours_tensor, weekends_tensor)` — GRADED, инженерия бинарного
  признака «час пик в будний день».
- `prepare_data(df)` — GRADED, полный пайплайн подготовки данных: DataFrame → тензоры →
  срезы по столбцам → инженерия признака → reshape в column-векторы → нормализация →
  объединение в `features`/`targets`.
- `init_model()` — GRADED, инициализация модели (3 линейных слоя + 2 `ReLU`), оптимизатора
  (`SGD`, `lr=0.01`) и функции потерь (`MSELoss`); `torch.manual_seed(41)` фиксирован
  внутри функции.
- `train_model(features, targets, epochs, verbose=True)` — GRADED, полный тренировочный
  цикл; возвращает обученную модель и список loss-значений (каждые 5000 эпох).
- `features`/`targets` — итоговые подготовленные тензоры формы `[100, 4]`/`[100, 1]`,
  используемые для обучения на полном датасете.

**В `helper_utils.py`:**

- `plot_delivery_data(df)` — scatter «время доставки vs. расстояние», цвет = время суток
  (`YlOrRd` colormap), стиль маркера = будни/выходной.
- `plot_rush_hour(data_df, features)` — тот же scatter, но группировка по значению
  `features[:, 3]` (час пик / не час пик), с кастомной раскраской и легендой.
- `plot_final_data(features, targets)` — scatter по 4 категориям (будни/выходные × час
  пик/не час пик), с отдельным маркером-стилем на каждую комбинацию.
- `plot_model_predictions(predicted_outputs, actual_targets)` — scatter «факт vs.
  предсказание» + линия идеального совпадения (`y = x`).
- `prediction(model, data_df, raw_inputs, rush_hour_feature_func)` — комплексная функция
  инференса: валидирует сырые входные значения (расстояние > 0 и ≤ 20 миль, время в
  (0, 24], `is_weekend` ∈ {0, 1}, время доставки в рабочих часах (8, 20]), пересчитывает
  статистики нормализации прямо из `data_df` (в реальном приложении их обычно сохраняют
  после обучения, а не пересчитывают — на это указано в комментарии кода), собирает
  итоговый тензор признаков, переводит модель в `model.eval()`, делает предсказание под
  `torch.no_grad()` и печатает результат в виде текстовой таблицы (день недели,
  расстояние, время, признак часа пик, предсказанное время в минутах).

**В `unittests.py`** (проверка через `dlai_grader.grading.test_case`/`print_feedback`):

- `exercise_1` — проверяет тип и форму результата `rush_hour_feature`, а также точные
  ожидаемые значения на отдельном тестовом наборе `[0., 1., 0.]`.
- `exercise_2` — проверяет типы `prepared_features`/`prepared_targets`, `dtype`
  промежуточного `full_tensor` (`float32`), точные значения нарезанных «сырых» столбцов и
  их `*_col` (после `unsqueeze(1)`) версий на подвыборке из 10 строк (`unittests_utils
  .load_rows`, строки 20–29), а также — через анализ исходного кода функции
  (`inspect.getsource` + `remove_comments`) — что `rush_hour_feature` действительно
  вызывается внутри `prepare_data`.
- `exercise_3` — проверяет типы `model`/`optimizer`/`loss_function`, количество и типы
  слоёв модели (`Linear, ReLU, Linear, ReLU, Linear`), размерности `in_features`/
  `out_features` каждого `Linear`-слоя (`[4,64]`, `[64,32]`, `[32,1]`) и `lr=0.01`.
- `exercise_4` — проверяет (через анализ исходного кода), что использованы
  `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`; обучает модель на 15000
  эпох и проверяет, что loss реально меняется между контрольными точками; затем прогоняет
  фиксированный тестовый вход `[[-0.0824, -0.3469, 1.0000, 0.0000]]` и сверяет
  предсказание с ожидаемым значением `21.649423599243164` с допуском `±0.4`.

**В `unittests_utils.py`:**

- `remove_comments(code)` — убирает `#`-комментарии и пустые строки из исходного кода
  функции перед статическим анализом в тестах.
- `load_rows(path_to_csv, row_range=(20, 29))` — загружает CSV и возвращает подвыборку
  строк (`.iloc`) для тестирования `prepare_data` на детерминированном подмножестве
  данных.

### Что демонстрирует каждая значимая часть

- Раздел 1 демонстрирует полный, более реалистичный препроцессинг: чтение табличных
  данных из CSV, инженерию нового признака на основе бизнес-логики (час пик только по
  будням), нормализацию нескольких непрерывных признаков и сборку многофакторного тензора
  признаков.
- Раздел 2 демонстрирует переход от однослойных/двухслойных моделей лабораторных работ к
  трёхслойной сети с двумя скрытыми слоями (64 и 32 нейрона) для более сложной,
  многофакторной задачи.
- Раздел 3 показывает, что структура тренировочного цикла (`zero_grad` → `backward` →
  `step`) остаётся неизменной независимо от сложности модели и данных, меняется лишь
  число эпох (30 000 вместо 500–3000 в лабораторных).
- Раздел 4 показывает стандартный способ визуальной оценки качества регрессионной модели
  — сравнение предсказаний с фактическими значениями относительно линии идеального
  совпадения.
- Раздел 5 и `helper_utils.prediction` демонстрируют, что инференс на новых данных должен
  проходить через тот же самый пайплайн подготовки (валидация входа → инженерия признаков
  → нормализация теми же статистиками) перед вызовом модели — иначе результат будет
  некорректен.

### Видимые результаты и выводы (из notebook outputs)

- `Dataset Shape: (100, 4)`.
- Проверка `rush_hour_feature` на семпле: `Is Rush Hour?: [1. 0. 0. 1. 0.]` — совпадает с
  Expected Output; `unittests.exercise_1` — все тесты пройдены.
- Проверка `prepare_data` на семпле из 5 строк: подготовленные `features`/`targets`
  совпадают с указанным Expected Output; `unittests.exercise_2` — все тесты пройдены.
- Инициализация модели: точная структура `Sequential(...)` (4→64→32→1 с `ReLU` между
  слоями), параметры оптимизатора SGD (`lr: 0.01`, остальные — значения по умолчанию),
  `MSELoss()` — совпадает с Expected Output; `unittests.exercise_3` — все тесты пройдены.
- Тестовое обучение (10 000 эпох): `Epoch [5000/10000], Loss: 3.0901`,
  `Epoch [10000/10000], Loss: 1.6064` — совпадает с Expected Output (приблизительно);
  `unittests.exercise_4` — все тесты пройдены.
- Полное обучение (30 000 эпох) на всём датасете:
  `Epoch [5000/30000], Loss: 3.0901`, `[10000/30000]: 1.6064`, `[15000/30000]: 1.1181`,
  `[20000/30000]: 0.7790`, `[25000/30000]: 0.7204`, `[30000/30000]: 0.3854` — loss
  устойчиво снижается на протяжении всего обучения.
- Оценка модели: markdown-комментарий в notebook описывает график предсказаний как
  плотно прилегающий к линии идеального предсказания («The results look fantastic!»).
- Финальное предсказание для сценария (`distance=20`, `time=14`, `is_weekend=True`):
  таблица с результатом `Estimated Delivery Time: 40.10 minutes`, `Is this considered a
  rush hour period? No` (14:00 не попадает ни в утренний, ни в вечерний час пик).
- Графики (`plot_delivery_data`, `plot_rush_hour`, `plot_final_data`,
  `plot_model_predictions`) присутствуют в outputs как изображения; их словесная
  интерпретация дана в markdown-ячейках рядом с каждым вызовом (см. выше).

### Ограничения, предпосылки и внешние зависимости

- Требуются `pandas`, `torch`, `matplotlib`; для тестов — внутренний пакет
  `dlai_grader.grading` (используется в `unittests.py`, недоступен вне среды курса) и
  локальные `unittests_utils.py`.
- Данные читаются из `./data/data_with_features.csv` — путь относительный, файл должен
  лежать в подпапке `data/` рядом с notebook.
- `torch.manual_seed(41)` зафиксирован внутри `init_model()` и явно помечен в коде как
  «DON'T MANIPULATE IT» — значения loss/предсказаний в конспекте воспроизводимы только
  при этом seed и соответствующей версии `PyTorch`.
- Функция `helper_utils.prediction` в реальном приложении, по собственному комментарию в
  коде, должна была бы загружать статистики нормализации, сохранённые после обучения, а
  не пересчитывать их заново из обучающего `DataFrame` — это упрощение, специфичное для
  учебного задания.
- Бизнес-правила датасета (доставки только 8:00–20:00, не дальше 20 миль) заложены как
  валидация внутри `helper_utils.prediction`, а не как ограничение самой модели.
- Специальных внешних API-ключей или сервисов не требуется — весь код выполняется
  локально.
