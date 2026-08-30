# DataLoader

## Источники

- Транскрипция: `04_video_DataLoader.trans.txt`
- Презентация модуля: `PyTorch_C1_M3.pdf` (раздел "DataLoader", слайды 99–145) — использована для сверки кода, чисел и вывода.

## Контекст

В прошлом видео данные были очищены с помощью трансформаций: изменение размера изображений, конвертация в тензоры, нормализация значений. Теперь настало время разобраться с двумя ещё более важными вещами: во-первых, разбить датасет на обучающую, валидационную и тестовую выборки (training, validation, test); во-вторых, использовать `DataLoader` для эффективной батчевой подачи (batch and serve) этих данных. Но важнее самого "как" — понять, **почему** эти шаги имеют значение.

## Зачем разбивать датасет

Если больше данных — это хорошо, почему бы не обучать модель на полном наборе изображений цветов? Проблема в том, что тогда невозможно узнать, работает ли модель на новых фотографиях, которых она не видела. Именно поэтому датасет разбивают на части.

Датасеты обычно делят на три части, у каждой — своё назначение:

- **Training set (обучающая выборка)** — то, на чём модель учится. Она видит эти изображения снова и снова во время обучения.
- **Validation set (валидационная выборка)** — помогает проверять производительность модели во время обучения, пока идёт настройка (tuning).
- **Test set (тестовая выборка)** — финальная проверка, используется **только один раз**, после завершения обучения.

Для Oxford Flowers (8189 изображений) пример разбиения:

- ~5700 изображений — training (**70%**)
- ~1200 изображений — validation (**15%**)
- ~1200 изображений — test (**15%**)

В курсе далее будет использоваться комбинация этих выборок в зависимости от основной цели конкретной лекции. Подробнее о том, как работают такие разбиения и почему они важны, — в дополнительных материалах курса (resources) или в специализации по data engineering.

## Разбиение датасета в PyTorch: `random_split`

```python
from torch.utils.data import random_split

# Split into train/val/test: 70/15/15
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"Training: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")
```

```
Training: 5732 images
Validation: 1228 images
Test: 1229 images
```

- Для обучающей выборки берётся 70% от общей длины датасета, для валидации — 15%.
- Размер тестовой выборки вычисляется как **остаток** (`len(dataset) - train_size - val_size`) — это гарантирует, что суммы всех частей точно равны полному датасету, и позволяет избежать ошибок округления.
- Ключевая функция — `random_split`. Она случайным образом распределяет изображения по каждой выборке, чтобы не получилось так, что все ромашки попали в одну выборку, а все розы — в другую. Это обеспечивает каждой выборке хорошую смесь всех 102 типов цветов.

**Важное уточнение**: исходный датасет (`dataset`) **не изменяется**. Данные не "нарезаются" физически — создаются лишь три отдельных **представления (views)** одних и тех же данных.

```python
# random_split creates view not copies
print(f"Original dataset still has: {len(dataset)} images")
```

```
Original dataset still has: 8189 images
```

## DataLoader: эффективная загрузка батчей

Далее `DataLoader` используется для эффективной загрузки батчей из каждой выборки. Это особенно важно во время обучения — для производительности.

Batch size (размер батча) равный 32 означает, что за раз будет получено 32 образца вместо одного.

Проход по `DataLoader` в цикле даёт **один батч за итерацию**:

```python
# This loop will go through ALL your data, one batch at a time
for images, labels in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
    break  # Stop after first batch
```

```
Batch of images shape: torch.Size([32, 3, 224, 224])
Batch of labels shape: torch.Size([32])
```

`break` после первого батча — удобный способ проверить содержимое первого батча. Каждый батч даёт **две вещи**: батч изображений и батч меток. Это означает 32 изображения, у каждого по 3 цветовых канала, размер каждого — 224×224. Тензор меток даёт по одной метке на каждое изображение в этом батче.

Альтернативный способ получить только первый батч без полного цикла — удобен для быстрой отладки:

```python
# Get just the first batch to inspect
images, labels = next(iter(train_loader))
```

## Параметр `shuffle`

К `DataLoader` нужно добавить один важный параметр — `shuffle`.

```python
# For training - shuffle=True
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# For validation and test - shuffle=False
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Зачем перемешивать обучающие данные — две причины:

1. Если датасет упорядочен (сначала все ромашки, потом розы, потом подсолнухи), модель может научиться ассоциировать **позицию** образца с типом цветка, вместо того чтобы изучать реальные признаки — а это бесполезно для реальных предсказаний.
2. Если модель видит только ромашки на протяжении многих батчей подряд, а затем только розы, она может фактически "забыть" то, что выучила про ромашки. Перемешивание смешивает типы цветов в каждом батче и помогает модели сохранять всё, чему она учится.

### Зачем `shuffle=False` для validation и test

Модель **не обучается** на этих выборках — она только оценивается, поэтому перемешивание не нужно, и проблемы, описанные выше, здесь неприменимы.

Важное уточнение: `shuffle` влияет только на то, как `DataLoader` подаёт батчи. Исходный датасет остаётся неизменным.

## Батчи и эпохи (batches and epochs)

Расчёт для обучающей выборки из 5,732 изображений при batch size = 32:

```
5,732 ÷ 32 = 179.125
```

- Не может быть 0.125 батча, поэтому получается **179 полных батчей** по 32 изображения и **1 неполный батч** с 4 изображениями (остаток).
- Итого: **180 батчей**.
- **Одна эпоха (epoch)** означает прохождение по всем 180 батчам один раз — т.е. каждое изображение датасета встречается ровно один раз.

Проверка на коде — вывод длины последних трёх батчей:

```python
batch_count = 0
total_images = 0

for images, labels in train_loader:
    batch_count += 1
    total_images += len(images)

    # Show the last few batches
    if batch_count >= 178:
        print(f"Batch {batch_count}: {len(images)} images")

print(f"\nTotal batches in one epoch: {batch_count}")
print(f"Total images seen: {total_images}")
```

```
Batch 178: 32 images
Batch 179: 32 images
Batch 180: 4 images

Total batches in one epoch: 180
Total images seen: 5732
```

Последний батч содержит всего 4 изображения — это нормально, это просто оставшиеся изображения, которые не составили полный батч. Это иногда может сбивать с толку тех, кто ожидает, что каждый батч будет одинакового размера.

При обучении в течение **10 эпох** модель проходит через все 180 батчей 10 раз. Модель видит каждое изображение 10 раз в сумме, но каждую эпоху — в разном случайном порядке благодаря `shuffle=True`.

## Две распространённые ошибки

### Ошибка 1: загрузка данных внутри `__getitem__`

```python
# Don't load your data in __getitem__!
class BadDataset(Dataset):
    def __getitem__(self, idx):
        data = pd.read_csv('huge_file.csv')
        return data.iloc[idx]
```

Здесь весь CSV-файл с диска загружается в память **каждый раз**, когда вызывается `__getitem__`. Если в датасете 5,732 записи, это значит, что полный CSV перезагружается 5,732 раза за одну эпоху. При обучении на 10 эпох это более **57,000 полных загрузок** одного и того же файла — огромное и абсолютно ненужное замедление.

Исправление простое — загрузка один раз, в `__init__`:

```python
# RIGHT - load once, access many times
class GoodDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('huge_file.csv')  # Load once

    def __getitem__(self, idx):
        return self.data.iloc[idx]  # Just access
```

### Ошибка 2: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

Если возникает такая ошибка, первое, что нужно попробовать — **уменьшить batch size**: начать с 32 или даже 16, и постепенно увеличивать его, подбирая размер под доступную память.

## Полная сборка пайплайна

```python
# Complete setup for the botanical garden app
from torch.utils.data import DataLoader, random_split

# Our dataset with transforms from the previous video
dataset = OxfordFlowersDataset(r'...\Oxford 102 flowers', transform=transform)

# Split into train/val/test: 70/15/15
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create DataLoaders with appropriate settings
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verify everything works
print(f"Train: {len(train_loader)} batches")
print(f"Val: {len(val_loader)} batches")
print(f"Test: {len(test_loader)} batches")

# Quick test - get one batch from each
for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
    images, labels = next(iter(loader))
    print(f"{name} batch: {images.shape}")
```

```
Train: 180 batches
Val: 39 batches
Test: 39 batches
Train batch: torch.Size([32, 3, 224, 224])
Val batch: torch.Size([32, 3, 224, 224])
Test batch: torch.Size([32, 3, 224, 224])
```

Вывод показывает, что всё работает корректно: правильные формы, правильные разбиения и эффективная загрузка. Теперь есть полный конвейер данных (complete data pipeline) для приложения ботанического сада — от "грязных" `.mat`-файлов до эффективной батчевой загрузки.

## Далее

В следующем видео — краткий обзор того, как сделать этот конвейер более устойчивым к ошибкам (bug proof).
