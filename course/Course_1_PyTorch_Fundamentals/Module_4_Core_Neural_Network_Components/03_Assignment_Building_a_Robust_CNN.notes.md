# Assignment: Overcoming Overfitting — Building a Robust CNN

Источник: `03_Assignment_Building_a_Robust_CNN/C1M4_Assignment.ipynb` (+ `helper_utils.py`, `unittests.py`, `c2_preview/c2_preview.py`).

## Конспект по коду

### Назначение

Это финальное graded-задание курса. Отправная точка — модель из лабораторной работы `07_lab`, показавшая явные признаки **overfitting** на полном наборе из 15 классов. Задача — не просто подправить один параметр, а систематически переработать весь ML-пайплайн профессиональными инструментами и техниками:

- **усилить пайплайн данных** более мощной аугментацией, чтобы получить более богатый обучающий набор;
- **сделать архитектуру модульной** — вынести повторяющийся паттерн в переиспользуемый `CNNBlock`;
- **добавить продвинутые слои**, в частности **Batch Normalization**, для стабилизации обучения и улучшения обобщения;
- **применить надёжную регуляризацию** — **Dropout** и **weight decay** — чтобы напрямую бороться с overfitting.

Задание построено как последовательность **graded**-упражнений (`### START CODE HERE ### ... ### END CODE HERE ###`), после каждого из которых идёт ячейка проверки (`unittests.exercise_N(...)`).

### Импорты

```python
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import helper_utils
import unittests
```

`device` выбирается автоматически (`cuda`, если доступна).

### 1. Апгрейд пайплайна данных

`cifar100_mean = (0.5071, 0.4867, 0.4408)`, `cifar100_std = (0.2675, 0.2565, 0.2761)` — те же значения нормализации CIFAR-100, что и в лабораторной работе.

#### Exercise 1 — `define_transformations`

Задача: реализовать функцию `define_transformations(mean, std)`, возвращающую пару `(train_transformations, val_transformations)`.

Решение:

```python
def define_transformations(mean, std):
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transformations, val_transformations
```

По сравнению с пайплайном из лабораторной работы 07 добавлена новая техника аугментации — `RandomVerticalFlip()`. Идея: вертикальные отражения помогают модели понять, что ориентация объекта не всегда "вертикальна как надо" — это полезно, например, для насекомых или цветов, снятых под разными углами.

`train_transform, val_transform = define_transformations(cifar100_mean, cifar100_std)`.

#### 1.2 — Data Loaders

Полный список из 15 классов (`all_target_classes`) — тот же набор flowers/mammals/insects, что и в лабораторной работе 07. Загрузка через `helper_utils.load_cifar100_subset(all_target_classes, train_transform, val_transform, "/tmp/cifar_100")` → `train_dataset`, `val_dataset`. Далее — `DataLoader` с `batch_size = 64` (`train_loader` с перемешиванием, `val_loader` без).

#### 1.3 — Визуализация

`helper_utils.visualise_images(train_loader, grid=(3, 5))` — в отличие от лабораторной работы 07, здесь функция принимает `DataLoader`, а не сам `Dataset` (сигнатура `visualise_images` в этой версии `helper_utils.py` изменена соответствующим образом).

### 2. Модульная и надёжная CNN

#### 2.1 — `BatchNorm2d` и Exercise 2 — `CNNBlock`

Перед упражнением даётся развёрнутое объяснение `nn.BatchNorm2d` как "регулировщика трафика" для данных между слоями: он нормализует активации внутри каждого мини-батча (приводя их к согласованному среднему и стандартному отклонению), а затем с помощью двух обучаемых параметров масштабирует и сдвигает результат. Отмечены три эффекта:

- **стабилизирует и ускоряет обучение** (позволяет использовать более высокий learning rate);
- **действует как регуляризатор** (небольшой шум от статистик мини-батча мешает точному запоминанию обучающих данных);
- **снижает чувствительность к инициализации весов**.

Задача: реализовать класс `CNNBlock`, упаковывающий `Conv2d → BatchNorm2d → ReLU → MaxPool2d` в единый `nn.Sequential`.

Решение:

```python
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)
```

Проверка (`CNNBlock(in_channels=3, out_channels=16)` на входе `torch.randn(1, 3, 32, 32)`) даёт выход формы `[1, 16, 16, 16]` — pooling с `stride=2` уменьшает пространственный размер вдвое.

#### 2.2 — Exercise 3 — `SimpleCNN`

Модель собирается из трёх `CNNBlock` (feature extractor) и `classifier` из полносвязных слоёв. Dropout rate увеличен до **0.6** (в лабораторной работе 07 было 0.5) — ещё один шаг в борьбе с overfitting.

Решение:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = CNNBlock(in_channels=3, out_channels=32)
        self.conv_block2 = CNNBlock(in_channels=32, out_channels=64)
        self.conv_block3 = CNNBlock(in_channels=64, out_channels=128)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128 * 4 * 4, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x
```

Число входов первого линейного слоя (`128 * 4 * 4`) объясняется так же, как в лабораторной работе: изображения 32×32, три `MaxPool2d(stride=2)` внутри `CNNBlock` уменьшают размер `32 → 16 → 8 → 4`, а последний блок выдаёт 128 каналов.

Проверка на `torch.randn(64, 3, 32, 32)` даёт выход `[64, 15]`.

Далее создаётся рабочий экземпляр: `num_classes = len(train_dataset.classes)`, `model = SimpleCNN(num_classes)`.

### 3. Обучение обновлённой модели

#### 3.1 — Loss и оптимизатор

```python
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
```

По сравнению с лабораторной работой 07 (`lr=0.001`, без `weight_decay`) здесь ниже learning rate и добавлен `weight_decay=0.0005`. Weight decay штрафует большие веса в функции потерь, побуждая сеть учить более простые значения весов — ещё один инструмент против запоминания обучающих данных.

#### Exercise 4 — `train_epoch`

Задача: реализовать пять шагов одной обучающей итерации внутри цикла по батчам.

```python
def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss
```

Проверка через `helper_utils.verify_training_process(...)` обучает свежий `SimpleCNN(15)` на подвыборке из 10 батчей в течение 5 эпох и проверяет, что (1) веса слоя `conv_block1.block[0]` изменились и (2) loss уменьшился от эпохи к эпохе.

#### Exercise 5 — `validate_epoch`

Задача: реализовать логику валидации — forward pass без вычисления градиентов, подсчёт loss и accuracy.

```python
def validate_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            val_loss = loss_function(outputs, labels)
            running_val_loss += val_loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total
    return epoch_val_loss, epoch_accuracy
```

Проверка через `helper_utils.verify_validation_process(...)`: убеждается, что функция возвращает `float` для loss и accuracy, и что веса модели **не** меняются во время валидации (сравнение весов `conv_block1.block[0]` до/после).

### Полный `training_loop` и запуск обучения

После прохождения graded-упражнений собирается `training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device)`, которая на каждой эпохе вызывает `train_epoch` и `validate_epoch`, отслеживает **лучшую** по `val_accuracy` эпоху, сохраняет `copy.deepcopy(model.state_dict())` для неё и в конце возвращает модель именно с этими "лучшими" весами (а не с весами последней эпохи) — защита от ситуации, когда качество модели со временем ухудшается при слишком долгом обучении.

Запуск: `trained_model, training_metrics = training_loop(model=model, train_loader=train_loader, val_loader=val_loader, loss_function=loss_function, optimizer=optimizer, num_epochs=50, device=device)`, визуализация через `helper_utils.plot_training_metrics(training_metrics)`.

**Анализ результатов** (по тексту markdown-ячейки после обучения): кривые training и validation loss теперь идут близко друг к другу, широкий разрыв, сигнализировавший об overfitting, исчез; validation accuracy показывает более здоровый, устойчивый рост. Совокупность более сильной аугментации, Batch Normalization и weight decay позволила модели лучше обобщать. При этом accuracy на валидации выходит на плато примерно на уровне **~70%** — в тексте объясняется, что это ожидаемый предел для набора техник, изученных в рамках этого курса ("foundational toolkit"), и это не финал возможностей модели, а демонстрация эффективного применения освоенных инструментов.

### 4. Заглядывая вперёд: `c2_preview`

В финальной части задания импортируется `course_2_preview` из `c2_preview/c2_preview.py` и запускается на тех же `train_dataset`/`val_dataset` с `num_epochs=5`. Она демонстрирует, что при использовании техник следующего курса (предобученная модель, dynamic learning rate scheduling, более продвинутые трансформации) validation accuracy превышает **80%** всего за 5 эпох — заметно быстрее и выше, чем у модели текущего задания за 50 эпох. Подробный разбор кода `course_2_preview` — в отдельном файле `c2_preview/c2_preview.notes.md`.

Перечислены три ключевых апгрейда, лежащих в основе этого результата:

- **Использование предобученной модели** — вместо старта со случайных весов используется модель, уже обученная на миллионах изображений, с последующей дообучением (fine-tuning) под конкретную задачу.
- **Динамическое планирование learning rate (learning rate scheduler)** — вместо фиксированного значения learning rate динамически подстраивается по ходу обучения: крупные шаги в начале, более точные — по мере приближения к оптимуму.
- **Более продвинутые трансформации** — аугментация, адаптированная специально под высокопроизводительные модели.

### Ограничения и зависимости

- Notebook изучен статически (по коду ячеек, markdown-описаниям заданий и приведённым в них ожидаемым outputs); сам notebook не запускался.
- Требуется скачивание CIFAR-100 (через `helper_utils.load_cifar100_subset`, локальный путь `/tmp/cifar_100`) и доступность GPU/CPU через `device`.
- Код зависит от локальных модулей `helper_utils.py` и `unittests.py`, а для последнего раздела — от пакета `c2_preview/` (см. `c2_preview/c2_preview.notes.md`), который использует `torchvision.models.mobilenet_v3_small` с предобученными весами (`weights='DEFAULT'`) и требует доступа к интернету для их загрузки.
- Отдельные проверочные ячейки (`unittests.exercise_1..5`) требуют, чтобы соответствующие graded-функции/классы были определены в глобальной области видимости notebook.
