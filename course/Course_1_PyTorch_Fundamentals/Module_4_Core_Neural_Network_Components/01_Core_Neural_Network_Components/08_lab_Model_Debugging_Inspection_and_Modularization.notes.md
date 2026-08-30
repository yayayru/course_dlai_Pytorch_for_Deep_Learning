# Lab: Model Debugging, Inspection, and Modularization

Источник: `01_Core_Neural_Network_Components/08_lab_Model_Debugging_Inspection_and_Modularization/C1_M4_Lab_2_debugging.ipynb` (+ `helper_utils.py`).

## Конспект по коду

### Назначение

Notebook проводит через роль "исследователя моделей" (model investigator): начиная со сломанной CNN, с помощью систематических техник отладки находится и исправляется ошибка; далее код рефакторится для ясности и переиспользования (модульность); наконец, разбирается архитектура сложной готовой модели (`SqueezeNet`), чтобы понять её внутреннее устройство.

Задачи лабораторной работы:

- **Debug** сломанной CNN — вставка print-выражений в `forward` для нахождения и исправления несовпадения формы (shape) тензора.
- **Refactor** исправленной модели с помощью `nn.Sequential` для более чистой, модульной и менее подверженной ошибкам архитектуры.
- **Inspect** статистику активаций модели — sanity check на предмет "взрывающихся" или "затухающих" значений.
- **Explore** архитектуру сложной готовой модели (`SqueezeNet`) — подсчёт слоёв и анализ распределения параметров.

### Импорты и данные

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import SqueezeNet
import helper_utils
```

Используется датасет **Fashion MNIST** (серые изображения одежды) — простой бенчмарк, который позволяет сосредоточиться на архитектуре модели, а не на сложной предобработке данных. Датасет загружается через `helper_utils.get_dataset()`, преобразование — `transforms.ToTensor()`. `DataLoader` создаётся с `batch_size = 64`, без перемешивания. Проверка формы батча: `img_batch.shape` должна быть `[64, 1, 28, 28]`.

### Часть 1. Отладка через forward pass

#### Первая версия модели — `SimpleCNN` (сломанная)

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x
```

При прогоне `simple_cnn(img_batch)` внутри `try/except` возникает ошибка:

```
Error during forward pass: mat1 and mat2 shapes cannot be multiplied (28672x14 and 6272x128)
```

Сообщение PyTorch описывает несовпадение форм двух матриц, но не указывает, **где именно** в модели это происходит.

#### Отладочная версия — `SimpleCNNDebug`

Создаётся подкласс `SimpleCNNDebug(SimpleCNN)`, переопределяющий `forward` так, чтобы для каждого слоя печатать:

- форму входа перед слоем;
- форму параметров слоя (`weight`, `bias`);
- форму активации после слоя.

Запуск показывает:

```
Input shape: torch.Size([64, 1, 28, 28])
(Layer components) Conv layer parameters (weights, biases): torch.Size([32, 1, 3, 3]) torch.Size([32])
===
(Activation) After convolution and ReLU: torch.Size([64, 32, 28, 28])
(Activation) After pooling: torch.Size([64, 32, 14, 14])
(Layer components) Linear layer fc1 parameters (weights, biases): torch.Size([128, 6272]) torch.Size([128])
Error during forward pass in debug model: mat1 and mat2 shapes cannot be multiplied (28672x14 and 6272x128)
```

Свёрточный блок работает корректно (batch_size=64, out_channels=32 сохраняются). **Ошибка — в полносвязном блоке**, на первом линейном слое: `x_pool` имеет форму `[64, 32, 14, 14]`, а линейный слой ожидает вход `[64, 2048]` (его матрица весов имеет форму `[128, 6272]`). Поскольку `fc1` ожидает 2D-вход формы `[batch_size, input_features]`, PyTorch неявно "расплющивает" 4D-тензор в 2D способом, дающим форму `[64*32*14, 14]`, — это не то, что нужно, отсюда и ошибка размерности.

#### Исправленная версия — `SimpleCNNFixed`

Фикс — добавить явное flatten перед первым линейным слоем:

```python
x_flattened = torch.flatten(x_pool, start_dim=1)  # Flatten all dimensions except batch
```

После этого модель отрабатывает без ошибок, финальный вывод имеет корректную форму `[64, 10]` (batch_size × число классов Fashion MNIST).

### Часть 2. `nn.Sequential` для модуляризации

Исправленная модель переписывается с использованием `nn.Sequential` — `SimpleCNN2Seq`:

```python
class SimpleCNN2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        flattened_size = 32 * 14 * 14
        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_block(x)
        return x
```

Отмечаемые преимущества такого подхода: **модульность**, **переиспользуемость**, более **чистый код** и **меньше ошибок** (поскольку блоки описаны в одном месте, риск рассинхронизации `__init__`/`forward` снижается).

### Часть 3. Статистическая инспекция инициализации

`SimpleCNN2SeqDebug(SimpleCNN2Seq)` добавляет метод `get_statistics(activation)`, который печатает `mean`, `std`, `min`, `max` активации, и переопределяет `forward` так, чтобы выводить эту статистику после `conv_block` и после `fc_block`.

Пример результатов по первым батчам:

```
=== Batch 0 ===
After conv_block, the activation statistics are:
 Mean: 0.120...  Std: 0.185...  Min: 0.0  Max: 1.289...
After fc_block, the activation statistics are:
 Mean: 0.014...  Std: 0.073...  Min: -0.130...  Max: 0.211...
```

Такая проверка — это sanity check, чтобы убедиться, что модель инициализирована корректно и что активации не "взрываются" и не "затухают" (issues, которые могут приводить к плохому обучению или проблемам со сходимостью).

### Часть 4. Инспекция сложной модели: `SqueezeNet`

Загружается `complex_model = SqueezeNet()` (без предобученных весов). Полный `print(complex_model)` показывает архитектуру из блока `features` (свёрточные слои, `ReLU`, `MaxPool2d` и модули `Fire`) и блока `classifier` (`Dropout`, `Conv2d`, `ReLU`, `AdaptiveAvgPool2d`).

#### Обзор архитектуры без перегрузки деталями

Для больших моделей печать всей архитектуры может быть избыточной. Используются `named_children()`/`children()` для обхода только верхнеуровневых блоков:

```python
for name, block in complex_model.named_children():
    print(f"Block {name} has a total of {len(list(block.children()))} layers:")
    for idx, layer in enumerate(block.children()):
        if len(list(layer.children())) == 0:
            print(f"\t {idx} - Layer {layer}")
        else:
            layer_name = layer._get_name()
            print(f"\t {idx} - Sub-block {layer_name} with {len(list(layer.children()))} layers")
```

Результат: блок `features` содержит **13** слоёв (в том числе 8 суб-блоков `Fire`, каждый с 6 слоями), блок `classifier` — **4** слоя (`Dropout`, `Conv2d`, `ReLU`, `AdaptiveAvgPool2d`).

Чтобы заглянуть внутрь одного модуля `Fire` (`complex_model.features[3]`), используется `modules()`, обходящий все вложенные слои (пропуская сам верхнеуровневый модуль через `if idx > 0`).

#### Детальная инспекция параметров

- Подсчёт количества слоёв конкретного типа (`nn.Conv2d`) через `isinstance` по всем `complex_model.modules()`: в `SqueezeNet` найдено **26** слоёв `Conv2d`.
- Подсчёт общего числа параметров модели: `sum(p.numel() for p in complex_model.parameters())` → **1 248 424** параметра.
- Подсчёт параметров по каждому **терминальному** слою (слою без дочерних модулей) через `complex_model.named_modules()`, с накоплением в словарь `counting_params` и последующей визуализацией через `helper_utils.plot_counting(counting_params)` (столбчатая диаграмма распределения параметров по слоям).

### Вспомогательные функции (`helper_utils.py`)

- `apply_dlai_style()` / глобальный `mpl.rcParams.update(PLOT_STYLE)` — задают единый стиль графиков (шрифты, размеры) и палитру фирменных цветов курса; вызывается на уровне импорта модуля.
- `get_dataset()` — скачивает (при отсутствии) валидационную часть **Fashion MNIST** в `/tmp/dataset` через `torchvision.datasets.FashionMNIST` и возвращает готовый датасет.
- `plot_counting(counting_params)` — строит столбчатую диаграмму (`matplotlib`) числа параметров по именам терминальных слоёв, с поворотом подписей по оси X на 90 градусов.

### Ограничения и зависимости

- Notebook изучен статически (код, markdown-ячейки и сохранённые текстовые outputs); визуальный output диаграммы `plot_counting` в самом notebook помечен как слишком большой для включения и не выполнялся заново.
- Требуется скачивание Fashion MNIST (через `torchvision.datasets.FashionMNIST`) и доступность `torchvision.models.SqueezeNet`.
- Код зависит от локального модуля `helper_utils.py`, лежащего рядом с notebook.
