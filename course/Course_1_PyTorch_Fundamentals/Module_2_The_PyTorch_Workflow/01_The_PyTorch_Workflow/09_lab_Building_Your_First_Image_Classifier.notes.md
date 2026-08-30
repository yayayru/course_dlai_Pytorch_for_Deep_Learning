# Lab: Building Your First Image Classifier

Источник: `01_The_PyTorch_Workflow/09_lab_Building_Your_First_Image_Classifier/C1_M2_Lab_1_mnist_classifier.ipynb` (+ вспомогательный модуль `helper_utils.py`)

## Конспект по коду

### Назначение
Лабораторная работа проводит по всему end-to-end процессу построения, обучения и оценки первого классификатора изображений на PyTorch — классификатора рукописных цифр из датасета **MNIST**. По итогам ноутбука выполняются четыре крупных шага:
- подготовка данных (загрузка MNIST, изучение формата, применение трансформаций);
- построение модели (кастомная нейронная сеть через `nn.Module`);
- обучение модели (loss-функция, оптимизатор, training loop);
- анализ результатов (оценка на невиданных данных, визуализация качества).

### Импорты и зависимости
```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import helper_utils
```
`helper_utils` — локальный модуль с функциями визуализации и не является частью PyTorch/torchvision.

### Выбор устройства
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```
Код выбирает лучшее доступное аппаратное обеспечение: **CUDA** (NVIDIA GPU, обычно самое быстрое обучение), **MPS** (GPU на Apple Silicon) или **CPU** (используется автоматически, если совместимый GPU не обнаружен). В примере запуска был выведен `Using device: CUDA`.

### Подготовка данных (MNIST)
Датасет MNIST — классический бенчмарк классификации изображений, «hello world» компьютерного зрения: 60 000 обучающих и 10 000 тестовых изображений, каждое 28×28 пикселей в оттенках серого, с рукописной цифрой от 0 до 9.

**Шаг 1 — сырые данные.** Сначала MNIST загружается без каких-либо трансформаций:
```python
data_path = "/tmp/data"
train_dataset_without_transform = torchvision.datasets.MNIST(
    root=data_path, train=True, download=True
)
```
Первый элемент датасета (`train_dataset_without_transform[0]`) — кортеж `(image, label)`, где `image` — объект `PIL.Image.Image` размером `(28, 28)`, а `label` — `int` (в примере равен `5`). Отмечается, что метки в PyTorch-датасетах всегда числовые индексы, а не текст (для MNIST индекс совпадает со значением цифры; для других датасетов, например «cat/dog», для перевода индексов в читаемые имена обычно заводят список вроде `class_names = ['cat', 'dog']`).

Функция `helper_utils.display_image(image_pil, label, "MNIST Digit (PIL Image)", show_values=True)` визуализирует необработанное изображение вместе с сеткой числовых значений пикселей (диапазон 0–255).

**Шаг 2 — трансформации.**
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])
```
`ToTensor()` конвертирует `PIL.Image` в `torch.Tensor` и масштабирует пиксели в диапазон [0, 1]; `Normalize()` дополнительно центрирует значения вокруг нуля, используя заранее вычисленные среднее и стандартное отклонение MNIST.

Датасет с трансформацией:
```python
train_dataset = torchvision.datasets.MNIST(
    root=data_path, train=True, download=True, transform=transform
)
```
После трансформации тот же элемент датасета (`train_dataset[0]`) — уже не `PIL.Image`, а `torch.Tensor` формы `(1, 28, 28)` (PyTorch хранит изображения в формате `[channels, height, width]`; так как изображение в оттенках серого, канал один). Значения пикселей больше не в диапазоне 0–255, а центрированы вокруг нуля (в примере — новый диапазон примерно от −0.42 до 2.82). Визуализация трансформированного изображения выполняется тем же `helper_utils.display_image`, но уже на тензоре.

**Шаг 3 — тестовый датасет и DataLoader'ы.**
```python
test_dataset = torchvision.datasets.MNIST(
    root=data_path, train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```
`DataLoader` берёт `Dataset` и отдаёт его батчами (batches), не загружая весь датасет в память сразу. Для обучающего набора используется `batch_size=64` и `shuffle=True` (перемешивание критично для обучения — не даёт модели запомнить порядок данных). Для тестового набора берётся batch size побольше (`1000`), так как градиенты во время оценки не считаются, и `shuffle=False`, так как порядок при финальном измерении качества не важен.

### Модель: `SimpleMNISTDNN`
```python
class SimpleMNISTDNN(nn.Module):
    def __init__(self):
        super(SimpleMNISTDNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
```
Класс наследуется от `nn.Module` (даёт полный контроль над структурой модели). Метод `__init__` определяет и инициализирует слои модели; `forward` определяет путь, по которому данные проходят через эти слои.

- `nn.Flatten()` превращает изображение 28×28 в одномерный вектор из 784 элементов (`28 * 28 = 784`) — это требуется, потому что линейные слои ожидают плоские векторы, а не сетки.
- Первый `nn.Linear(784, 128)` отображает 784 входных пикселя в 128 скрытых признаков.
- `nn.ReLU()` — функция активации, вносящая нелинейность (без неё модель могла бы выучивать только линейные зависимости).
- Второй `nn.Linear(128, 10)` отображает 128 признаков в 10 выходных классов (по одному на каждую цифру 0–9).

### Инициализация модели, loss-функции и оптимизатора
```python
model = SimpleMNISTDNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- `nn.CrossEntropyLoss` — стандартный выбор для многоклассовой (multi-class) классификации вроде MNIST: она специально измеряет ошибку, когда модель должна выбрать один класс из нескольких (одну цифру из 0–9).
- `optim.Adam` — популярный и эффективный оптимизатор, который адаптирует learning rate по ходу обучения. Перенос модели и данных на устройство (`device`) в этом ноутбуке выполняется внутри самой функции обучения — это гарантирует, что и модель, и данные корректно размещены непосредственно перед вычислениями.

### Функция обучения на одну эпоху — `train_epoch`
```python
def train_epoch(model, loss_function, optimizer, train_loader, device):
    model = model.to(device)
    model.train()

    epoch_loss = 0.0
    running_loss = 0.0
    num_correct_predictions = 0
    total_predictions = 0
    total_batches = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value
        running_loss += loss_value

        _, predicted_indices = outputs.max(1)
        batch_size = targets.size(0)
        total_predictions += batch_size
        num_correct_in_batch = predicted_indices.eq(targets).sum().item()
        num_correct_predictions += num_correct_in_batch

        if (batch_idx + 1) % 134 == 0 or (batch_idx + 1) == total_batches:
            avg_running_loss = running_loss / 134
            accuracy = 100. * num_correct_predictions / total_predictions
            print(f'\tStep {batch_idx + 1}/{total_batches} - Loss: {avg_running_loss:.3f} | Acc: {accuracy:.2f}%')
            running_loss = 0.0
            num_correct_predictions = 0
            total_predictions = 0

    avg_epoch_loss = epoch_loss / total_batches
    return model, avg_epoch_loss
```
Функция реализует один полный проход по датасету (одну эпоху, epoch); одна итерация по одному батчу внутри эпохи называется шагом (step). Ключевые части:
- **подготовка модели**: перенос на `device` и `model.train()`;
- **основной цикл обучения**: для каждого батча выполняется стандартная пятишаговая последовательность — очистка градиентов (`optimizer.zero_grad()`), forward pass, вычисление loss, backpropagation (`loss.backward()`), обновление весов (`optimizer.step()`);
- **отчёт о прогрессе**: отслеживаются running loss и accuracy, периодически печатаются обновления. При 60 000 обучающих изображений и batch size 64 прогресс печатается каждые 134 шага, что даёт 7 обновлений за эпоху.

Функция возвращает обученную модель и среднее значение loss за всю эпоху (`avg_epoch_loss`).

### Функция оценки — `evaluate`
```python
def evaluate(model, test_loader, device):
    model.eval()
    num_correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted_indices = outputs.max(1)
            batch_size = targets.size(0)
            total_predictions = total_predictions + batch_size
            correct_predictions = predicted_indices.eq(targets)
            num_correct_in_batch = correct_predictions.sum().item()
            num_correct_predictions = num_correct_predictions + num_correct_in_batch

    accuracy_percentage = (num_correct_predictions / total_predictions) * 100
    print((f'\tAccuracy - {accuracy_percentage:.2f}%'))
    return accuracy_percentage
```
Похожа на training loop, но оптимизирована под inference (вывод предсказаний без обучения):
- **настройка для inference**: `model.eval()` и блок `torch.no_grad()` — необходимы для корректных результатов и делают процесс быстрее и более экономным по памяти;
- **упрощённый forward pass**: только вычисление предсказаний и accuracy, без loss, backpropagation и обновления весов оптимизатором.

### Training loop
```python
num_epochs = 5
train_loss = []
test_acc = []

for epoch in range(num_epochs):
    print(f'\n[Training] Epoch {epoch+1}:')
    trained_model, loss = train_epoch(model, loss_function, optimizer, train_loader, device)
    train_loss.append(loss)

    print(f'[Testing] Epoch {epoch+1}:')
    accuracy = evaluate(trained_model, test_loader, device)
    test_acc.append(accuracy)
```
Цикл выполняется заданное число эпох (`num_epochs = 5`, но допускается изменить). На каждой эпохе:
1. `train_epoch` обучает модель на всех обучающих данных.
2. `evaluate` сразу после этого измеряет качество модели на невиданном тестовом наборе — это важный шаг для проверки, действительно ли модель учится **обобщать (generalize)**, или же просто запоминает обучающий набор.

Loss и accuracy каждой эпохи сохраняются в списки `train_loss` и `test_acc` для последующего анализа.

**Видимые результаты (output ноутбука):** за 5 эпох loss стабильно снижался (например, с ~0.539 в начале первой эпохи до ~0.046 к концу пятой), а test accuracy росла от 95.68% (эпоха 1) до 97.65% (эпоха 5).

### Анализ производительности модели
- `helper_utils.display_predictions(trained_model, test_loader, device)` — визуализирует предсказания модели на случайной выборке тестовых изображений (по одному примеру на каждый из 10 классов), показывая, где модель угадывает верно, а где ошибается.
- `helper_utils.plot_metrics(train_loss, test_acc)` — строит два графика рядом: training loss по эпохам (ожидается устойчивое снижение) и test accuracy по эпохам (ожидается рост; выход кривой на плато обычно означает, что модель выучила максимум из текущей архитектуры/данных).

### Вспомогательный модуль `helper_utils.py`
Локальный файл, импортируемый в ноутбуке, содержит три функции:
- **`display_image(image, label, title, num_ticks=6, show_values=True)`** — отображает изображение (поддерживает как `PIL.Image`, так и `torch.Tensor`) вместе с заголовком, цветовой шкалой (color bar) и, опционально, наложенными числовыми значениями каждого пикселя. Определяет диапазон значений (`vmin`/`vmax`) автоматически в зависимости от типа входных данных (0–255 для `PIL.Image`, min/max тензора — для `torch.Tensor`).
- **`display_predictions(model, test_loader, device)`** — переводит модель в режим `eval()`, находит по одному случайному образцу для каждого из 10 классов цифр в тестовом датасете, прогоняет их через модель внутри `torch.no_grad()` и отображает сетку 2×5 изображений с истинной и предсказанной меткой (заголовок зелёный при совпадении, красный при ошибке).
- **`plot_metrics(train_loss, test_acc)`** — строит два линейных графика (loss и accuracy) по эпохам с помощью `matplotlib`.

### Ограничения и предпосылки
- Notebook рассчитан на запуск с доступом к интернету при первом прогоне (для скачивания MNIST через `torchvision.datasets.MNIST(..., download=True)`); данные сохраняются локально в `/tmp/data`.
- Использование GPU (CUDA или MPS) не обязательно, но ускоряет обучение; при их отсутствии автоматически используется CPU.
- Число эпох (`num_epochs`) задаётся вручную и в примере равно 5; в ноутбуке отмечено, что его можно изменить.
