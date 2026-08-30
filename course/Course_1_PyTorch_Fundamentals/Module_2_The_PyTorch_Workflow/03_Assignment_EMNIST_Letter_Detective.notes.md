# Programming Assignment: EMNIST Letter Detective

Источник: `03_Assignment_EMNIST_Letter_Detective/C1M2_Assignment.ipynb` (+ вспомогательные файлы `helper_utils.py`, `unittests.py`, `unittests_utils.py`)

## Конспект по коду

### Назначение
Assignment расширяет лабораторную работу по MNIST на более сложную задачу: классификацию рукописных **букв** из датасета **EMNIST** (подмножество `letters`, 26 классов вместо 10 цифр, данные «более грязные» (messier) и более разнообразные). В конце задания обученная модель используется, чтобы декодировать рукописное секретное сообщение от Эндрю Ына (Andrew Ng).

Что предстоит сделать по ходу ноутбука:
- загрузить и изучить датасет EMNIST Letters;
- предобработать изображения (исправить ориентацию, нормализовать значения пикселей, преобразовать в тензоры);
- построить многослойную нейронную сеть для классификации букв;
- обучить и оценить модель на невиданных примерах;
- использовать обученную модель для декодирования секретного сообщения.

Ноутбук выполнен в формате graded assignment: часть ячеек — это заготовки функций с маркерами `### START CODE HERE ###` / `### END CODE HERE ###`, которые нужно заполнить самостоятельно; в файле, который был изучен, эти блоки уже заполнены эталонным решением. После каждой ячейки-упражнения идёт вызов `unittests.exercise_N(...)`, проверяющий корректность реализации.

### Импорты и устройство
```python
import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torchvision.transforms.functional as F

import helper_utils
import unittests

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
`helper_utils` и `unittests` — локальные модули: первый содержит вспомогательные функции для визуализации, оценки по классам и сохранения модели; второй — тесты для проверки решений.

### 1. Letter Images — загрузка и подготовка данных

**1.1 Загрузка датасета.** Используется `torchvision.datasets.EMNIST` с `split='letters'` (буквы `a`–`z`, 26 классов):
```python
data_path = '/tmp/EMNIST_data'
train_dataset = datasets.EMNIST(root=data_path, split='letters', train=True, download=download)
test_dataset = datasets.EMNIST(root=data_path, split='letters', train=False, download=download)
```
Подмножество `letters` включает **124 800** обучающих изображений и **20 800** тестовых, каждое размечено одним из 26 классов букв.

**1.2 Изучение сырых данных.** Через `helper_utils.visualize_image(img, label)` отображается пример изображения (например, по индексу `90000`). Отмечается, что рукописные буквы в EMNIST хранятся повёрнутыми/отражёнными — это не мешает обучению, пока все изображения ориентированы одинаково (модель не «знает», как должна выглядеть правильно ориентированная буква, она лишь учит паттерны, согласованные по всему датасету), но затрудняет визуальный осмотр человеком.

**1.3 Предобработка изображений.**
```python
mean = (0.1736,)
std = (0.3317,)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset.transform = transform
test_dataset.transform = transform
```
`mean`/`std` — заранее вычисленные значения для EMNIST Letters. `ToTensor()` переводит изображение в тензор и масштабирует пиксели в диапазон [0, 1]; `Normalize` затем вычитает среднее и делит на стандартное отклонение, центрируя значения вокруг нуля. Трансформации применяются автоматически при каждой загрузке образца из датасета.

Для визуализации в правильной, привычной человеку ориентации используется вспомогательная функция:
```python
def correct_image_orientation(image):
    rotated = F.rotate(image, 90)   # поворот на 90 градусов по часовой стрелке
    flipped = F.vflip(rotated)      # отражение по вертикали
    return flipped
```
Важно: это исправление ориентации применяется **только для визуализации** — сами данные, на которых обучается модель, остаются без изменений.

**1.4 Загрузка данных батчами — Exercise 1: `create_emnist_dataloaders`.**
```python
def create_emnist_dataloaders(train_dataset, test_dataset, batch_size=64):
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
```
Обучающий `DataLoader` создаётся с `shuffle=True` (перемешивание при каждой эпохе), тестовый — с `shuffle=False`. Ожидаемый результат при `batch_size=64`: 1950 батчей на 124 800 обучающих изображений и 325 батчей на 20 800 тестовых, форма батча данных — `[64, 1, 28, 28]`, форма меток — `[64]`.

### 2. Building the Neural Network — Exercise 2: `initialize_emnist_model`
```python
def initialize_emnist_model(num_classes=26):
    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_function, optimizer
```
Условия задания к архитектуре модели:
- первый слой обязательно `nn.Flatten()`;
- промежуточные слои — только `nn.Linear` и `nn.ReLU`, не более 5 слоёв в сумме;
- размер скрытого слоя (hidden unit size) — не более 256;
- последний слой обязательно `nn.Linear`, отображающий признаки в `num_classes` выходов;
- общее число слоёв (включая первый и последний) — не более 7.

Loss-функция — `nn.CrossEntropyLoss()`, оптимизатор — `optim.Adam` с `lr=0.001`. В решении, приведённом в ноутбуке, выбрана архитектура `Flatten → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 26)`.

### 3. Training and Evaluation of the Model

**3.1 Exercise 3 — `train_epoch`.**
```python
def train_epoch(model, loss_function, optimizer, train_loader, device, verbose=True):
    model.to(device)
    model.train()
    running_loss = 0.0
    num_correct_predictions = 0
    total_predictions = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets - 1  # метки EMNIST letters 1-индексированы -> сдвиг к 0-индексации

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss = loss_value

        predicted_indices = outputs.argmax(dim=1)
        correct_predictions = predicted_indices.eq(targets)
        num_correct_in_batch = correct_predictions.sum().item()
        num_correct_predictions = num_correct_in_batch

        batch_size = targets.size(0)
        total_predictions += batch_size

    average_loss = running_loss / len(train_loader)
    accuracy_percentage = (num_correct_predictions / total_predictions) * 100

    if verbose:
        print(f"Epoch Loss (Avg): {average_loss:.3f} | Epoch Acc: {accuracy_percentage:.2f}%")

    return model, average_loss
```
Важная деталь, отмеченная в docstring: метки в `train_loader` для EMNIST letters **1-индексированы** (от 1 до 26), поэтому перед вычислением loss из targets вычитается 1, чтобы привести их к 0-индексации (0–25).

Внутри цикла по батчам выполняется стандартная последовательность: `optimizer.zero_grad()` → `outputs = model(inputs)` → `loss = loss_function(outputs, targets)` → `loss.backward()` → `optimizer.step()`. Для метрик используются `outputs.argmax(dim=1)` (индексы предсказанных классов) и `predicted_indices.eq(targets)` (сравнение с истинными метками).

В приведённом в ноутбуке коде `running_loss` и `num_correct_predictions` внутри цикла **присваиваются** (`=`), а не накапливаются (`+=`), то есть после цикла эти переменные хранят значения только от последнего обработанного батча, а не сумму по всем батчам; `total_predictions`, напротив, накапливается через `+=` по всем батчам. Ожидаемый результат после одной эпохи обучения (согласно ноутбуку): `Epoch Loss (Avg): 0.602 | Epoch Acc: 81.40%`.

**3.2 Exercise 4 — `evaluate`.**
```python
def evaluate(model, test_loader, device, verbose=True):
    model.eval()
    num_correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets - 1

            outputs = model(inputs)
            predicted_indices = outputs.argmax(dim=1)
            correct_predictions = predicted_indices.eq(targets)
            num_correct_in_batch = correct_predictions.sum().item()
            num_correct_predictions = num_correct_in_batch

            batch_size = targets.size(0)
            total_predictions += batch_size

        accuracy_percentage = (num_correct_predictions / total_predictions) * 100

    if verbose:
        print(f'Test Accuracy: {accuracy_percentage:.2f}%')

    return accuracy_percentage
```
Аналогично `train_epoch`, метки сдвигаются на −1 для приведения к 0-индексации. Градиенты отключены через `with torch.no_grad()`. Как и в `train_epoch`, `num_correct_predictions` внутри цикла присваивается (`=`), а не накапливается. Ожидаемый результат: `Test Accuracy: 87.59%`.

**3.3 Exercise 5 — `train_and_evaluate`.**
```python
def train_and_evaluate(model, train_loader, test_loader, num_epochs, loss_function, optimizer, device):
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch+1}")

        trained_model, _ = train_epoch(
            model=model, loss_function=loss_function, optimizer=optimizer,
            train_loader=train_loader, device=device
        )
        accuracy = evaluate(model=trained_model, test_loader=test_loader, device=device)

    return trained_model
```
Объединяет `train_epoch` и `evaluate` в общий цикл на заданное число эпох (`num_epochs`, в ноутбуке задано `num_epochs = 10`, с ограничением на редактируемую ячейку не более 15). Ожидаемый результат за 10 эпох (из ноутбука): loss снижается с 0.600 до 0.151, epoch accuracy растёт с 81.48% до 94.40%, test accuracy колеблется в районе 87–91% (от 87.25% на 1-й эпохе до 90.57% на 10-й).

**Оценка по классам.** Функция `helper_utils.evaluate_per_class(trained_model, test_loader, DEVICE)` возвращает словарь `class_accuracies` — accuracy отдельно для каждой из 26 букв, что позволяет увидеть, какие буквы модель распознаёт лучше/хуже.

**Сохранение модели.** После прохождения теста `unittests.exercise_5(class_accuracies)` вызывается:
```python
helper_utils.save_student_model(model=trained_model, filename='trained_student_model.pth')
```
Это сохраняет обученную модель — она используется для последующей проверки (grading) задания; ноутбук явно предупреждает, что без выполнения этой ячейки сдача задания завершится ошибкой.

### 4. Decoding the Secret Message
```python
message_imgs = helper_utils.load_hidden_message_images("./data/hidden_message_images.pkl")

def decode_word_imgs(word_imgs, model, device):
    model.eval()
    decoded_chars = []
    with torch.no_grad():
        for char_img in word_imgs:
            char_img = char_img.unsqueeze(0).to(device)
            output = model(char_img)
            _, predicted = output.max(1)
            predicted_label = predicted.item()
            lowercase_char = chr(ord("a") + predicted_label)
            decoded_chars.append(f"{lowercase_char}")
    return "".join(decoded_chars)

for sentence_imgs in message_imgs:
    decoded_sentence = []
    for word_imgs in sentence_imgs:
        decoded_word = decode_word_imgs(word_imgs, trained_model, DEVICE)
        decoded_sentence.append(decoded_word)
    print(" ".join(decoded_sentence))
```
Изображения секретного сообщения загружаются из pickle-файла `./data/hidden_message_images.pkl` (структура — список предложений, каждое предложение — список слов, каждое слово — список изображений отдельных символов). Функция `decode_word_imgs` по логике похожа на `evaluate`, но вместо подсчёта accuracy собирает предсказанные символы модели в строку: для каждого символьного изображения добавляется размерность батча (`unsqueeze(0)`), вычисляется предсказание, индекс класса переводится в букву через `chr(ord("a") + predicted_label)`, и символы объединяются в слово.

В ноутбуке отмечается, что при декодировании возможны отдельные ошибки (например, `d` распознаётся как `a`, `l` как `i`, `n` как `m`/`h`) из-за схожести написания этих букв, что согласуется с показателями `class_accuracies`. Исходный (скрытый под спойлером) текст сообщения:
> Dear Laurence,
> Hope the PyTorch course is going well.
> Do not forget to keep the labs interesting and engaging.
> Maybe you could have the students try to decode my messy handwriting.
> That might be a bit too challenging though.
> I am impressed you are able to read this.

### Вспомогательный модуль `helper_utils.py` (assignment)
- **`visualize_image(img, label=None, ax=None)`** — отображает изображение EMNIST (переводя `torch.Tensor` в `numpy`, при необходимости) с заголовком вида `EMNIST Letter: {uppercase}/{lowercase}`, сеткой и цветовой шкалой.
- **`convert_emnist_label_to_char(label)`** — переводит числовую метку EMNIST (1–26) в пару символов (заглавная/строчная буква), используя коды ASCII (`chr(64 + label)` и `chr(96 + label)`).
- **`display_data_loader_contents(data_loader)`** — печатает число изображений и число батчей в `DataLoader`, а также форму данных и меток первого батча.
- **`evaluate_per_class(model, test_loader, device)`** — прогоняет модель по всему тестовому датасету в режиме `eval()`/`no_grad()`, собирает предсказания и истинные метки (со сдвигом `targets - 1`), затем через `sklearn.metrics.accuracy_score` считает accuracy отдельно для каждого из 26 классов.
- **`save_student_model(model, filename)`** — сохраняет модель (обёрнутую в словарь `{"model": model}`) через `torch.save`.
- **`load_hidden_message_images(file_name)`** — загружает изображения секретного сообщения из pickle-файла.
- Также в файле присутствует собственная копия функции `decode_word_imgs` (дублирующая версию из ноутбука) и список `letter_ref` с оригинальным текстом сообщения (используется как справочные данные).

### Вспомогательные модули `unittests.py` и `unittests_utils.py`
Содержат тесты для автоматической проверки правильности реализованных функций (`create_emnist_dataloaders`, `initialize_emnist_model`, `train_epoch`, `evaluate`, `train_and_evaluate` — вызываются как `unittests.exercise_1(...)` … `unittests.exercise_5(...)`). Это инфраструктура для грейдинга задания, отдельно от основной логики машинного обучения, поэтому подробно не разбирается в этом конспекте.

### Ограничения и предпосылки
- Для первого запуска требуется скачивание датасета EMNIST через `torchvision.datasets.EMNIST(..., download=True)`; данные сохраняются в `/tmp/EMNIST_data`.
- Для декодирования секретного сообщения требуется локальный файл `./data/hidden_message_images.pkl`, который не является частью изученных файлов кода, но упоминается и используется в ноутбуке.
- Обучение и оценка используют `DEVICE` (CUDA, если доступна, иначе CPU); отдельная поддержка MPS (Apple Silicon), в отличие от лабораторной работы модуля, в этом assignment не реализована.
- Число эпох обучения ограничено значением от 1 до 15 (проверяется явной веткой `if num_epochs > 15 or num_epochs < 1`).
- Assignment — graded: для успешной сдачи требуется прохождение всех `unittests.exercise_N(...)` и обязательное выполнение ячейки сохранения модели (`save_student_model`) перед отправкой на проверку.
