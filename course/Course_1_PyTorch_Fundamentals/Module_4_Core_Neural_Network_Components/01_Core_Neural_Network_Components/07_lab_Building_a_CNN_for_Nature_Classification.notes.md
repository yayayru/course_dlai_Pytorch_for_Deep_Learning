# Lab: Building a CNN for Nature Classification

Источник: `01_Core_Neural_Network_Components/07_lab_Building_a_CNN_for_Nature_Classification/C1_M4_Lab_1_cnn_nature_classifier.ipynb` (+ `helper_utils.py`).

## Конспект по коду

### Назначение

Notebook проводит через полный процесс построения convolutional neural network (CNN) для расширенного классификатора природы (Botanical Garden app), который теперь должен распознавать не только цветы, но и насекомых и мелких животных. Демонстрируется реалистичный итеративный workflow: сначала обучается прототип на подмножестве классов, затем модель масштабируется на полный набор классов, после чего анализируется возникшая проблема переобучения (overfitting).

По ходу лабораторной работы решаются задачи:

- подготовка разнообразного датасета (подмножество **CIFAR-100**);
- построение архитектуры CNN "с нуля";
- обучение прототипа модели на упрощённом наборе из 9 классов;
- масштабирование до полной модели на 15 классов и диагностика overfitting.

### Импорты и настройки

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import helper_utils
```

Устройство выбирается автоматически: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.

### Датасет: подмножество CIFAR-100

Используется коллекция изображений из **CIFAR-100** — цветные изображения **32×32**. Из 100 классов датасета выбраны **15**, подходящих под тематику классификатора природы:

- **Цветы**: orchid, poppy, rose, sunflower, tulip;
- **Млекопитающие**: fox, porcupine, possum, raccoon, skunk;
- **Насекомые**: bee, beetle, butterfly, caterpillar, cockroach.

#### Трансформации изображений

Заданы точные значения нормализации для CIFAR-100:

```python
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
```

Определены два отдельных пайплайна трансформаций:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])
```

Поскольку все изображения CIFAR-100 уже имеют стандартный размер 32×32, шаг ресайза не нужен. Обучающий пайплайн включает аугментацию (случайное отражение по горизонтали и поворот до 15 градусов), валидационный — только преобразование в тензор и нормализацию.

#### Итеративный подход: прототип на 9 классах

Чтобы быстро показать рабочий прототип, вместо всех 15 классов сначала берётся сбалансированное подмножество из **9 классов** (по 3 из каждой категории):

```python
subset_target_classes = [
    'orchid', 'poppy', 'sunflower',       # Flowers
    'fox', 'raccoon', 'skunk',            # Mammals
    'butterfly', 'caterpillar', 'cockroach'  # Insects
]
```

Загрузка и фильтрация выполняется вспомогательной функцией `helper_utils.load_cifar100_subset(subset_target_classes, train_transform, val_transform, "/tmp/cifar_100")`, которая скачивает полный CIFAR-100, применяет трансформации и отфильтровывает только нужные классы, возвращая `train_dataset_proto` и `val_dataset_proto`.

Далее оба датасета оборачиваются в `DataLoader` с `batch_size = 64` (обучающий — с перемешиванием, валидационный — без).

Визуализация обучающих изображений выполняется вызовом `helper_utils.visualise_images(train_dataset_proto, grid=(3, 3))`.

### Архитектура `SimpleCNN`

Определяется класс `SimpleCNN(nn.Module)`, принимающий `num_classes` в конструкторе:

- **Три свёрточных блока**, каждый — `Conv2d` → `ReLU` → `MaxPool2d(kernel_size=2, stride=2)`:
  - `conv1`: `in_channels=3, out_channels=32, kernel_size=3, padding=1`;
  - `conv2`: `in_channels=32, out_channels=64`;
  - `conv3`: `in_channels=64, out_channels=128`.
- **`Flatten`** — превращает 2D feature maps в 1D вектор.
- **Полносвязная (fully connected) часть**:
  - `fc1 = nn.Linear(128 * 4 * 4, 512)` (входное изображение 32×32, после трёх pooling-слоёв — 4×4);
  - `relu4 = nn.ReLU()`;
  - `dropout = nn.Dropout(0.5)`;
  - `fc2 = nn.Linear(512, num_classes)`.

`forward` последовательно пропускает вход через три блока (conv → relu → pool), затем flatten и полносвязные слои.

В markdown-ячейках даны развёрнутые описания назначения каждого типа слоя (`nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear`, `nn.Dropout`), включая смысл параметров `in_channels`, `out_channels`, `kernel_size`, `padding`, `stride`.

### Трассировка формы тензора: `print_data_flow`

Вспомогательная функция `print_data_flow(model)` пропускает случайный тензор `torch.randn(1, 3, 32, 32)` последовательно через все слои модели и печатает форму после каждого шага. Наблюдаемый (по outputs notebook) результат для прототипа (9 классов):

```
Input shape:            torch.Size([1, 3, 32, 32])
After conv1:            torch.Size([1, 32, 32, 32])
After pool1:             torch.Size([1, 32, 16, 16])
After conv2:            torch.Size([1, 64, 16, 16])
After pool2:             torch.Size([1, 64, 8, 8])
After conv3:            torch.Size([1, 128, 8, 8])
After pool3:             torch.Size([1, 128, 4, 4])
After flatten:           torch.Size([1, 2048])
After fc1:               torch.Size([1, 512])
Output shape (fc2):      torch.Size([1, 9])
```

Это подтверждает, что при прохождении через свёрточные и pooling-слои число каналов растёт, а пространственный размер уменьшается вдвое на каждом шаге; итоговый выход даёт по одному значению на каждый из 9 классов прототипа.

### Обучение прототипа

- Функция потерь: `loss_function = nn.CrossEntropyLoss()`.
- Оптимизатор: `optimizer_prototype = optim.Adam(prototype_model.parameters(), lr=0.001)`.

Определяется функция `training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device)`, которая на каждой эпохе:

1. переводит модель в режим `train()`, проходит по батчам `train_loader`, выполняет `zero_grad()` → forward → `loss.backward()` → `optimizer.step()`, накапливает `running_loss`;
2. переводит модель в режим `eval()`, в контексте `torch.no_grad()` проходит по `val_loader`, считает `running_val_loss` и число верных предсказаний (`torch.max(outputs, 1)`);
3. печатает `Train Loss`, `Val Loss`, `Val Accuracy` за эпоху;
4. возвращает обученную модель и список метрик `[train_losses, val_losses, val_accuracies]`.

Прототип обучается **15 эпох**: `trained_proto_model, training_metrics_proto = training_loop(...)`. Затем метрики визуализируются через `helper_utils.plot_training_metrics(training_metrics_proto)` (два графика: loss и accuracy).

По тексту markdown-ячейки после обучения, точность на валидации превысила **75%** на 9-классовом подмножестве — прототип признан успешным.

Далее вызывается `helper_utils.visualise_predictions(model=trained_proto_model, data_loader=val_loader_proto, device=device, grid=(3, 3))` — визуализация предсказаний модели на случайных изображениях из валидационного набора, с подписями истинного и предсказанного класса.

### Масштабирование до полной модели (15 классов)

Определяется полный список из 15 классов (`all_target_classes`, те же классы, что перечислены выше), и через `helper_utils.load_cifar100_subset` загружаются полные `train_dataset`/`val_dataset`, оборачиваемые в `DataLoader` с тем же `batch_size = 64`.

Визуализация: `helper_utils.visualise_images(train_dataset, grid=(3, 5))`.

Создаётся новый экземпляр `SimpleCNN(num_classes=15)` (`model`) и новый оптимизатор `optimizer = optim.Adam(model.parameters(), lr=0.001)`.

Полная модель обучается **25 эпох** через тот же `training_loop`, после чего строятся графики через `plot_training_metrics`.

### Диагностика overfitting

После обучения полной модели markdown-ячейка разбирает результат: в отличие от прототипа, полная модель "упирается в стену". На графиках видно:

- **Training Loss** стабильно снижается;
- **Validation Loss** какое-то время снижается, а затем начинает расти и колебаться;
- **Validation Accuracy** выходит на плато и перестаёт заметно расти.

Это классический случай **overfitting**: модель слишком хорошо выучивает обучающие данные (включая их шум и специфические особенности) вместо общих закономерностей, которые позволили бы ей работать на новых данных. Расширяющийся разрыв между training и validation loss — явный признак того, что модель запоминает обучающую выборку вместо обобщения (generalization).

Объясняется, почему это не проявилось на прототипе из 9 классов: причина — рост **сложности задачи**. Различать 15 классов значительно труднее, чем 9, это требует более тонких признаков. Столкнувшись с более сложной задачей, мощная CNN-модель нашла более лёгкий путь снижения training loss — начала запоминать обучающие данные вместо обобщения.

Отмечается, что эта проблема overfitting будет решаться в graded assignment этого модуля путём обновления всего пайплайна.

Опциональная (закомментированная) ячейка позволяет визуализировать предсказания полной модели тем же способом (`helper_utils.visualise_predictions`).

### Вспомогательные функции (`helper_utils.py`)

- `load_cifar100_subset(target_classes, train_transform, val_transform, root)` — скачивает (если нужно) CIFAR-100, фильтрует `train`/`test` наборы по списку целевых классов через булеву маску (`np.isin`) и переиндексирует метки в непрерывный диапазон `0..N-1` через `label_map`. Возвращает отфильтрованные `train`/`test` датасеты с обновлённым атрибутом `.classes`.
- `visualise_images(dataset, grid)` — берёт по одному случайному изображению для каждого класса (используя `defaultdict(list)` с индексами по классам) и отображает их сеткой `rows × cols` через `matplotlib`, подписывая каждое изображение названием класса.
- `plot_training_metrics(metrics)` — строит два графика side-by-side: (1) Training vs Validation Loss, (2) Validation Accuracy, с общими настройками осей и шагом делений по X, вычисляемым динамически от числа эпох.
- `visualise_predictions(model, data_loader, device, grid)` — для случайного изображения каждого класса выполняет предсказание модели (`model.eval()`, `torch.no_grad()`), денормализует изображение (`cifar100_std * img + cifar100_mean`, обрезка в `[0, 1]`) и отображает его с подписью `True: ... / Pred: ...`, окрашенной в зелёный (верно) или красный (неверно) цвет.

### Ограничения и зависимости

- Notebook не запускался в рамках конспектирования — код и markdown-ячейки, а также сохранённые в notebook текстовые outputs, изучены статически; отдельные визуальные outputs (графики, сетки изображений) в самом notebook помечены как слишком большие для включения и не воспроизводятся здесь.
- Для выполнения требуется скачивание датасета CIFAR-100 (через `torchvision.datasets.CIFAR100`, путь по умолчанию `/tmp/cifar_100`) и доступ к GPU (`cuda`) либо CPU.
- Код зависит от локального модуля `helper_utils.py`, лежащего рядом с notebook.
