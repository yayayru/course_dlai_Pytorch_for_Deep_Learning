# Лабораторная работа: Data Management

Источник: `01_Data_Management_in_PyTorch/06_lab_Data_Management/C1_M3_Lab_data_management.ipynb`
(вспомогательный модуль: `01_Data_Management_in_PyTorch/06_lab_Data_Management/helper_utils.py`)

## Конспект по коду

### Назначение

Лабораторная работа "Data Management" — практическое продолжение видео модуля 3. В предыдущих лабораторных использовались хорошо структурированные данные, но в реальном мире данные редко идеальны. Даже самая мощная архитектура модели может провалиться, если её "кормить" грязным, неэффективным или ненадёжным потоком данных — поэтому нужен надёжный **data pipeline**.

В лабораторной используется реальный датасет **Oxford 102 Flowers**: изображения и метки хранятся в отдельных файлах, с непоследовательным форматированием и, возможно, повреждёнными образцами. Для решения этих проблем используются ключевые инструменты PyTorch для работы с данными — классы **`Dataset`** и **`DataLoader`**.

Цели лабораторной:

- исследовать реальный датасет с неорганизованными файлами и отдельно хранимыми метками;
- построить пользовательский PyTorch `Dataset` для загрузки и предобработки изображений и меток "на лету" (on-the-fly);
- применить трансформации (transformations) и аугментацию данных (data augmentation) для подготовки данных и повышения устойчивости модели;
- использовать `DataLoader` для эффективного создания и перемешивания батчей для обучения;
- разбить данные на training, validation и test выборки;
- реализовать техники обработки ошибок для управления проблемами с данными и мониторинга производительности конвейера.

### Импорты и зависимости

```python
import os
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import helper_utils
```

Локальный модуль `helper_utils.py` содержит вспомогательные функции для визуализации и отладки (см. ниже).

### Часть 1: Data Access — доступ к данным

**Скачивание датасета** — функция `download_dataset()`:

- Директория данных: `/tmp/flower_data`.
- Проверяет, существуют ли уже локально папка с изображениями (`jpg/`) и файл меток (`imagelabels.mat`) — если да, скачивание пропускается.
- Иначе скачивает архив с изображениями (`102flowers.tgz`) и файл меток (`imagelabels.mat`) с `https://www.robots.ox.ac.uk/~vgg/data/flowers/102/` через библиотеку `requests` (с прогресс-баром через `tqdm`), затем распаковывает `.tgz` через `tarfile`.
- Дополнительно создаёт файл `labels_description.txt` со списком из 102 текстовых названий классов цветов (например, `pink primrose`, `hard-leaved pocket orchid`, ..., `blackberry lily`) — по одному названию на строку, индекс строки соответствует числовой метке.

Структура датасета после скачивания (`helper_utils.print_data_folder_structure`):

```
flower_data/
├── 102flowers.tgz
├── imagelabels.mat
├── jpg/
└── labels_description.txt
```

То есть датасет состоит из: папки `jpg` с изображениями в формате JPEG, файла `imagelabels.mat` с метками (формат MATLAB) и файла `labels_description.txt` с текстовым описанием каждой метки.

**Класс `FlowerDataset(Dataset)`** — базовый пользовательский датасет:

```python
class FlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(self.root_dir, "jpg")
        self.labels = self.load_and_correct_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.retrieve_image(idx)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

    def retrieve_image(self, idx):
        img_name = f"image_{idx + 1:05d}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        return image

    def load_and_correct_labels(self):
        self.labels_mat = scipy.io.loadmat(
            os.path.join(self.root_dir, "imagelabels.mat")
        )
        labels = self.labels_mat["labels"][0] - 1
        return labels

    def get_label_description(self, label):
        path_labels_description = os.path.join(self.root_dir, "labels_description.txt")
        with open(path_labels_description, "r") as f:
            lines = f.readlines()
        description = lines[label].strip()
        return description
```

Ключевые моменты:

- `load_and_correct_labels`: читает `.mat`-файл через `scipy.io.loadmat`, вычитает 1 из меток (коррекция с 1-based индексации MATLAB на 0-based индексацию Python) — предотвращает off-by-one ошибки при обучении и оценке.
- `retrieve_image`: строит имя файла по индексу (`image_00001.jpg` и т.д., +1 к индексу из-за нумерации файлов с 1), открывает через **Pillow**, конвертирует в **RGB** для единообразия.
- `get_label_description`: возвращает человекочитаемое название класса по числовой метке.
- **Ленивая загрузка (lazy loading)**: изображения не загружаются все сразу при создании объекта датасета, а загружаются "на лету" через `__getitem__` — экономит память на больших датасетах.

Проверка датасета после создания (`dataset = FlowerDataset(path_dataset)`):

- `len(dataset)` → **8189** образцов.
- Пример: `dataset[10]` → изображение размером `(500, 748)`, метка `76`.
- Визуализация через `helper_utils.plot_img(img, label=label, info=img_size_info)`.
- Проверка всех 102 уникальных меток и их текстовых описаний (`dataset.get_label_description(label)`) — от `0: pink primrose` до `101: blackberry lily`.

**Функция `visual_exploration(dataset, num_rows, num_cols)`** — визуализирует сетку случайных образцов из датасета (случайные индексы через `np.random.choice`, без повторов), для каждого показывает изображение, метку с описанием и информацию об индексе/размере. Используется для быстрого визуального обзора датасета и выявления **проблем качества (quality problems)** — как видно, размеры изображений в датасете сильно различаются.

### Часть 2: Quality Problems — трансформации (Transformations)

Так как размер изображений сильно варьируется, а большинство моделей ожидают одинаковый размер входа, применяется конвейер трансформаций через `torchvision.transforms`.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    # images transforms
    transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
    transforms.CenterCrop(224),     # Center crop to 224x224 pixels
    # bridge to tensor
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    # tensor transforms
    transforms.Normalize(mean=mean, std=std),
])
```

Порядок трансформаций важен: изменение размера и обрезка (`Resize`, `CenterCrop`) применяются **до** конвертации в тензор (`ToTensor`), а нормализация (`Normalize`) — уже после, на тензоре.

Новый экземпляр датасета с трансформацией: `dataset_transformed = FlowerDataset(path_dataset, transform=transform)`. Проверка того же образца (`dataset_transformed[sel_idx]`) через `helper_utils.quick_debug(img_transformed)`:

```
Shape: torch.Size([3, 224, 224])
Type: torch.float32
Range of pixel values: [-2.1, 2.6]
```

Изображение теперь тензор `[3, 224, 224]` (3 канала RGB, 224×224 пикселя). Из-за нормализации значения пикселей выходят за пределы `[0, 255]`, поэтому при прямой визуализации изображение выглядит искажённым (matplotlib выдаёт предупреждение о клиппинге диапазона).

**Класс `Denormalize`** — обратная нормализация для корректной визуализации:

```python
class Denormalize:
    def __init__(self, mean, std):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        self.denormalize = transforms.Normalize(mean=new_mean, std=new_std)

    def __call__(self, tensor):
        return self.denormalize(tensor)
```

Вычисляет обратные параметры (`new_mean`, `new_std`) и применяет `transforms.Normalize` с ними, чтобы отменить исходную нормализацию перед визуализацией.

### Часть 3: Data Loading — разбиение и DataLoader

Полный процесс обучения обычно включает три стадии: **training** (модель обучается, веса корректируются по функции потерь), **validation** (оценка на отдельных данных для подбора гиперпараметров и предотвращения переобучения), **evaluation/test** (оценка обобщающей способности на невиданных данных).

**Функция `split_dataset`**:

```python
def split_dataset(dataset, val_fraction=0.15, test_fraction=0.15):
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_dataset, val_dataset, test_dataset
```

По умолчанию делит данные 70% / 15% / 15%. `train_size` вычисляется как остаток, чтобы избежать ошибок округления. Использует `random_split` из `torch.utils.data`.

Результат для `dataset_transformed` (8189 образцов):

```
Length of training dataset:   5733
Length of validation dataset: 1228
Length of test dataset:       1228
```

**DataLoader**'ы для каждой выборки:

```python
batch_size = 32

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

- `shuffle=True` для обучающей выборки, `shuffle=False` для валидационной и тестовой.
- Демонстрационный цикл на 2 эпохи (`n_epochs = 2`) проходит по `train_dataloader` и `val_dataloader` с прогресс-барами (`helper_utils.get_dataloader_bar` / `update_dataloader_bar`), затем один финальный проход по `test_dataloader`.
- Наблюдение: обучающий `DataLoader` даёт всего **180 батчей** (для 5733 образцов), валидационный и тестовый — по **39 батчей** каждый (для 1228 образцов).

### Часть 4: Augmentation — аугментация данных

Аугментация данных повышает устойчивость и обобщающую способность модели: случайные трансформации (отражение, поворот, изменение яркости) во время обучения помогают модели распознавать объекты в разных реальных условиях. В PyTorch аугментация выполняется "на лету" (on-the-fly) — генерирует бесконечные вариации изображений без дополнительного хранилища.

**Функция `get_augmentation_transform(mean, std)`**:

```python
def get_augmentation_transform(mean, std):
    augmentations_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),   # горизонтальное отражение с вероятностью 0.5
        transforms.RandomRotation(degrees=10),    # поворот в диапазоне ±10 градусов
        transforms.ColorJitter(brightness=0.2),   # случайное изменение яркости/контраста/насыщенности/оттенка
    ]

    main_transforms = [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform = transforms.Compose(augmentations_transforms + main_transforms)
    return transform
```

Новый датасет с аугментацией: `dataset_augmented = FlowerDataset(path_dataset, transform=augmentation_transform)`.

**Функция `visualize_augmentations(dataset_aug, idx, num_versions)`** — отладочный инструмент: многократно достаёт один и тот же образец по индексу (каждый раз с новой случайной аугментацией из-за случайных трансформаций), денормализует изображения (`Denormalize`) и показывает сетку из `num_versions` (например, 8) разных версий одного цветка — позволяет визуально проверить, что аугментация работает разумно (не слишком слабая и не слишком агрессивная).

**Разбиение с аугментацией: проблема `Subset` и класс `SubsetWithTransform`**

Тонкий момент: при использовании `random_split` каждая из частей — это объект `Subset`, ссылающийся на исходный датасет, поэтому все части наследуют трансформацию, заданную в исходном датасете, и им нельзя напрямую назначить разные трансформации.

Решение — класс-обёртка `SubsetWithTransform(Dataset)`:

```python
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

Применение — аугментация только для обучающей выборки, обычный transform для validation/test:

```python
train_dataset = SubsetWithTransform(train_dataset, transform=augmentation_transform)
val_dataset = SubsetWithTransform(val_dataset, transform=transform)
test_dataset = SubsetWithTransform(test_dataset, transform=transform)
```

Проверка (`print(train_dataset.transform)` и т.д.) подтверждает: у `train_dataset` конвейер включает `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` + стандартные шаги; у `val_dataset`/`test_dataset` — только стандартные шаги без аугментации.

### Часть 5: Robust Datasets — устойчивые датасеты

В реальных проектах конвейеры данных должны быть устойчивы к неожиданным проблемам: повреждённым файлам, несогласованным форматам изображений, проблемным образцам, способным "уронить" процесс обучения.

**Класс `RobustFlowerDataset(Dataset)`** — расширяет идею `FlowerDataset`, добавляя обработку ошибок:

```python
class RobustFlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "jpg")
        self.transform = transform
        self.labels = self.load_and_correct_labels()
        self.error_logs = []

    def __getitem__(self, idx):
        for attempt in range(len(self)):
            try:
                image = self.retrieve_image(idx)
                if self.transform:
                    image = self.transform(image)
                label = self.labels[idx]
                return image, label
            except Exception as e:
                self.log_error(idx, e)
                idx = (idx + 1) % len(self)

    def __len__(self):
        return len(self.labels)

    def retrieve_image(self, idx):
        img_name = f"image_{idx+1:05d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        with Image.open(img_path) as img:
            img.verify()
        image = Image.open(img_path)
        image.load()
        if image.size[0] < 32 or image.size[1] < 32:
            raise ValueError(f"Image too small: {image.size}")
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def load_and_correct_labels(self):
        self.labels_mat = scipy.io.loadmat(
            os.path.join(self.root_dir, "imagelabels.mat")
        )
        labels = self.labels_mat["labels"][0] - 1
        labels = labels[:10]  # усечение до первых 10 меток для быстрого тестирования
        return labels

    def log_error(self, idx, e):
        img_name = f"image_{idx + 1:05d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        self.error_logs.append({
            "index": idx,
            "error": str(e),
            "path": img_path if "img_path" in locals() else "unknown",
        })
        print(f"Warning: Skipping corrupted image {idx}: {e}")

    def get_error_summary(self):
        if not self.error_logs:
            print("No errors encountered - dataset is clean!")
        else:
            print(f"\nEncountered {len(self.error_logs)} problematic images:")
            for error in self.error_logs[:5]:
                print(f"  Index {error['index']}: {error['error']}")
            if len(self.error_logs) > 5:
                print(f"  ... and {len(self.error_logs) - 5} more")
```

Ключевые отличия от `FlowerDataset`:

- `__getitem__` оборачивает загрузку в `try/except`: при исключении (например, повреждённый файл) вызывается `log_error`, затем делается попытка загрузить **следующий** индекс (`idx = (idx + 1) % len(self)`), с ограничением числа попыток `len(self)`, чтобы избежать бесконечного цикла.
- `retrieve_image` проверяет целостность файла через `img.verify()` (Pillow), затем переоткрывает и полностью загружает изображение (`image.load()`); проверяет размер (ошибка, если меньше 32 пикселей по любой стороне); конвертирует не-RGB (например, grayscale) изображения в RGB.
- `get_error_summary` выводит сводку по всем накопленным ошибкам (первые 5 + количество остальных).

**Демонстрация на намеренно повреждённом датасете**: датасет Oxford Flowers 102 в оригинале чистый, но для иллюстрации техник устойчивости часть изображений в отдельной папке `./data/corrupted_flower_data` была намеренно повреждена (`corrupted_dataset_path = './data/corrupted_flower_data'`, `robust_dataset = RobustFlowerDataset(corrupted_dataset_path)`).

Продемонстрированные случаи (при `load_and_correct_labels` усечении до 10 меток):

- **Индекс 2** — изображение слишком маленькое (`Image too small: (20, 20)`) → пропускается, возвращается изображение по индексу 3.
- **Индекс 4** — grayscale-изображение (mode `'L'`) → автоматически конвертируется в RGB (`img.mode` после загрузки становится `'RGB'`), не пропускается, так как технически валидно.
- **Индекс 6** — файл не читается вовсе (`cannot identify image file '.../image_00007.jpg'`) → пропускается, возвращается изображение по индексу 7.

`robust_dataset.get_error_summary()` выводит:

```
Encountered 2 problematic images:
  Index 2: Image too small: (20, 20)
  Index 6: cannot identify image file './data/corrupted_flower_data/jpg/image_00007.jpg'
```

### Часть 6: Tracking Errors — мониторинг конвейера

Помимо обработки исключений важно систематически отслеживать и анализировать ошибки и аномалии, возникающие при загрузке и предобработке данных: логировать проблемные образцы, отслеживать, какие изображения загружаются (и как часто), и просматривать сводки ошибок после обучения.

**Класс `MonitoredDataset(RobustFlowerDataset)`**:

```python
class MonitoredDataset(RobustFlowerDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_counts = {}
        self.load_times = []

    def __getitem__(self, idx):
        import time
        start_time = time.time()
        self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
        result = super().__getitem__(idx)
        load_time = time.time() - start_time
        self.load_times.append(load_time)
        if load_time > 1.0:
            print(f"⚠️ Slow load: Image {idx} took {load_time:.2f}s")
        return result

    def print_stats(self):
        print("\n=== Pipeline Statistics ===")
        print(f"Total images: {len(self)}")
        print(f"Unique images accessed: {len(self.access_counts)}")
        print(f"Errors encountered: {len(self.error_logs)}")
        if self.load_times:
            avg_time = sum(self.load_times) / len(self.load_times)
            max_time = max(self.load_times)
            print(f"Average load time: {avg_time*1000:.1f} ms")
            print(f"Slowest load: {max_time*1000:.1f} ms")
        all_indices = set(range(len(self)))
        accessed_indices = set(self.access_counts.keys())
        never_accessed = all_indices - accessed_indices
        if never_accessed:
            print(f"\n⚠️ WARNING: {len(never_accessed)} images were never loaded!")
            print(f"   Examples: {list(never_accessed)[:5]}")
```

- **Access Tracking**: при каждом вызове `__getitem__` увеличивается счётчик обращений `self.access_counts[idx]`.
- **Load Time Measurement**: измеряется время загрузки каждого образца (`self.load_times`); если загрузка занимает больше 1 секунды — выводится предупреждение.
- **Statistics Reporting** (`print_stats()`): выводит общее число изображений, число уникальных загруженных индексов, число ошибок (унаследовано от `RobustFlowerDataset`), среднее и максимальное время загрузки, а также предупреждение, если какие-то изображения ни разу не были загружены (с примерами индексов).

Демонстрация: полный проход по `monitored_dataset` (10 образцов, усечённый датасет) → 2 предупреждения о повреждённых изображениях (индексы 2 и 6), затем `print_stats()`:

```
=== Pipeline Statistics ===
Total images: 10
Unique images accessed: 10
Errors encountered: 2
Average load time: 1.4 ms
Slowest load: 1.9 ms
```

Такой мониторинг помогает выявлять "бутылочные горлышки" (bottlenecks), диагностировать ошибки и обеспечивать надёжность конвейера данных — это хорошая практика для встраивания в собственные классы `Dataset`.

### Вспомогательный модуль `helper_utils.py`

Содержит функции, используемые в ноутбуке:

- `get_dataloader_bar(dataloader, color)` / `update_dataloader_bar(p_bar, batch, current_bs, n_samples)` — создание и обновление прогресс-бара `tqdm` для визуализации прохода по `DataLoader` (показывает номер батча и количество обработанных образцов из общего числа).
- `plot_img(img, label=None, info=None, ax=None)` — отображение изображения (через `fastai.vision.all.show_image` / `show_titled_image`) с опциональной подписью-меткой и дополнительным текстом информации под изображением.
- `get_grid(num_rows, num_cols, figsize)` — создаёт сетку `matplotlib`-сабплотов, нормализуя оси в единообразный 2D-формат для случаев с одной строкой/колонкой.
- `print_data_folder_structure(root_dir, max_depth)` — выводит дерево структуры папки датасета (через `directory_tree.DisplayTree`).
- `explore_extensions(root_dir)` — обходит директорию (`os.walk`) и группирует пути файлов по расширению (в нижнем регистре) в словарь.
- `quick_debug(img)` — печатает форму (shape), тип (dtype) и диапазон значений тензора изображения — быстрая диагностика после трансформаций (ожидаемо: `[3, 224, 224]`, `torch.float32`, диапазон примерно `[-2, 2]` после нормализации).

### Заключение ноутбука

В результате лабораторной построен полноценный, готовый к продакшену (production-ready) конвейер данных в PyTorch: организация "грязного" датасета через пользовательский класс `Dataset`, загружающий изображения и метки из отдельных файлов; применение трансформаций для нормализации данных и аугментации для повышения устойчивости будущей модели; использование `DataLoader` для батчевой загрузки с перемешиванием; повышение надёжности конвейера через обработку ошибок и мониторинг производительности.

### Ограничения и зависимости

- Требует доступа к сети для скачивания датасета (либо уже закэшированные данные в `/tmp/flower_data`).
- Для отладочного раздела "Robust Datasets" требуется отдельная, намеренно повреждённая версия датасета в папке `./data/corrupted_flower_data` (не создаётся кодом в этом ноутбуке — предполагается, что предоставлена отдельно).
- `helper_utils.py` использует внешние зависимости `directory_tree` и `fastai.vision.all` (для отображения дерева файлов и изображений) в дополнение к стандартным `matplotlib`/`tqdm`.
- Код ноутбука статически прочитан; ячейки не выполнялись повторно — вывод и результаты в конспекте взяты из сохранённых outputs ноутбука.
