# Assignment: Building a Robust Data Pipeline

Источник: `03_Assignment_Building_a_Robust_Data_Pipeline/C1M3_Assignment.ipynb`
(вспомогательные модули: `03_Assignment_Building_a_Robust_Data_Pipeline/helper_utils.py`,
`03_Assignment_Building_a_Robust_Data_Pipeline/unittests.py`,
`03_Assignment_Building_a_Robust_Data_Pipeline/unittests_utils.py`)

## Конспект по коду

### Назначение

Практическое (оцениваемое, graded) задание модуля 3 — "Building a Robust Data Pipeline". В отличие от лабораторной работы, где решение уже дано полностью, здесь часть кода (`### START CODE HERE ### ... ### END CODE HERE ###`) нужно дописать самостоятельно; далее приводится **финальный, уже заполненный** вариант кода, как он присутствует в ноутбуке.

Используется датасет **Plants Classification** (Kaggle, https://www.kaggle.com/datasets/marquis03/plants-classification) — 30 000 `.jpg` изображений, 30 видов растений (алоэ вера, банан, шпинат, арбуз и т.д.). Для задания используется подвыборка из **3000 изображений**. Как и во многих реальных датасетах, изображения различаются по размеру и качеству и разложены по папкам согласно классу.

Что делается в задании:

- доступ и исследование структуры датасета с изображениями;
- построение пользовательского класса `Dataset` для загрузки изображений и меток по требованию (on demand);
- определение серии трансформаций (`resizing`, `tensor conversion`, `normalization`) для предобработки данных;
- определение трансформаций аугментации (augmentation) для расширения обучающей выборки;
- разбиение датасета на training/validation/test с применением соответствующих трансформаций к каждой части и созданием экземпляров `DataLoader` для эффективной батчевой загрузки.

Задание поделено на **3 оцениваемых упражнения** (graded exercises), проверяемых юнит-тестами из `unittests.py`.

### Импорты и зависимости

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import helper_utils
import unittests
```

### Часть 1 — Data Access

**Скачивание и структура датасета** (директория `/tmp/plants_dataset`, скачивается через `gdown` и распаковывается через `unzip` в bash-ячейке). Структура (`helper_utils.print_data_folder_structure`):

```
plants_dataset/
├── classname.txt
├── df_labels.csv
├── df_labels_orig.csv
├── aloevera/
├── banana/
├── bilimbi/
... (всего 30 папок классов) ...
└── watermelon/
```

- `df_labels.csv` — содержит пути к изображениям (колонка `image:FILE`, например `aloevera/aloevera700.jpg`) и числовую метку класса (колонка `category`).
- `classname.txt` — список названий 30 классов (по одному на строку): `['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peperchili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweetpotatoes', 'tobacco', 'waterapple', 'watermelon']`.

#### Exercise 1 — `PlantsDataset`

Пользовательский класс `Dataset`, наследующий от `torch.utils.data.Dataset`. Часть методов уже дана (`read_df`, `read_classname`, `load_labels`, `get_label_description`, `retrieve_image`), нужно было дописать `__init__`, `__len__`, `__getitem__`:

```python
class PlantsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df_info = self.read_df()

        # Load labels from the DataFrame using the `load_labels` method
        self.labels = self.load_labels(self.df_info)
        # Create a mapping from label integers to class names
        self.class_names = self.read_classname()

    def read_df(self):
        path_csv = self.root_dir + "/df_labels.csv"
        df = pd.read_csv(path_csv)
        return df

    def read_classname(self):
        path_txt = self.root_dir + "/classname.txt"
        with open(path_txt, "r") as f:
            class_names = f.read().splitlines()
        return class_names

    def load_labels(self, df):
        labels = []
        for idx, row in df.iterrows():
            label_int = row["category"]
            labels.append(label_int)
        return labels

    def get_label_description(self, label: int):
        description = self.class_names[label]
        return description

    def retrieve_image(self, idx: int):
        img_path = self.root_dir + "/" + self.df_info.iloc[idx]["image:FILE"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        return image

    def __len__(self):
        length = len(self.labels)
        return length

    def __getitem__(self, idx):
        image = self.retrieve_image(idx)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
```

Проверка (`plants_dataset = PlantsDataset(root_dir=path_dataset, transform=None)`):

- `len(plants_dataset)` → **3000**.
- `plants_dataset[10]` → изображение `(269, 187)`, описание `aloevera`.
- Тесты: `unittests.exercise_1(PlantsDataset)` → **все тесты прошли** (`All tests passed!`).

**Визуальный обзор** датасета через `helper_utils.visual_exploration(plants_dataset, num_rows=2, num_cols=4)` — показывает, что изображения различаются по размеру, цвету и фону, что типично для реальных датасетов и подчёркивает необходимость предобработки (resizing, normalization, augmentation) для лучшей генерализации модели.

### Часть 2 — Transformations

#### 2.1 Вычисление среднего и стандартного отклонения

Вспомогательная (уже данная в задании) функция `get_mean_std(dataset)` вычисляет **mean** и **std** обучающего датасета для нормализации — вычисляется **после** изменения размера и конвертации в тензор, поскольку эти шаги меняют распределение значений пикселей:

```python
def get_mean_std(dataset: Dataset):
    preprocess = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )

    # First pass: compute mean
    total_pixels = 0
    sum_pixels = torch.zeros(3)

    for img, _ in dataset:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)  # [channels, pixels]
        sum_pixels += pixels.sum(dim=1)
        total_pixels += pixels.size(1)

    mean = sum_pixels / total_pixels

    # Second pass: compute std
    sum_squared_diff = torch.zeros(3)

    for img, _ in dataset:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)  # [channels, pixels]
        diff = pixels - mean.unsqueeze(1)
        sum_squared_diff += (diff ** 2).sum(dim=1)

    std = torch.sqrt(sum_squared_diff / total_pixels)

    return mean, std
```

- Первый проход по датасету: суммируются значения пикселей по каждому каналу (после `Resize(128,128)` + `ToTensor()`), делится на общее число пикселей — получается канальное среднее (**mean**).
- Второй проход: вычисляется сумма квадратов отклонений от среднего, делится на число пикселей и извлекается корень — получается канальное стандартное отклонение (**std**).
- **Примечание из задания**: обычно mean/std следует считать только на обучающей выборке — иначе возникает **утечка данных (data leakage)**, при которой информация из тестовой/валидационной выборки влияет на обучение. Здесь, поскольку датасет ещё не разбит на этом этапе, статистики для простоты считаются на всём датасете (для этого конкретного датасета итоговые значения почти не изменились бы при расчёте только на train).

Результат: `mean = tensor([0.6659, 0.6203, 0.4784])`, `std = tensor([0.2888, 0.2884, 0.3426])`.

#### 2.2 Exercise 2 — `get_transformations`

Функция создаёт два конвейера трансформаций: базовый (`main_transform`) и с аугментацией (`transform_with_augmentation`):

```python
def get_transformations(mean, std):
    main_tfs = [
        # Resize images to 128x128 pixels
        transforms.Resize((128, 128)),
        # Convert images to PyTorch tensors
        transforms.ToTensor(),
        # Normalize images using the provided mean and std
        transforms.Normalize(mean=mean, std=std),
    ]

    augmentation_tfs = [
        # Randomly flip the image vertically
        transforms.RandomVerticalFlip(p=0.5),
        # Randomly rotate the image by ±15 degrees
        transforms.RandomRotation(degrees=15),
    ]

    # Compose the main transformations into a single pipeline
    main_transform = transforms.Compose(main_tfs)

    transform_with_augmentation = transforms.Compose(augmentation_tfs + main_tfs)

    return main_transform, transform_with_augmentation
```

Отличия от аугментации, показанной в видео/лабораторной (там — `RandomHorizontalFlip` + `RandomRotation(10)` + `ColorJitter`): здесь используется **`RandomVerticalFlip(p=0.5)`** (вертикальное отражение) и **`RandomRotation(degrees=15)`** (поворот ±15°), без `ColorJitter`. Аугментационные трансформации применяются **до** основных (`augmentation_tfs + main_tfs`).

Вывод при печати конвейеров:

```
Compose(
    Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=tensor([0.6659, 0.6203, 0.4784]), std=tensor([0.2888, 0.2884, 0.3426]))
)
Compose(
    RandomVerticalFlip(p=0.5)
    RandomRotation(degrees=[-15.0, 15.0], interpolation=nearest, expand=False, fill=0)
    Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=tensor([0.6659, 0.6203, 0.4784]), std=tensor([0.2888, 0.2884, 0.3426]))
)
```

Тесты: `unittests.exercise_2(get_transformations)` → **все тесты прошли**.

Проверка на образце изображения: `main_transform(img)` → форма `torch.Size([3, 128, 128])`; аугментированная версия визуализируется через `helper_utils.Denormalize` (обратная нормализация) + `helper_utils.plot_img`.

### Часть 3 — Data Loading

**Класс `SubsetWithTransform`** (уже дан в задании, аналогичен классу из лабораторной работы) — оборачивает `Subset`, позволяя назначить свою трансформацию каждой части после `random_split` (иначе все части наследуют трансформацию исходного датасета):

```python
class SubsetWithTransform(Dataset):
    """A subset of a dataset with a specific transform applied."""

    def __init__(self, subset: Subset, transform=None):
        # subset should be a subset WITHOUT transform
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

#### Exercise 3 — `get_dataloaders`

```python
def get_dataloaders(
    dataset,
    batch_size,
    val_fraction,
    test_fraction,
    main_transform,
    augmentation_transform,
):
    # Calculate the sizes of each split
    total_size = len(dataset)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataset with the corresponding transforms for each split
    train_dataset = SubsetWithTransform(train_dataset, transform=augmentation_transform)
    val_dataset = SubsetWithTransform(val_dataset, transform=main_transform)
    test_dataset = SubsetWithTransform(test_dataset, transform=main_transform)

    # Create DataLoaders for each split
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
```

- `train_size` вычисляется как остаток после вычитания `val_size` и `test_size` — избегает ошибок округления.
- Обучающая часть получает `augmentation_transform`, валидационная и тестовая — `main_transform` (без аугментации).
- `shuffle=True` только для `train_loader`.

Вызов:

```python
train_loader, val_loader, test_loader = get_dataloaders(
    dataset=plants_dataset,
    batch_size=32,
    val_fraction=0.15,
    test_fraction=0.2,
    main_transform=main_transform,
    augmentation_transform=transform_with_augmentation,
)
```

Обратите внимание: здесь `test_fraction=0.2` (не 0.15, как в лабораторной работе), поэтому разбиение получается **65% train / 15% val / 20% test** от 3000 образцов.

Результат проверки:

```
=== Train Loader ===
Number of batches in train_loader: 61
Number of samples in train_dataset: 1950
Transforms applied to train_dataset: Compose(
    RandomVerticalFlip(p=0.5)
    RandomRotation(degrees=[-15.0, 15.0], interpolation=nearest, expand=False, fill=0)
    Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=tensor([0.6659, 0.6203, 0.4784]), std=tensor([0.2888, 0.2884, 0.3426]))
)
train_dataset type: <class '__main__.SubsetWithTransform'>

=== Test Loader ===
Number of batches in test_loader: 19
Number of samples in test_dataset: 600
Transforms applied to test_dataset: Compose(
    Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=True)
    ToTensor()
    Normalize(mean=tensor([0.6659, 0.6203, 0.4784]), std=tensor([0.2888, 0.2884, 0.3426]))
)
test_dataset type: <class '__main__.SubsetWithTransform'>
```

(1950 train / 450 val / 600 test = 3000; 1950/32 ≈ 61 батч, 600/32 ≈ 19 батчей.)

Тесты: `unittests.exercise_3(get_dataloaders, plants_dataset)` → **все тесты прошли**.

### Вспомогательный модуль `helper_utils.py` (assignment)

Похож на одноимённый модуль из лабораторной работы, но с отличиями:

- **`Denormalize`** — здесь класс вынесен непосредственно в `helper_utils` (а не определяется в самом ноутбуке, как в лабораторной): вычисляет обратные `mean`/`std` и применяет `transforms.Normalize` для отмены нормализации перед визуализацией.
- `plot_img`, `get_grid`, `print_data_folder_structure`, `explore_extensions` — функционально аналогичны версии из лабораторной работы (используют `fastai.vision.core.show_image`/`show_titled_image` и `directory_tree.DisplayTree`).
- **`visual_exploration(dataset, num_rows, num_cols)`** — здесь тоже вынесена в `helper_utils` (а не определена в самом ноутбуке): выбирает случайные индексы (`np.random.choice`, без повторов), отображает сетку изображений с меткой (номер + текстовое описание через `dataset.get_label_description`) и информацией об индексе/размере.

### Инфраструктура тестирования (`unittests.py`, `unittests_utils.py`)

Файлы `unittests.py` и `unittests_utils.py` — служебная инфраструктура автоматической проверки решений, не являются учебным материалом по PyTorch как таковым. `unittests.py` определяет три функции проверки — `exercise_1(learner_class)`, `exercise_2(learner_func)`, `exercise_3(learner_func, dataset)` — которые прогоняют реализации студента (`PlantsDataset`, `get_transformations`, `get_dataloaders`) на тестовых случаях и печатают `All tests passed!` при успехе. `unittests_utils.py` содержит общие вспомогательные функции для этой проверки (эталонные значения, сравнение и т.п.).

### Заключение ноутбука

В результате выполнения задания построен полный сквозной (end-to-end) конвейер данных в PyTorch для реального датасета изображений: пользовательский `Dataset` для доступа к данным, последовательность трансформаций (resizing, normalization, augmentation) для повышения устойчивости обучения, разбиение на train/validation/test и `DataLoader` для эффективной батчевой загрузки и итерирования. Эти три компонента — `Dataset`, `Transforms` и `DataLoader` — образуют чистый, эффективный и переиспользуемый рабочий процесс для подготовки любого датасета изображений к обучению нейросети.

### Ограничения и зависимости

- Требует скачивания датасета Plants Classification через `gdown` (Google Drive) и его распаковки в `/tmp/plants_dataset` — нужен доступ к сети.
- Часть кода (`### START CODE HERE ### ... ### END CODE HERE ###`) в исходном (незаполненном) варианте задания предназначена для самостоятельного заполнения студентом; в этом конспекте отражён уже заполненный, финальный вариант, присутствующий в ноутбуке.
- Раздел "Expected Output" в ноутбуке (ячейка после Exercise 2) показывает пример вывода с другими значениями `std` (`[0.2119, 0.2155, 0.2567]`), чем фактически получено в выполненном ноутбуке (`[0.2888, 0.2884, 0.3426]`) — расхождение, вероятно, связано с иной версией/выборкой датасета на момент подготовки эталонного вывода; сохранено как есть, без исправления.
- `helper_utils.py` использует внешние зависимости `directory_tree` и `fastai.vision.core`.
- Код ноутбука статически прочитан; ячейки не выполнялись повторно — вывод и результаты в конспекте взяты из сохранённых outputs ноутбука.
