# c2_preview.py

Источник: `03_Assignment_Building_a_Robust_CNN/c2_preview/c2_preview.py`.

## Конспект по коду

### Назначение

Модуль содержит одну функцию, `course_2_preview`, которая демонстрирует в рамках финального задания курса 1, каких результатов можно достичь с помощью техник, изучаемых в **следующем курсе** (transfer learning на предобученной модели, learning rate scheduling, более продвинутая аугментация). Используется в `C1M4_Assignment.ipynb` как "заглядывание вперёд" после того, как студент завершил собственную `SimpleCNN`.

### Импорты и зависимости

```python
import copy
from IPython.display import display
from ipywidgets import Output
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as tv_models
from torchvision import transforms
from tqdm.notebook import tqdm
```

### Сигнатура и назначение аргументов

```python
def course_2_preview(train_dataset, val_dataset, loss_function, device, num_epochs):
```

- `train_dataset`, `val_dataset` — уже загруженные датасеты (в assignment это те же `train_dataset`/`val_dataset` из CIFAR-100-подмножества, что использовались для `SimpleCNN`);
- `loss_function` — функция потерь (в assignment передаётся `nn.CrossEntropyLoss()`, определённая ранее);
- `device` — устройство вычислений;
- `num_epochs` — число эпох обучения.

Возвращает: `model` — лучший по результатам валидации экземпляр модели.

### Основные шаги

1. **Нормализация под ImageNet.** Задаются стандартные `imagenet_mean = [0.485, 0.456, 0.406]` и `imagenet_std = [0.229, 0.224, 0.225]` — так как используется модель, предобученная на ImageNet, входные данные нормализуются под её исходное распределение (а не под статистики CIFAR-100).

2. **Более продвинутые трансформации.**

   ```python
   train_transform = transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip(),
       transforms.RandomRotation(15),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
       transforms.ToTensor(),
       transforms.Normalize(imagenet_mean, imagenet_std)
   ])

   val_transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(imagenet_mean, imagenet_std)
   ])
   ```

   По сравнению с трансформациями из основной части задания добавлены: увеличение размера изображения до **224×224** (`RandomResizedCrop`/`Resize`+`CenterCrop` — требование входного размера предобученной модели), `ColorJitter` (случайное изменение яркости/контраста/насыщенности) в дополнение к отражениям и повороту.

   Трансформации присваиваются напрямую атрибуту `.transform` уже переданных датасетов: `train_dataset.transform = train_transform`, `val_dataset.transform = val_transform`.

3. **DataLoader'ы.** `batch_size = 64`; `train_loader` — с перемешиванием, `val_loader` — без.

4. **Предобученная модель — `MobileNetV3 Small`.**

   ```python
   model = tv_models.mobilenet_v3_small(weights='DEFAULT')
   ```

   Закомментированные строки в коде показывают альтернативный путь — инициализацию без весов (`weights=None`) и ручную загрузку локального checkpoint (`torch.load(".../mobilenet_v3_small-047dcff4.pth")`), но в рабочей версии используется вариант с автоматически подгружаемыми предобученными весами (`weights='DEFAULT'`).

   Финальный классификационный слой заменяется под число классов задачи:

   ```python
   num_classes = len(train_dataset.classes)
   in_features = model.classifier[3].in_features
   model.classifier[3] = nn.Linear(in_features, num_classes)
   model.to(device)
   ```

   Это классический **transfer learning**-приём: у предобученной сети сохраняются все "признаковые" слои, а заменяется только последний линейный слой классификатора, чтобы он выдавал нужное число выходов (15 классов вместо 1000 у ImageNet).

5. **Оптимизатор и scheduler.**

   ```python
   optimizer = optim.AdamW(model.parameters(), lr=1e-4)
   scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
   ```

   Используется `AdamW` (Adam с корректно реализованным weight decay) и **cosine annealing** расписание learning rate — оно плавно уменьшает learning rate по косинусоидальной кривой на протяжении `num_epochs`, что и есть та самая "динамическая настройка learning rate", упомянутая в assignment.

6. **Цикл обучения.** Для каждой из `num_epochs` эпох:

   - **Обучение**: `model.train()`, проход по `train_loader` (с прогресс-баром `tqdm`), на каждом батче — `optimizer.zero_grad()` → forward → `loss_function(outputs, labels)` → `loss.backward()` → `optimizer.step()`, накопление `running_loss`; в конце — `epoch_loss = running_loss / len(train_loader.dataset)`.
   - **Валидация**: `model.eval()`, в контексте `torch.no_grad()` — проход по `val_loader`, подсчёт `running_val_loss`, числа верных предсказаний (`torch.max(outputs, 1)`) и итоговых `epoch_val_loss`/`epoch_accuracy`.
   - Печать метрик эпохи (`Train Loss`, `Val Loss`, `Val Accuracy`).
   - `scheduler.step()` — обновление learning rate согласно расписанию cosine annealing.
   - Если `epoch_accuracy` — лучшая на данный момент, сохраняются `best_val_accuracy`, `best_epoch` и глубокая копия весов модели (`copy.deepcopy(model.state_dict())`) — тот же паттерн "запоминания лучшей эпохи", что и в `training_loop` основного assignment.

7. **Возврат лучшей модели.** После цикла, если была сохранена `best_model_state`, она загружается обратно в модель (`model.load_state_dict(best_model_state)`) перед возвратом — функция всегда возвращает версию модели с наивысшей достигнутой validation accuracy, а не с весами последней эпохи.

### Что демонстрирует код

Функция целиком реализует полный цикл **transfer learning** поверх готовой архитектуры (`MobileNetV3 Small`) с более сильной аугментацией и планировщиком learning rate — то есть именно тот "более мощный" подход, который в assignment противопоставляется собственноручно написанной `SimpleCNN`, обучаемой с нуля. В assignment эта функция запускается с `num_epochs=5` и, по тексту задания, достигает validation accuracy свыше 80% — заметно быстрее и выше, чем `SimpleCNN` за 50 эпох.

### Ограничения и зависимости

- Код статически прочитан, не выполнялся.
- Импортирует `IPython.display.display` и `ipywidgets.Output`, но в теле функции эти имена не используются напрямую — вероятно, задел на будущее использование или побочный эффект импорта окружения ноутбука.
- Требует доступа к интернету для скачивания предобученных весов `MobileNetV3 Small` (`weights='DEFAULT'`), а также библиотек `torchvision`, `tqdm` (вариант `tqdm.notebook`, то есть предполагается запуск именно в Jupyter/Colab-подобной среде).
- Ожидает, что переданные `train_dataset`/`val_dataset` уже являются объектами `Dataset` с атрибутами `.classes` и изменяемым `.transform` (как это устроено в `helper_utils.load_cifar100_subset` из основного assignment).
