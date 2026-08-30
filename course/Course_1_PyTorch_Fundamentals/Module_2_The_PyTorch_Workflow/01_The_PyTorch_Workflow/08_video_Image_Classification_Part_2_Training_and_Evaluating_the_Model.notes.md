# Image Classification. Часть 2: обучение и оценка модели

## Источники
- Транскрипция: `08_video_Image_Classification_Part_2_Training_and_Evaluating_the_Model.trans.txt`
- Презентация модуля: `PyTorch_C1_M2.pdf` (использована для сверки синтаксиса кода)

## Вступление
В прошлом видео были построены два ключевых компонента: пайплайн данных, загружающий изображения MNIST, и архитектура нейронной сети, способная их обрабатывать. Но если запустить модель прямо сейчас, она будет просто угадывать. В этом видео разбирается код, который реально обучает эту модель, шаг за шагом.

## Настройка перед обучением

### Устройство, loss-функция и оптимизатор
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

model = MNISTClassifier().to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
- Сначала выбирается устройство: если CUDA доступна, используется GPU, иначе PyTorch переключается обратно на CPU.
- Модель создаётся и переносится на устройство с помощью `.to(device)`. И модель, и данные должны быть на одном устройстве — иначе во время обучения возникнет ошибка.
- В качестве loss-функции используется `CrossEntropyLoss` — как уже говорилось ранее, она предназначена для задач классификации, где модель выбирает один класс из многих, и это идеально подходит для выбора цифры от 0 до 9.
- Используется оптимизатор Adam с learning rate `0.001`. Adam адаптирует свой learning rate по ходу обучения: делает бо́льшие корректировки в начале, когда градиенты «шумные» (noisy), и меньшие корректировки позже, когда обучение стабилизируется.

## Функция обучения на одну эпоху

```python
def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            accuracy = 100. * correct / total
            print(f'  [{batch_idx * 64}/{60000}] '
                  f'Loss: {avg_loss:.3f} | Accuracy: {accuracy:.1f}%')
            running_loss = 0.0
```
Функция принимает пять аргументов: модель, data loader, loss-функцию, оптимизатор и устройство (device), на котором всё должно выполняться.

- `model.train()` переводит модель в режим обучения (training mode).
- Настраиваются три переменные для отслеживания прогресса: `running_loss` накапливает значения loss, `correct` считает предсказания, совпавшие с истинными метками, `total` считает все виденные к текущему моменту образцы.
- Далее выполняется цикл по всем батчам. Для каждого батча: данные и target переносятся на нужное устройство; очищаются оставшиеся градиенты с помощью `optimizer.zero_grad()`; выполняется forward pass — вызов модели с данными для получения outputs; вычисляется loss через loss-функцию; выполняется backpropagation через `loss.backward()`; веса обновляются через `optimizer.step()`.
- Затем отслеживается прогресс: `loss.item()` даёт числовое значение, а `output.max()` показывает, какой класс цифры получил наивысший балл, что позволяет сравнить его с меткой (label).
- Каждые 100 батчей выводятся текущие loss и accuracy. При 60 000 обучающих изображениях и batch size 64 это около 938 батчей на эпоху, то есть около 9 обновлений за эпоху.

В примере из видео loss падает с 0.64 до примерно 0.17, а accuracy поднимается примерно с 81% до примерно 95% — и это всего за один проход по обучающему набору данных.

## Функция оценки (evaluation)

```python
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')
```
Обучение — только половина истории. Модель нужно протестировать на новых, невиданных данных. Есть две ключевые особенности, отличающие evaluation от обучения:
1. `model.eval()` переключает модель в режим оценки (evaluation mode).
2. Всё оборачивается в `with torch.no_grad()`. Во время оценки обучение не выполняется, поэтому градиенты не нужны — это экономит память и ускоряет всё.

Сама оценка довольно проста: каждый батч прогоняется через модель, определяется класс с наивысшим баллом через `output.max()`, подсчитывается, сколько предсказаний совпало с истинными метками, и в итоге возвращается процент accuracy. Здесь нет ни оптимизатора, ни отслеживания loss, ни обновления весов — только подсчёт правильных ответов.

## Собираем всё вместе: полный training loop

```python
num_epochs = 10
for epoch in range(num_epochs):
    print(f'\nEpoch: {epoch+1}')
    train_epoch(model, train_loader, loss_function, optimizer, device)
    accuracy = evaluate(model, test_loader, device)
    print(f'Test Accuracy:{accuracy:.2f}%')
```
Модель обучается на протяжении 10 эпох (epochs) — это 10 полных проходов по всему обучающему датасету. Но это не просто повторение: с каждым проходом модель уточняет своё понимание того, что отличает, например, двойку от семёрки.

После каждой обучающей эпохи модель оценивается на тестовом наборе, чтобы увидеть, насколько хорошо она работает на невиданных данных. Это показывает, действительно ли модель учит обобщаемые (generalizable) закономерности, или же она просто запоминает обучающие данные.

К 10-й эпохе loss становится совсем маленьким, а accuracy — высокой. Когда accuracy перестаёт улучшаться, это часто признак того, что модель на данный момент закончила обучение, поэтому все 10 эпох могут даже не понадобиться.

## Итог
Теперь всё готово, чтобы обучить свой первый классификатор изображений на PyTorch — далее предлагается перейти к лабораторной работе (lab) и попробовать это на практике.
