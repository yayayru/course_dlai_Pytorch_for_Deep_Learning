Если речь именно о **deep learning**, то сегодня я бы сравнивал четыре основных варианта: **PyTorch, TensorFlow, JAX и Keras 3**. Причём Keras теперь стоит рассматривать не как «обёртку над TensorFlow»: Keras 3 умеет работать поверх **PyTorch, TensorFlow и JAX**, а для inference также поддерживает OpenVINO. ([Keras][1])

| Критерий                   | PyTorch          | TensorFlow          | JAX               | Keras 3              |
| -------------------------- | ---------------- | ------------------- | ----------------- | -------------------- |
| Порог входа                | ⭐⭐⭐⭐⭐            | ⭐⭐⭐                 | ⭐⭐⭐               | ⭐⭐⭐⭐⭐                |
| Гибкость кастомных моделей | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐                | ⭐⭐⭐⭐⭐             | ⭐⭐⭐⭐                 |
| Research / эксперименты    | **Отлично**      | Хорошо              | **Отлично**       | Хорошо               |
| LLM / GenAI                | **Отлично**      | Средне              | Отлично           | Хорошо               |
| Production                 | **Отлично**      | **Отлично**         | Хорошо            | Зависит от backend   |
| GPU                        | **Отлично**      | Отлично             | **Отлично**       | Зависит от backend   |
| TPU                        | Хорошо           | **Отлично**         | **Отлично**       | Отлично с JAX/TF     |
| Distributed training       | **Очень развит** | Очень развит        | **Очень мощный**  | Абстрагирует backend |
| Mobile / browser           | Нормально        | **Сильная сторона** | Слабо             | Через TF/backend     |
| Debugging                  | **Удобно**       | Средне              | Сложнее из-за JIT | Удобно               |
| «Похож на обычный Python»  | **Да**           | Меньше              | Не совсем         | Да                   |

### PyTorch

Для большинства новых ML/DL-проектов это самый безопасный старт.

Код обычно выглядит естественно для Python-разработчика: определил `nn.Module`, написал `forward`, loss, вызвал `backward()`. При этом современный PyTorch уже далеко не только eager execution: `torch.compile` компилирует код через стек TorchDynamo/Inductor и предназначен как для training, так и inference. ([PyTorch Documentation][2])

Для больших моделей есть DDP, FSDP2, Tensor Parallel, Device Mesh и другие механизмы распределённого обучения. ([PyTorch Documentation][3])

**Выбирал бы PyTorch для:** LLM, CV, NLP, diffusion, fine-tuning, исследований, собственного нестандартного training loop.

### TensorFlow

TensorFlow по-прежнему мощный, но его главное преимущество сейчас скорее **целая production-экосистема**, чем удобство написания исследовательского кода.

Есть TensorFlow Serving для production serving и TensorFlow.js для запуска и даже обучения моделей непосредственно в браузере/Node.js. ([TensorFlow][4])

Высокоуровневый код обычно пишется через Keras, который TensorFlow сам позиционирует как основной high-level API. ([TensorFlow][5])

Поэтому TensorFlow особенно логичен, если у компании уже существует большая TF-инфраструктура или нужен тесный путь к web/mobile/существующему TF production stack.

### JAX

JAX интереснее всего, когда важны **математика, компиляция и максимальный контроль над вычислениями**.

Его философия примерно такая:

`NumPy + autodiff + transformations + XLA/JIT`

Вместо объектно-ориентированного подхода PyTorch ты часто работаешь с функциями:

```python
f = jax.jit(f)
grad_f = jax.grad(f)
batch_f = jax.vmap(f)
```

`jax.jit` компилирует последовательность операций, а JAX имеет продвинутую модель распределения массивов и вычислений между устройствами. Современная документация также рекомендует `shard_map` вместо старого `pmap` для более сложного multi-device parallelism. ([JAX Documentation][6])

Цена — более высокая когнитивная сложность: immutable arrays, tracing, static shapes/JIT restrictions и функциональный стиль.

**JAX особенно хорош:** большие TPU-кластеры, research, научные вычисления, custom numerical algorithms, проекты, где compiler transformations являются частью архитектуры.

### Keras 3 — интересный компромисс

Сейчас Keras стал гораздо интереснее.

Можно написать:

```python
import keras

model = keras.Sequential([
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(10),
])
```

а backend выбрать отдельно:

```bash
KERAS_BACKEND=torch
```

или:

```bash
KERAS_BACKEND=jax
```

или:

```bash
KERAS_BACKEND=tensorflow
```

Официально поддерживаются все три backend'а. ([Keras][7])

То есть можно получить простоту API Keras, но не привязываться жёстко к TensorFlow. Это особенно интересно для обучения, prototyping и проектов, где нужна переносимость.

### А scikit-learn / XGBoost?

Они немного из другой категории.

Если задача — **табличные данные**, то нейросеть вообще часто не является первым инструментом. Я бы сначала смотрел на:

**XGBoost / LightGBM / CatBoost → scikit-learn → и только затем PyTorch/TensorFlow**, если для нейросети действительно есть основания.

А для deep learning выбор в большинстве случаев сводится к следующему:

* **Хочу просто выбрать один framework → PyTorch.**
* **Нужен существующий TensorFlow production stack / web / специфичная TF-инфраструктура → TensorFlow.**
* **Нужны TPU, compiler-first подход, sophisticated sharding или research-level numerical computing → JAX.**
* **Хочу максимально простой high-level API и возможность переключать PyTorch/JAX/TF → Keras 3.**

Если бы я начинал изучать deep learning **с нуля в 2026 году**, последовательность была бы: **PyTorch → Hugging Face → distributed PyTorch → затем JAX**, а TensorFlow изучал бы уже при наличии конкретной рабочей необходимости.

[1]: https://keras.io/keras_3/?utm_source=chatgpt.com "Keras: Deep Learning for humans"
[2]: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial?utm_source=chatgpt.com "Introduction to torch.compile — PyTorch Tutorials 2.13.0+cu130 documentation"
[3]: https://docs.pytorch.org/tutorials/distributed.html?utm_source=chatgpt.com "Distributed — PyTorch Tutorials 2.13.0+cu130 documentation"
[4]: https://www.tensorflow.org/tfx/serving/serving_config?utm_source=chatgpt.com "Tensorflow Serving Configuration  |  TFX  |  TensorFlow"
[5]: https://www.tensorflow.org/guide/keras?utm_source=chatgpt.com "Keras: The high-level API for TensorFlow  |  TensorFlow Core"
[6]: https://docs.jax.dev/en/latest/quickstart.html?utm_source=chatgpt.com "Quickstart: How to think in JAX — JAX documentation"
[7]: https://keras.io/getting_started/?utm_source=chatgpt.com "Getting started with Keras"

№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№

Да. И здесь есть интересная закономерность: у современных frontier-LLM **PyTorch и JAX заметно важнее TensorFlow**.

| Сервис / семейство           | Основной публично известный ML-стек                                      | Уверенность |
| ---------------------------- | ------------------------------------------------------------------------ | ----------: |
| **ChatGPT / OpenAI GPT**     | **PyTorch + собственные CUDA/Triton-компоненты**                         |       🟢/🟡 |
| **Claude / Anthropic**       | **PyTorch + JAX + Triton**                                               |          🟢 |
| **Gemini / Google DeepMind** | **JAX + ML Pathways + TPU**                                              |          🟢 |
| **Grok / xAI**               | **JAX + собственная distributed-инфраструктура**                         |          🟢 |
| **Meta AI / Llama**          | **PyTorch**                                                              |          🟢 |
| **Mistral / Le Chat**        | PyTorch в публичном inference-стеке; training stack полностью не раскрыт |          🟡 |
| **Perplexity**               | Нет одного: использует модели разных поставщиков + собственные модели    |           — |
| **Microsoft Copilot**        | Нет одного: OpenAI-модели + модели Microsoft и инфраструктура Azure      |           — |

### ChatGPT — в основном PyTorch

OpenAI ещё в 2020 году официально объявила, что стандартизирует deep-learning research на **PyTorch**, потому что он позволял быстрее проводить эксперименты и хорошо масштабировался на GPU. При этом OpenAI сразу оговаривала, что при технической необходимости может применять и другие инструменты. ([OpenAI][1])

Поэтому упрощённо можно представить стек так:

```text
ChatGPT
   ↓
GPT
   ↓
PyTorch
   ↓
Triton / custom kernels / CUDA
   ↓
GPU clusters
```

Но тут есть важный нюанс: **современный training stack OpenAI целиком публично не описан**. Нельзя уверенно утверждать, что актуальные GPT-модели — это просто большой `torch.nn.Module`. На масштабе frontier-моделей поверх фреймворка появляются собственные runtime, distributed training, communication libraries, kernels, checkpointing и inference infrastructure.

---

### Claude — интереснее: PyTorch + JAX

У Anthropic есть гораздо более свежее публичное подтверждение. В Transparency Hub для современных Claude Anthropic прямо указывает среди development frameworks:

**PyTorch, JAX и Triton**, а вычислительную инфраструктуру — AWS и Google Cloud. ([Anthropic][2])

То есть Claude нельзя корректно описать просто как «модель на PyTorch».

Скорее:

```text
Claude
       │
   ┌───┴────┐
PyTorch    JAX
   │         │
   └── Triton/custom kernels
             │
        GPU / TPU infrastructure
```

Причём **Triton** здесь — язык/компилятор для высокопроизводительных GPU kernels, а не TensorFlow.

---

### Gemini — яркий пример JAX

Google DeepMind — главный пример того, зачем вообще JAX нужен в эпоху PyTorch.

В техническом отчёте Gemini 1.0 Google прямо указала:

```text
Software:
JAX
ML Pathways

Hardware:
TPUv4
TPUv5e
```

И описала JAX + Pathways как единый механизм оркестрации огромного distributed training. ([Google DeepMind][3])

Поэтому архитектурно экосистема Google выглядит примерно так:

```text
Gemini
   ↓
JAX
   ↓
XLA
   ↓
ML Pathways
   ↓
TPU pods
```

DeepMind ещё до Gemini широко перешла на JAX для research и разработала вокруг него библиотеки вроде Haiku и Optax. ([Google DeepMind][4])

Именно поэтому я говорил в прошлом ответе, что **JAX особенно важен для TPU и очень большого distributed training**.

---

### Grok — тоже JAX

Здесь подтверждение очень прямое. xAI пишет, что для создания Grok построила собственный training и inference stack на базе:

> Kubernetes, Rust и JAX. ([SpaceXAI][5])

А открытая версия Grok-1 также опубликована с JAX-кодом; там используются JAX, Haiku и JAX sharding primitives. ([GitHub][6])

То есть:

```text
Grok
 ↓
JAX
 ↓
custom distributed stack
 ↓
tens of thousands of GPUs
```

Любопытный момент: JAX вовсе не означает «только TPU». xAI использует его на огромных **GPU-кластерах**.

---

### Llama / Meta AI — PyTorch

Здесь связь максимально естественная: **PyTorch появился внутри Meta/Facebook**.

Официальные Llama-репозитории используют PyTorch, а Meta выпускает `torchtune` как PyTorch-native библиотеку для обучения и fine-tuning LLM. ([AI Meta][7])

Типичный стек получается:

```text
Llama
 ↓
PyTorch
 ↓
TorchTune / FSDP / custom kernels
 ↓
CUDA
 ↓
NVIDIA GPUs
```

Meta также работает напрямую с PyTorch, vLLM и TensorRT для production-инфраструктуры Llama. ([AI Meta][8])

---

### Mistral

Здесь надо быть осторожнее.

Публичная reference implementation Mistral написана на **PyTorch**:

```python
import torch
import torch.nn as nn
```

и для multi-GPU запуска использовала `torchrun`. ([GitHub][9])

Но это подтверждает прежде всего публичный **inference/model implementation**. Из этого нельзя автоматически заключить, что внутреннее обучение всех моделей Le Chat/Mistral сейчас выполняется исключительно через стандартный PyTorch.

Поэтому я бы обозначил Mistral как **«PyTorch-ориентированный стек, внутренности обучения частично proprietary»**.

### А где TensorFlow?

Вот это, на мой взгляд, самое показательное.

Если очень грубо посмотреть на frontier LLM:

```text
              FRONTIER LLM
                   │
          ┌────────┴────────┐
          │                 │
       PyTorch             JAX
          │                 │
 OpenAI* / Meta       Google / xAI
 Mistral* / Anthropic  Anthropic
          │                 │
         GPU             GPU/TPU
```

`*` — с оговорками о закрытой внутренней инфраструктуре.

**TensorFlow в этом списке практически не виден.**

Это не означает, что TensorFlow умер. Он остаётся огромной платформой, особенно для существующих production-систем, inference, enterprise ML, mobile/web и огромного количества классических ML/DL pipeline.

Но для **обучения новых frontier LLM** индустрия фактически разделилась преимущественно на два лагеря:

**PyTorch:** более Pythonic, гибкий, огромная CUDA/NVIDIA и open-source LLM ecosystem.

**JAX:** compiler-first, XLA, очень мощный model/data sharding, TPU и огромные distributed workloads.

Именно поэтому, если твоя цель — понять технологии, которые лежат **под ChatGPT/Claude/Llama/Gemini**, я бы сейчас расставил приоритет обучения так:

**1. PyTorch → 2. CUDA/Triton основы → 3. distributed training → 4. JAX → 5. TensorFlow при конкретной необходимости.**

Это уже гораздо ближе к реальному стеку современных AI labs, чем классическая последовательность «TensorFlow → Keras → PyTorch».

[1]: https://openai.com/index/openai-pytorch/?utm_source=chatgpt.com "OpenAI standardizes on PyTorch | OpenAI"
[2]: https://www.anthropic.com/transparency/model-report?utm_source=chatgpt.com "Anthropic’s Transparency Hub \ Anthropic"
[3]: https://deepmind.google/gemini/gemini_1_report.pdf?utm_source=chatgpt.com "Gemini: A Family of Highly Capable"
[4]: https://deepmind.google/blog/using-jax-to-accelerate-our-research/?utm_source=chatgpt.com "Using JAX to accelerate our research — Google DeepMind"
[5]: https://x.ai/blog/grok?utm_source=chatgpt.com "Announcing Grok | xAI"
[6]: https://github.com/xai-org/grok-1/blob/main/README.md?utm_source=chatgpt.com "grok-1/README.md at main · xai-org/grok-1 · GitHub"
[7]: https://ai.meta.com/blog/meta-llama-3/?utm_source=chatgpt.com "Introducing Meta Llama 3: The most capable openly available LLM to date"
[8]: https://ai.meta.com/blog/meta-llama-3-1/?utm_source=chatgpt.com "Introducing Llama 3.1: Our most capable models to date"
[9]: https://github.com/mistralai/mistral-inference?utm_source=chatgpt.com "GitHub - mistralai/mistral-inference: Official inference library for Mistral models · GitHub"
