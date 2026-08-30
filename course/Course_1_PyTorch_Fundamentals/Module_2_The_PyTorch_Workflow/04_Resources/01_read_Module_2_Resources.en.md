# Module 2 Lecture Slides
The lecture notes are available on our community platform. If you’re already a member, log in to your account and access the lecture notes [here](https://community.deeplearning.ai/t/pytorch-for-deep-learning-course-1-lecture-notes/882257).

**NOTE:** If you don’t have an account yet, kindly follow the instructions [here](https://learn.deeplearning.ai/courses/pytorch-fundamentals/lesson/9ii6mh/join-the-deeplearning.ai-forum-to-ask-questions,-get-support,-or-share-amazing-ideas!), then come back to this page.

# Module 2 Resources
## ML cycle (end-to-end, PyTorch-agnostic)
- Full Stack Deep Learning (FSDL) — ML Projects / Lifecycle. Practical notes + slides on scoping, baselines, metrics, data, debugging, eval. [fullstackdeeplearning.com](fullstackdeeplearning.com) [fall2019.fullstackdeeplearning.com](fall2019.fullstackdeeplearning.com)
- Made With ML (Goku Mohandas) — MLOps course. Step-by-step pipeline: data → training → eval → serving → testing/monitoring. Great, modern, code-first. [madewithml.com](madewithml.com) [madewithlm.com/evaluation](madewithlm.com/evaluation)

## Deep Learning Fundamentals: Loss functions, Optimizers & gradients
- Understanding Deep Learning (Prince) — Loss notebooks — quick Colab-ready demos for least squares, binary/multiclass CE. [udlbook.github.io/udlbook/](udlbook.github.io/udlbook/)
- Ruder — An overview of gradient descent optimization algorithms — momentum, AdaGrad, RMSProp, Adam; great mental models. [ruder.io]

## MNIST family (original sources)
- MNIST — LeCun's page. Canonical dataset description, formats, splits, and downloads for handwritten digits. Link
- Fashion-MNIST — ZalandoResearch GitHub. A harder, drop-in MNIST replacement of clothing images with the same split structure. [Link](https://yann.lecun.org/exdb/mnist/index.html)
- EMNIST — NIST dataset page. MNIST extended to letters and digits, with multiple splits and download details. [Link](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- KMNIST — ROIS-CODH (site + GitHub). Kuzushiji (classical Japanese) characters in MNIST-like format for classification benchmarks. [Link](https://codh.rois.ac.jp/kmnist/index.html.en) [GitHub](https://github.com/rois-codh/kmnist)

## Python classes (light OOP refresher for nn.Module-style code)
- Real Python — Python Classes: OOP — quick, readable refresher (init, inheritance, super(), etc.). [Real Python](https://realpython.com/python-classes)
- DataQuest — How to Use dataclasses — modern, lightweight class patterns you'll use in configs. [Dataquest](https://www.dataquest.io/blog/how-to-use-python-data-classes)
- Fluent Python (2e) — Free Chapter: Python Data Model — dunder methods & how Python objects really work (excellent for mental model). [Thoughtworks](https://www.thoughtworks.com/en-us/insights/books/fluent-python-2nd-edition)

## Computational graphs & autograd (what .backward() really does)
- Karpathy — micrograd (repo + video) — tiny reverse-mode autodiff engine; read code then watch "spelled-out intro." [GitHub](https://github.com/karpathy/micrograd) [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Nielsen — How the backpropagation algorithm works — intuitive, math-first chapter.[Neural Networks and Deep Learning](https://neuralnetworksanddeeplearning.com/chap2.html)
- Parr & Howard — The Matrix Calculus You Need for Deep Learning — superb bridge from calculus to backprop. [ArXiv](https://arxiv.org/pdf/1802.01528)
