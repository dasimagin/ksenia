# Ksenia (Ksenia stores externally now it all)

### Oleges plans
* RL-LSTM for copy and repeat copy task of binary vector of bits, try method:
   - [ ] Q-Learning with Q-dinamic:
     - [ ] Watkins Q-dynamic from [Learning simple algorithms from examples](https://github.com/wojzaremba/algorithm-learning), paper can be [found](https://arxiv.org/abs/1511.07275)
     - [ ] Penalty on Q-function (from same paper)
   - [ ] Reinforce, maybe reinforce with baseline

* NTM version with RL:
   - [ ] Implement best scheme and compare D-NTM (simple argma in memory) and obvious NTM
   - [ ] Compare other schemes

* Update NTM with additional critic based on different models (future plans):
   - [ ] Understand Recurrent Deterministic Policy Gradient from [Memory-based control with recurrent neural networks](https://paperswithcode.com/paper/memory-based-control-with-recurrent-neural)
   - [ ] Try convolutional critic based on memory
   - [ ] Compare on different datasets

* Read papers:
  - [ ] [Learning Simple algorithms from examples](https://arxiv.org/pdf/1511.07275.pdf)
  - [ ] [Reinforcement learning NTM](https://arxiv.org/pdf/1505.00521.pdf)
  - [ ] [Memory-based control with recurrent neural networks](https://paperswithcode.com/paper/memory-based-control-with-recurrent-neural)
  - [ ] [GLOBAL-TO-LOCAL MEMORY POINTER NETWORKS FOR TASK-ORIENTED DIALOGUE](https://arxiv.org/pdf/1901.04713v1.pdf)
  - [ ] [Learning to Remember More with Less Memorization](https://arxiv.org/pdf/1901.01347.pdf)

### Nikita's plans
* Implement utils:
  - [ ] Implement abstract class for arithmetic task dataset
  - [ ] Implement memory visualization for further research
* Compare next arithmetic tasks for [NTM](https://arxiv.org/abs/1410.5401) and [DNC](https://www.nature.com/articles/nature20101)
  - [ ] a + b; a and b - binary numbers
  - [ ] a * b; a and b - binary numbers
  - [ ] a + b + c; a, b and c - binary numbers
  - [ ] a + b * c; a, b and c - binary numbers
  - [ ] a * b + c; a, b and c - binary numbers
  - [ ] Compare results of experiments
* Read papers:
  - [ ] [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508)

### Vanya's plans
#### DNC architecture comparisons:
Plan is to implement dnc with plugable modules
([dnc](https://github.com/xdever/dnc) as a reference), and test it's
performance on algorithmic tasks with different configurations.

* Modules:
  - [ ] Content based attention:
    - default;
    - key-value (with masks/and explicit) [1](https://openreview.net/pdf?id=HyGEM3C9KQ);
    - use gumbel softmax for weightings;
    - use location based addresing
  - [ ] Memory allocation mechanisms
    - default;
    - deallocation [1](https://openreview.net/pdf?id=HyGEM3C9KQ);
    - differentiable allocation [2](https://ttic.uchicago.edu/~klivescu/MLSLP2017/MLSLP2017_ben-ari.pdf);
  - [ ] Temporal link matrix
    - default;
    - sharpend [1](https://openreview.net/pdf?id=HyGEM3C9KQ)
    - without link matrix (but with location based addresing);
  - [ ] *Maybe:* bidirectional, dropout and other architectural tweaks from [3](https://arxiv.org/pdf/1807.02658.pdf) (experiments as a reference)
* Experiments:
  - [ ] *Approximate dedline: 15 april:* Simple bitmap tasks (test on all variant)
    - [ ] Copy task (as a test)
    - [ ] Repeat copy
    - [ ] Repeated copy
    - [ ] Associative recall (+ take a look at working memory tasks from [paper](https://arxiv.org/pdf/1809.11087v1.pdf))
    - [ ] Sorting/priority sorting (or another more complex algorithmic task)
* Memory vizualisations:
  - [ ] Memory contents + attention distribution
  - [ ] Temporal link matrix
  - [ ] Gates for read modes
  - [ ] Visualize memorization (maybe, bet need to read about [this](https://distill.pub/2019/memorization-in-rnns/))
* Additionalf ideas:
  - [ ] Plot loss landscape (maybe, but need to read about it first)
* Someday:
  - [ ] *After everything else is ready:* Question answering on babi
  - [ ] One-shot learning on omniglot (as in
        [paper](https://deepmind.com/research/publications/one-shot-learning-memory-augmented-neural-networks/))
