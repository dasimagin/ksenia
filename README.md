# Ksenia (Ksenia stores externally now it all)

### Oleges plans
* NTM version with RL:
   - [ ] Discrete attention schemes from [D-NTM](https://github.com/caglar/dntm)
   - [ ] Combine dicreсrete and traditional learning like in mentioned above paper
   - [ ] Q-Learning and (maybe) Watkins Q-dynamic from [Learning simple algorithms from examples](https://github.com/wojzaremba/algorithm-learning), paper can be [found](https://arxiv.org/abs/1511.07275)
   - [ ] Penalty on Q-function (from same paper)
   - [ ] Compare on different datasets
* Update NTM with additional critic based on different models:
   - [ ] Understand Recurrent Deterministic Policy Gradient from [Memory-based control with recurrent neural networks](https://paperswithcode.com/paper/memory-based-control-with-recurrent-neural)
   - [ ] Try convolutional critic based on memory
   - [ ] Try different learning schemes:
     - [ ] Q-learning
     - [ ] Watkins Q-dynamic
     - [ ] Penalty on Q-function
   - [ ] Compare on different datasets
* Implement some more experimets:
  - [ ] Convex Hull of set of point [link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html)
* Read papers:
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
* DNC architecture comparisons:
  Plan is to implement all dnc
  - [ ] Implement DNC in pytorch with different modifications:
        modules ([dnc](https://github.com/xdever/dnc) as a reference)
    - [ ] Key-Value content based attention, deallocation, and sharpening for
          temporal link matrix from [IMPROVING DIFFERENTIABLE NEURAL COMPUTERS
          THROUGH MEMORY MASKING, DEALLOCATION, AND LINK DISTRIBUTION SHARPNESS
          CONTROL](https://openreview.net/pdf?id=HyGEM3C9KQ)
    - [ ] Differentiable memory allocation from [Differentiable Memory
          Allocation Mechanism For Neural
          Computing](https://ttic.uchicago.edu/~klivescu/MLSLP2017/MLSLP2017_ben-ari.pdf)
    - [ ] Bidirectional, dropout and other architectural tweaks from [Robust and
          Scalable Differentiable Neural Computer for Question
          Answering](https://arxiv.org/pdf/1807.02658.pdf)
* Experiments:
  - [ ] *Approximate dedline: 15 april:* Simple bitmap tasks (test on all variant)
    - [ ] Copy task (as a test)
    - [ ] Repeat copy
    - [ ] Repeated copy
    - [ ] Associative recall
  - [ ] *After everything else is ready:* Question answering on babi
  - [ ] One-shot learning on omniglot (as in
        [paper](https://deepmind.com/research/publications/one-shot-learning-memory-augmented-neural-networks/))
* Additional ideas:
  - [ ] Use gumbel softmax instead of softmax in soft attention (and use argmax
        for inference)
  - [ ] Plot loss landscape (maybe, but need to read about it first)
  - [ ] Memory vizualisations
    - [ ] Memory contents + attention distribution
    - [ ] Temporal link matrix
    - [ ] Gates for read modes
    - [ ] Visualize memorization (maybe, bet need to read about [this](https://distill.pub/2019/memorization-in-rnns/))
