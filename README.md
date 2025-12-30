# Forward-Forward (FF)

Based on Geoffrey Hinton's Forward-Forward algorithm. Instead of backpropagation, each layer is trained to distinguish between "positive" data (real MNIST digits with correct labels) and "negative" data (digits with incorrect labels).

For a comparison with Hyperdimensional/Binary computing approaches (HDC), see [Engram](https://github.com/jesper-olsen/engram).

## Getting Started

Clone the repository:

```sh
git clone https://github.com/jesper-olsen/forward-forward-rs.git
cd forward-forward-rs
```

[Download](https://github.com/jesper-olsen/mnist-rs) the MNIST dataset.

Run the models:

```bash
cargo run --bin ff --release
```

Sample output:
```text
Training Forward-Forward Model...
Epoch   0 | Cost:  86.0361
Epoch   1 | Cost:  39.5928
Epoch   2 | Cost:  29.8963
Epoch   3 | Cost:  25.5904
Epoch   4 | Cost:  21.5753
Epoch   5 | Cost:  19.1607 | Err Train: (1817/50000), Err Val: (359/10000)
...[snip]...
Epoch 196 | Cost:   2.3645
Epoch 197 | Cost:   2.2378
Epoch 198 | Cost:   2.3523
Epoch 199 | Cost:   2.2710 | Err Train: (97/50000), Err Val: (85/10000)
Test Errors: (71/10000)
Train Errors: (182/60000)
Calculating confusion matrix...

Confusion Matrix (Actual vs Predicted):

Actual |  P0    P1    P2    P3    P4    P5    P6    P7    P8    P9
-------|------------------------------------------------------------
  A0   |  976     .     1     .     .     .     2     1     .     .
  A1   |    .  1133     2     .     .     .     .     .     .     .
  A2   |    .     2  1022     .     1     .     .     4     3     .
  A3   |    .     .     .  1006     .     3     .     .     .     1
  A4   |    .     1     1     .   976     .     2     .     .     2
  A5   |    1     .     .     4     .   886     1     .     .     .
  A6   |    3     2     .     .     .     1   952     .     .     .
  A7   |    .     1     6     1     .     1     .  1015     2     2
  A8   |    1     .     1     .     .     1     .     1   967     3
  A9   |    .     2     .     .     4     3     .     3     1   996
--------------------------------------------------------------------
```

## Overview
This model achieves near-state-of-the-art performance (99.29%) for MNIST without ever calculating a global gradient or using a chain rule.

### Key Techniques

* **Architecture**: 4 layers [784, 1000, 1000, 1000] using Rectified Linear Units (ReLU).

* **Local Objective**: Each layer independently maximises "goodness" (sum of squared activations) for positive data and minimises it for negative data, avoiding the need for a global backward pass.

* **Symmetric Augmentation**: Training samples are augmented by randomly shifting images up/down and left/right by one pixel. 

* **Regularization**: 10% Dropout and Weight Decay (0.002) to prevent the model from memorising specific training pixels.

* **Supervised Head**: A softmax layer sits on top of the normalised hidden states, accumulating scores from all supervised layers to provide the final classification.

### Results 

|               | Test Accuracy | Errors Test | Epochs
|--------------:|--------------:|------------:|-------:
| Train on 50k  | 99.29%        | 71 / 10,000 | 200
| Train on 60k  |   .  %        |    / 10,000 | 200

## References

* [The Forward-Forward Algorithm: Some Preliminary Investigations, Geoffrey Hinton, NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf) <br/>
* [Hinton's NIPS'22 Talk](https://nips.cc/virtual/2022/invited-talk/55869) <br/>
* [Hinton's matlab code](https://www.cs.toronto.edu/~hinton/ffcode.zip) <br/>

