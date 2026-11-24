# Forward-Forward (FF)

Rust implementation of Geoffrey Hinton's Forward-Forward algorithm [1]. The model is evaluated on MNIST and Fashion MNIST.

On MNIST it achieves an error rate of 65/10,000 = 0.65% compared to the papers 1.40% baseline - achieved mainly by adding
dropout, data augmentation and ensemble modelling to the training mix.

On Fashion-MNIST it achieves 89.5% accuracy, which is competitive with standard backpropagation for non-convolutional networks.

For a comparison with Hyperdimensional/Binary computing approaches (HDC), see [Engram](https://github.com/jesper-olsen/engram).

## Getting Started

Clone the repository:

```sh
git clone https://github.com/jesper-olsen/forward-forward-rs.git
cd forward-forward-rs
```

[Download](https://github.com/jesper-olsen/mnist-rs) the MNIST dataset.

Run the program:

```bash
cargo run --bin ff --release -- --dir MNIST
```

<details>
<summary>Sample output</summary>

```text
Rayon using default thread count (logical cores)
Training Forward-Forward Model
Config { threads: 0, model: None, dir: "MNIST", seed: 1234, dropout: 0.15, lambda_mean: 0.03, batch_size: 100, weight_decay: 0.002, sup_weight_decay: 0.003, momentum: 0.9, epsilon: 0.01, epsilon_sup: 0.1, label_strength: 1.0, temperature: 1.0, max_epoch: 200 }
Epoch   0 | Cost:  75.3137
Epoch   1 | Cost:  41.2214
Epoch   2 | Cost:  31.6026
Epoch   3 | Cost:  28.5548
Epoch   4 | Cost:  23.7960
Epoch   5 | Cost:  21.5926 | Err Train: (1784/50000), Err Val: (374/10000)
Epoch   6 | Cost:  19.7960
Epoch   7 | Cost:  17.8721
Epoch   8 | Cost:  17.6805
Epoch   9 | Cost:  15.9694
Epoch  10 | Cost:  15.6569 | Err Train: (1314/50000), Err Val: (279/10000)
Epoch  11 | Cost:  14.7362
Epoch  12 | Cost:  14.4912
Epoch  13 | Cost:  14.1683
Epoch  14 | Cost:  13.3812
Epoch  15 | Cost:  13.3085 | Err Train: (1132/50000), Err Val: (263/10000)
Epoch  16 | Cost:  12.2248
Epoch  17 | Cost:  12.0405
Epoch  18 | Cost:  11.3864
Epoch  19 | Cost:  11.8272
Epoch  20 | Cost:  11.0848 | Err Train: (1046/50000), Err Val: (241/10000)
Epoch  21 | Cost:  11.2434
Epoch  22 | Cost:  11.1427
Epoch  23 | Cost:  10.6964
Epoch  24 | Cost:  10.5967
Epoch  25 | Cost:  10.2273 | Err Train: (880/50000), Err Val: (224/10000)
Epoch  26 | Cost:  10.1467
Epoch  27 | Cost:   9.6964
Epoch  28 | Cost:   9.4676
Epoch  29 | Cost:   9.6553
Epoch  30 | Cost:   9.4068 | Err Train: (756/50000), Err Val: (202/10000)
Epoch  31 | Cost:   9.2629
Epoch  32 | Cost:   9.3529
Epoch  33 | Cost:   9.5264
Epoch  34 | Cost:   9.1246
Epoch  35 | Cost:   9.0272 | Err Train: (613/50000), Err Val: (191/10000)
Epoch  36 | Cost:   8.8506
Epoch  37 | Cost:   9.0465
Epoch  38 | Cost:   8.6917
Epoch  39 | Cost:   8.4538
Epoch  40 | Cost:   8.7463 | Err Train: (606/50000), Err Val: (171/10000)
Epoch  41 | Cost:   8.6206
Epoch  42 | Cost:   8.5054
Epoch  43 | Cost:   8.4740
Epoch  44 | Cost:   8.2155
Epoch  45 | Cost:   7.7900 | Err Train: (586/50000), Err Val: (180/10000)
Epoch  46 | Cost:   8.0504
Epoch  47 | Cost:   7.6681
Epoch  48 | Cost:   7.6372
Epoch  49 | Cost:   7.7239
Epoch  50 | Cost:   8.0996 | Err Train: (539/50000), Err Val: (161/10000)
Epoch  51 | Cost:   7.9021
Epoch  52 | Cost:   7.8442
Epoch  53 | Cost:   7.8295
Epoch  54 | Cost:   7.6461
Epoch  55 | Cost:   7.7414 | Err Train: (483/50000), Err Val: (151/10000)
Epoch  56 | Cost:   7.5961
Epoch  57 | Cost:   7.4150
Epoch  58 | Cost:   7.4403
Epoch  59 | Cost:   7.3059
Epoch  60 | Cost:   7.3545 | Err Train: (454/50000), Err Val: (150/10000)
Epoch  61 | Cost:   7.1188
Epoch  62 | Cost:   6.9659
Epoch  63 | Cost:   7.0306
Epoch  64 | Cost:   7.2122
Epoch  65 | Cost:   6.9616 | Err Train: (404/50000), Err Val: (133/10000)
Epoch  66 | Cost:   6.7325
Epoch  67 | Cost:   7.0221
Epoch  68 | Cost:   7.1175
Epoch  69 | Cost:   6.8114
Epoch  70 | Cost:   7.0174 | Err Train: (459/50000), Err Val: (148/10000)
Epoch  71 | Cost:   6.6218
Epoch  72 | Cost:   6.8120
Epoch  73 | Cost:   6.7989
Epoch  74 | Cost:   6.4660
Epoch  75 | Cost:   6.6984 | Err Train: (448/50000), Err Val: (160/10000)
Epoch  76 | Cost:   6.9285
Epoch  77 | Cost:   6.7398
Epoch  78 | Cost:   6.3609
Epoch  79 | Cost:   6.6373
Epoch  80 | Cost:   6.5464 | Err Train: (475/50000), Err Val: (155/10000)
Epoch  81 | Cost:   6.7182
Epoch  82 | Cost:   6.7682
Epoch  83 | Cost:   6.3914
Epoch  84 | Cost:   6.5800
Epoch  85 | Cost:   6.5381 | Err Train: (550/50000), Err Val: (170/10000)
Epoch  86 | Cost:   6.6454
Epoch  87 | Cost:   6.9496
Epoch  88 | Cost:   6.3471
Epoch  89 | Cost:   6.3602
Epoch  90 | Cost:   6.5189 | Err Train: (463/50000), Err Val: (158/10000)
Epoch  91 | Cost:   6.2575
Epoch  92 | Cost:   6.2123
Epoch  93 | Cost:   6.2552
Epoch  94 | Cost:   6.1805
Epoch  95 | Cost:   6.0254 | Err Train: (326/50000), Err Val: (132/10000)
Epoch  96 | Cost:   6.1805
Epoch  97 | Cost:   6.0596
Epoch  98 | Cost:   6.0142
Epoch  99 | Cost:   6.2868
Epoch 100 | Cost:   6.1953 | Err Train: (416/50000), Err Val: (160/10000)
Epoch 101 | Cost:   6.1110
Epoch 102 | Cost:   6.2033
Epoch 103 | Cost:   5.9858
Epoch 104 | Cost:   6.1351
Epoch 105 | Cost:   5.7957 | Err Train: (570/50000), Err Val: (179/10000)
Epoch 106 | Cost:   5.7949
Epoch 107 | Cost:   5.8469
Epoch 108 | Cost:   6.0738
Epoch 109 | Cost:   5.5776
Epoch 110 | Cost:   5.6831 | Err Train: (356/50000), Err Val: (116/10000)
Epoch 111 | Cost:   5.2523
Epoch 112 | Cost:   5.5775
Epoch 113 | Cost:   5.5117
Epoch 114 | Cost:   5.3062
Epoch 115 | Cost:   5.3215 | Err Train: (307/50000), Err Val: (129/10000)
Epoch 116 | Cost:   5.0228
Epoch 117 | Cost:   5.0612
Epoch 118 | Cost:   4.9904
Epoch 119 | Cost:   4.9332
Epoch 120 | Cost:   4.9974 | Err Train: (276/50000), Err Val: (128/10000)
Epoch 121 | Cost:   4.9554
Epoch 122 | Cost:   4.6484
Epoch 123 | Cost:   4.7509
Epoch 124 | Cost:   4.5316
Epoch 125 | Cost:   4.6006 | Err Train: (258/50000), Err Val: (127/10000)
Epoch 126 | Cost:   4.5365
Epoch 127 | Cost:   4.5575
Epoch 128 | Cost:   4.5139
Epoch 129 | Cost:   4.4578
Epoch 130 | Cost:   4.2983 | Err Train: (260/50000), Err Val: (132/10000)
Epoch 131 | Cost:   4.4998
Epoch 132 | Cost:   4.2637
Epoch 133 | Cost:   4.1415
Epoch 134 | Cost:   4.2987
Epoch 135 | Cost:   4.2186 | Err Train: (209/50000), Err Val: (122/10000)
Epoch 136 | Cost:   3.9810
Epoch 137 | Cost:   4.0182
Epoch 138 | Cost:   3.9813
Epoch 139 | Cost:   3.8016
Epoch 140 | Cost:   3.9653 | Err Train: (180/50000), Err Val: (107/10000)
Epoch 141 | Cost:   3.6950
Epoch 142 | Cost:   3.7394
Epoch 143 | Cost:   3.7272
Epoch 144 | Cost:   3.7599
Epoch 145 | Cost:   3.7448 | Err Train: (193/50000), Err Val: (109/10000)
Epoch 146 | Cost:   3.7047
Epoch 147 | Cost:   3.8045
Epoch 148 | Cost:   3.5139
Epoch 149 | Cost:   3.2879
Epoch 150 | Cost:   3.4777 | Err Train: (179/50000), Err Val: (116/10000)
Epoch 151 | Cost:   3.4770
Epoch 152 | Cost:   3.5877
Epoch 153 | Cost:   3.5555
Epoch 154 | Cost:   3.3157
Epoch 155 | Cost:   3.5225 | Err Train: (177/50000), Err Val: (115/10000)
Epoch 156 | Cost:   3.3094
Epoch 157 | Cost:   3.3169
Epoch 158 | Cost:   3.3549
Epoch 159 | Cost:   3.1808
Epoch 160 | Cost:   3.2925 | Err Train: (175/50000), Err Val: (108/10000)
Epoch 161 | Cost:   3.2442
Epoch 162 | Cost:   3.2861
Epoch 163 | Cost:   3.1294
Epoch 164 | Cost:   3.2210
Epoch 165 | Cost:   3.1697 | Err Train: (120/50000), Err Val: (102/10000)
Epoch 166 | Cost:   2.9769
Epoch 167 | Cost:   3.0443
Epoch 168 | Cost:   2.9500
Epoch 169 | Cost:   2.8603
Epoch 170 | Cost:   2.9291 | Err Train: (124/50000), Err Val: (103/10000)
Epoch 171 | Cost:   2.9128
Epoch 172 | Cost:   2.9230
Epoch 173 | Cost:   3.0179
Epoch 174 | Cost:   2.8399
Epoch 175 | Cost:   2.8829 | Err Train: (125/50000), Err Val: (99/10000)
Epoch 176 | Cost:   2.8061
Epoch 177 | Cost:   2.7793
Epoch 178 | Cost:   2.8793
Epoch 179 | Cost:   2.7279
Epoch 180 | Cost:   2.8580 | Err Train: (137/50000), Err Val: (92/10000)
Epoch 181 | Cost:   2.6960
Epoch 182 | Cost:   2.7862
Epoch 183 | Cost:   2.6665
Epoch 184 | Cost:   2.6030
Epoch 185 | Cost:   2.6382 | Err Train: (94/50000), Err Val: (98/10000)
Epoch 186 | Cost:   2.6602
Epoch 187 | Cost:   2.4575
Epoch 188 | Cost:   2.6329
Epoch 189 | Cost:   2.5532
Epoch 190 | Cost:   2.5938 | Err Train: (97/50000), Err Val: (95/10000)
Epoch 191 | Cost:   2.5012
Epoch 192 | Cost:   2.6531
Epoch 193 | Cost:   2.6051
Epoch 194 | Cost:   2.4373
Epoch 195 | Cost:   2.5153 | Err Train: (96/50000), Err Val: (89/10000)
Epoch 196 | Cost:   2.5265
Epoch 197 | Cost:   2.4250
Epoch 198 | Cost:   2.5217
Epoch 199 | Cost:   2.4576 | Err Train: (84/50000), Err Val: (92/10000)
TestMode: Softmax
Errors - Test: 68/10000, Val: 92/10000, Train: 84/50000
TestMode: Energy
Errors - Test: 91/10000, Val: 92/10000, Train: 134/50000
Calculating confusion matrix...

Confusion Matrix (Actual vs Predicted):

Actual |  P0    P1    P2    P3    P4    P5    P6    P7    P8    P9
-------|------------------------------------------------------------
  A0   |  978     .     .     .     .     .     2     .     .     .
  A1   |    .  1131     1     1     .     1     .     1     .     .
  A2   |    .     1  1023     .     2     .     1     2     2     1
  A3   |    .     .     .  1007     .     1     .     .     1     1
  A4   |    .     .     .     .   978     .     1     .     .     3
  A5   |    2     .     .     5     .   882     2     1     .     .
  A6   |    2     1     .     1     1     1   951     .     1     .
  A7   |    .     1     4     1     1     .     .  1018     1     2
  A8   |    .     .     1     1     .     2     .     2   967     1
  A9   |    .     2     .     .     6     3     1     .     .   997
--------------------------------------------------------------------
```

</details>

## Overview
This model achieves state-of-the-art performance (99.32%) for fully connected Forward-Forward networks, halving the error rate of the original paper's baseline implementation [1].

### Key Techniques

* **Architecture**: 4 layers [784, 1000, 1000, 1000] using Rectified Linear Units (ReLU).

* **Local Objective**: Each layer independently maximises "goodness" (sum of squared activations) for positive data and minimises it for negative data, avoiding the need for a global backward pass.

* **Symmetric Augmentation**: Training samples are augmented by randomly shifting images up/down and left/right by one pixel. 

* **Regularization**: 15% Dropout and Weight Decay (0.002) to prevent the model from memorising specific training pixels.

* **Supervised Head**: A softmax layer sits on top of the normalised hidden states, accumulating scores from all supervised layers to provide the final classification.

### Results - MNIST

|               | Test Accuracy | Errors Test | Epochs
|--------------:|--------------:|------------:|-------:
| Train on 50k  | 99.32%        | 68 / 10,000 | 200
| Train on 60k  | 99.34%        | 66 / 10,000 | 200

#### Ensemble and test-time augmentation

Test errors / 10,000 with the softmax readout, trained on 50k. Single models are mean ± sd over 5 seeds;
the ensemble sums the logits of the 5 models. TTA sums the scores over the 9 one-pixel shifts of each test image.

|                       | 200 epochs   | 500 epochs
|----------------------:|-------------:|-----------:
| Single model          | 74.6 ± 2.5   | 71.0 ± 5.9
| Single model + TTA    | 70.8 ± 3.7   | 66.8 ± 1.8
| 5-model ensemble      | 68           | 65
| 5-model ensemble + TTA| 66           | 65

```bash
cargo run --release --bin ff -- --seed 1 --out model_s1.bin     # repeat for seeds 2-5
cargo run --release --bin ff -- --tta --model model_s1.bin,model_s2.bin,model_s3.bin,model_s4.bin,model_s5.bin
```

The gains are modest (about 10%): the seeds make largely the same mistakes (5→3, 3→9, 7→2, 4→9), and three models
capture most of the ensemble gain. TTA helps less once the models are ensembled or trained for 500 epochs.

### Results - Fashion MNIST

|               | Test Accuracy |  Errors Test  | Epochs
|--------------:|--------------:|--------------:|-------:
| Train on 50k  | 89.31%        | 1069 / 10,000 | 200
| Train on 60k  | 89.52%        | 1048 / 10,000 | 200

## References

[1] [The Forward-Forward Algorithm: Some Preliminary Investigations, Geoffrey Hinton, NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf) <br/>
[2] [Hinton's NIPS'22 Talk](https://nips.cc/virtual/2022/invited-talk/55869) <br/>
[3] [Hinton's matlab code](https://www.cs.toronto.edu/~hinton/ffcode.zip) <br/>

