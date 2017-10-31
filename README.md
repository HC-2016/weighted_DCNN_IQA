# weighted DCNN

### Introduction
This is a tensorflow implementation of the network in "Bosse S, Maniry D, Müller K R, et al. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment[J]. arXiv preprint arXiv:1612.01697, 2016." .
We only reproduce the weighted FR network here. Refer to [dmaniry/deepIQA][source code] for official source codes.

### Dependencies
- Python 3.5
- TensorFlow 1.0

### Results
We repeat 10 times on tid2013.

| val img | test img | best epoch | train loss|  val metric | test metric |
|:-----:|:-----:|:-----:|:-----:|:-----:| :-----:|
| 7 13 5 24 6 | 18 0 11 1 15 | 2887 | 0.190 | 0.338/0.923/0.770/0.931/0.475 | 0.324/0.936/0.7850.939/0.434 |
| 17 10 16  3  6 | 11 21  5 22 24 | 2269 | 0.361 | 0.291/0.944/0.799/0.946/0.416 | 0.461/0.868/0.691/0.869/0.635 |
| 17  3 21 14  2 | 19  8 20 16  4 | 599 | 0.337 | 0.312/0.938/0.790/0.940/0.426 | 0.292/0.936/0.794/0.946/0.418 |
| 11 21 10 6 22 | 8 15 16 17 1| 2934 | 0.133 | 0.287/0.932/0.787/0.945/0.399 | 0.330/0.926/0.774/0.934/0.447 |
| 10 19 12 23 1 | 8  6  3  7 16 | 1806 | 0.452 | 0.307/0.927/0.776/0.933/0.430 | 0.374/0.910/0.751/0.918/0.501 |
| 1 15 17 12 21 | 3 9 0 2 4 | 2800 | 0.120 | 0.327/0.925/0.772/0.926/0.460 | 0.354/0.930/0.771/0.931/0.461 |
| 14 20 23 21 19 | 2 16 17 3 15| 1880 | 0.621 | 0.312/0.934/0.784/0.940/0.418 | 0.422/0.907/0.753/0.920/0.532 |
| 11 1 5 9 14 | 2 13 18 6 22| 2698 | 0.082 | 0.301/0.935/0.787/0.946/0.404 | 0.317/0.935/0.790/0.939/0.443 |


Note:
- the reference images and the epoch are indexed from 0.
- the metric  includes  MAE/SROCC/KROCC/PLCC/RMSE.

### TODO
- [ ] complement all experiments.

[source code]: https://github.com/dmaniry/deepIQA
