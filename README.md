# weighted DCNN

### Introduction
This is a tensorflow implementation of the network in "Bosse S, Maniry D, Müller K R, et al. Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment[J]. arXiv preprint arXiv:1612.01697, 2016." .
We only reproduce the weighted FR network here. Refer to [dmaniry/deepIQA][source code] for official source codes.

### Dependencies
For the code in src
- Python 3.5
- TensorFlow 1.0
- Anaconda3
- Windows 10

For the code in src_tid2013_cluster
- Python 2.7
- TensorFlow 1.0
- CentOS 6.5

### Results
We repeat 20 times on tid2013.

![enter image description here](https://github.com/HC-2016/weighted_DCNN_IQA/raw/master/result/result.png)



Note:
- the reference images and the epoch are indexed from 0.
- the metric  includes  MAE/SROCC/KROCC/PLCC/RMSE.

### TODO
- [x] complement the experiments on tid2013.

[source code]: https://github.com/dmaniry/deepIQA
